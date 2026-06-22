#!/usr/bin/env python3
"""
preprocess_triplets.py — TESS Transit Triplet Preprocessor

Phase-fold all available TESS sectors to get a high-SNR master batman fit,
then apply the fixed shape to each individual transit, extracting equal-duration
pre-baseline | in-transit | post-baseline triplets.

Pipeline:
  1. Load planets_ready_for_modeling.csv
  2. Select planet (--test = most estimated transits, or --planet NAME)
  3. Download TESS PDC-SAP via lightkurve, stitch & clean
  4. Phase-fold all data, bin, fit batman master model
     Free params: t0_offset, Rp/Rs, a/Rs, inclination, u1, u2 (LD)
     TEPCat values used as initial guesses
  5. For each transit (up to --max):
     - Extract triplet: pre (duration D) | in-transit (D) | post (D)
     - Fit only T0 + linear baseline per transit (shape fixed from master)
     - Compute residuals
  6. Save outputs:
     results/{planet}/phase_fold_fit.png        — master fit diagnostic
     results/{planet}/individual_transits_grid.png  — grid of individual fits
     results/{planet}/transit_fits.csv          — per-transit T0, rchi2

Usage:
    python preprocess_triplets.py --test
    python preprocess_triplets.py --planet "WASP-100"
    python preprocess_triplets.py --test --grid 24 --max 80
"""

import argparse
import ast
import os
import warnings
warnings.filterwarnings("ignore")

import batman
import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ── Paths ─────────────────────────────────────────────────────────────────────
CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "planets_ready_for_modeling.csv")
RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── Unit correction ───────────────────────────────────────────────────────────
# planet_filter.ipynb computed Rp_Rs = R_b / R_A, but TEPCat stores
# R_b in Jupiter radii and R_A in solar radii, so the ratio is off.
# Correct factor = R_Jup / R_sun = 71492 / 695700 = 0.10276
R_JUP_IN_RSUN = 71492.0 / 695700.0   # ≈ 0.10276

# ── Catalog helpers ───────────────────────────────────────────────────────────

def load_catalog():
    df = pd.read_csv(CATALOG_PATH)
    df["sector_list"] = df["sector_list"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
    )
    df["est_n_transits"] = (
        df["n_sectors"] * 27.4 / df["Period"].replace(0, np.nan)
    )
    return df.dropna(subset=["est_n_transits"])


def pick_planet(df, name=None):
    if name:
        # Try exact match (TEPCat uses underscores; user may give spaces)
        match = df[df["System"] == name]
        if len(match) == 0:
            match = df[df["System"] == name.replace(" ", "_")]
        if len(match) == 0:
            raise ValueError(f"Planet '{name}' not found. Check spelling against System column.")
        return match.iloc[0]
    return df.loc[df["est_n_transits"].idxmax()]


# ── Light curve download ──────────────────────────────────────────────────────

def download_lc(host_star):
    print(f"  Querying MAST for: {host_star}")
    results = lk.search_lightcurve(host_star, mission="TESS", exptime=120)
    if results is None or len(results) == 0:
        raise ValueError(f"No TESS 2-min PDC-SAP data found for '{host_star}'.")
    print(f"  Found {len(results)} sector entries. Downloading...")
    collection = results.download_all(quality_bitmask="default")
    lc = collection.stitch(corrector_func=lambda x: x.normalize())
    lc = lc.remove_nans()
    # Only clip UPWARD outliers (flares, cosmic rays) — NOT downward (those are transits)
    lc = lc.remove_outliers(sigma_lower=1e6, sigma_upper=4.0)
    span = lc.time.value[-1] - lc.time.value[0]
    print(f"  Light curve: {len(lc):,} points  span={span:.1f} d")
    return lc


def lc_arrays(lc):
    time = lc.time.value
    flux = lc.flux.value
    ferr = (lc.flux_err.value if hasattr(lc, "flux_err") and lc.flux_err is not None
            else np.ones_like(flux) * np.nanstd(flux) * 0.1)
    ferr = np.where(np.isfinite(ferr) & (ferr > 0), ferr, np.nanmedian(ferr[ferr > 0]))
    return time, flux, ferr


# ── Limb darkening initial guess (quadratic, Teff-based) ─────────────────────

def ld_guess(teff):
    table = [(7000, 0.20, 0.20), (6000, 0.33, 0.25),
             (5000, 0.44, 0.25), (4000, 0.55, 0.22), (0, 0.65, 0.15)]
    for threshold, u1, u2 in table:
        if teff >= threshold:
            return u1, u2
    return 0.65, 0.15


# ── Batman helpers ────────────────────────────────────────────────────────────

def batman_flux(t, period, t0, rp, a, inc, u1, u2):
    p = batman.TransitParams()
    p.t0 = t0;  p.per = period; p.rp = rp; p.a = a; p.inc = inc
    p.ecc = 0.0; p.w = 90.0; p.u = [u1, u2]; p.limb_dark = "quadratic"
    return batman.TransitModel(p, t).light_curve(p)


def transit_duration_days(period, rp, a, inc_deg):
    """T14 — first to fourth contact duration in days."""
    b = a * np.cos(np.radians(inc_deg))
    inner = np.clip(((1 + rp) ** 2 - b ** 2) / (a ** 2), 0.0, 1.0)
    return period / np.pi * np.arcsin(np.sqrt(inner))


# ── Step 1: Find rough T0 via grid search ─────────────────────────────────────

def rough_t0(lc_flat, period):
    """
    Find transit epoch from the phase minimum of the flattened light curve.
    Uses lc.fold() so the result is in the same time system as lc.time.
    Avoids a T0 grid search, which fails on active M dwarfs.
    """
    # Fold at the known period with an arbitrary reference (first data point)
    ref = lc_flat.time.value[0]
    lc_fold = lc_flat.fold(period=period, epoch_time=ref)

    ph = lc_fold.phase.value   # days from ref, range [-period/2, period/2]
    fl = lc_fold.flux.value

    # Bin finely (500 bins) and smooth to find the minimum robustly
    n_bins = 500
    bins  = np.linspace(-period / 2, period / 2, n_bins + 1)
    bc    = 0.5 * (bins[:-1] + bins[1:])
    bfl   = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (ph >= bins[i]) & (ph < bins[i + 1])
        if m.sum() >= 3:
            bfl[i] = np.median(fl[m])

    # Fill NaN gaps with local median before smoothing
    nan_mask = np.isnan(bfl)
    bfl[nan_mask] = np.nanmedian(bfl)

    # 11-point Savitzky-Golay smooth to average over noise
    from scipy.signal import savgol_filter
    bfl_smooth = savgol_filter(bfl, window_length=11, polyorder=2)

    # Transit phase offset (in days from ref)
    t0_phase = bc[np.argmin(bfl_smooth)]
    depth_est = 1.0 - np.min(bfl_smooth)
    print(f"  Rough T0 phase = {t0_phase * 24 * 60:.1f} min from ref  "
          f"(depth est. = {depth_est * 1e3:.1f} mmag)")

    # Convert phase offset to absolute BJD
    t0_abs = ref + t0_phase
    return t0_abs


# ── Step 2: Phase-fold, bin, master batman fit ────────────────────────────────

def phase_fold_and_bin(lc, period, t0_abs, n_bins=200):
    lc_fold = lc.fold(period=period, epoch_time=t0_abs)
    # lightkurve 2.x: phase.value is in the same time unit as period (days),
    # ranging from -period/2 to +period/2
    ph  = lc_fold.phase.value           # days from transit center
    fl  = lc_fold.flux.value
    fe  = (lc_fold.flux_err.value
           if hasattr(lc_fold, "flux_err") and lc_fold.flux_err is not None
           else np.ones_like(fl) * 1e-3)

    # Bins in days
    bins = np.linspace(-period / 2, period / 2, n_bins + 1)
    bc   = 0.5 * (bins[:-1] + bins[1:])   # bin centers in days
    bfl  = np.full(n_bins, np.nan)
    bfe  = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (ph >= bins[i]) & (ph < bins[i + 1])
        if m.sum() >= 3:
            bfl[i] = np.median(fl[m])
            bfe[i] = np.std(fl[m]) / np.sqrt(m.sum())

    ok = ~np.isnan(bfl)
    bc_ok, bfl_ok, bfe_ok = bc[ok], bfl[ok], bfe[ok]

    # Sigma-clip outlier bins (sector-boundary SG artifacts, flares)
    med  = np.median(bfl_ok)
    mad  = np.median(np.abs(bfl_ok - med))
    good = np.abs(bfl_ok - med) < 8 * mad
    return bc_ok[good], bfl_ok[good], bfe_ok[good]


def master_fit(ph_bin, fl_bin, fe_bin, period, rp0, a0, inc0, u1_0, u2_0):
    """
    Two-stage batman fit to phase-folded binned data.
    ph_bin is in days from transit center (lightkurve fold convention).

    Stage 1: Fix Rp/Rs=rp0, a/Rs=a0, u1/u2 from LD table. Fit only t0 and inc.
             This avoids the optimizer escaping to degenerate solutions.
    Stage 2: Release all 6 params with tight bounds centred on Stage 1 result.
    """
    t_bat = ph_bin   # days from center

    # ── Stage 1: constrained fit (t0 + inclination only) ──
    def loss_s1(params):
        t0, inc = params
        try:
            mod = batman_flux(t_bat, period, t0, rp0, a0, inc, u1_0, u2_0)
            return np.sum(((fl_bin - mod) / (fe_bin + 1e-8)) ** 2)
        except Exception:
            return 1e12

    bounds_s1 = [
        (-0.08 * period, 0.08 * period),  # t0 offset
        (70.0, 90.0),                      # inclination (all transiting planets)
    ]
    res1 = minimize(loss_s1, [0.0, inc0], bounds=bounds_s1, method="L-BFGS-B",
                    options={"maxiter": 5000, "ftol": 1e-14})
    t0_s1, inc_s1 = res1.x
    print(f"    Stage 1: t0={t0_s1*24*60:.2f} min, inc={inc_s1:.2f}°  loss={res1.fun:.2f}")

    # ── Stage 2: release all params, tight bounds around Stage 1 result ──
    def loss_s2(params):
        t0, rp, a, inc, u1, u2 = params
        try:
            mod = batman_flux(t_bat, period, t0, rp, a, inc, u1, u2)
            return np.sum(((fl_bin - mod) / (fe_bin + 1e-8)) ** 2)
        except Exception:
            return 1e12

    bounds_s2 = [
        (t0_s1 - 0.02 * period, t0_s1 + 0.02 * period),   # t0: tight around S1
        (max(rp0 * 0.6, 0.01),  min(rp0 * 1.6, 0.99)),     # Rp/Rs: ±40% of TEPCat
        (max(a0  * 0.6, 1.5),   a0  * 1.8),                 # a/Rs: ±40% of TEPCat
        (max(inc_s1 - 10, 70),  90.0),                      # inc: near S1 result
        (0.0,  1.0),                                          # u1
        (-0.5, 1.0),                                          # u2
    ]
    x0_s2 = [t0_s1, rp0, a0, inc_s1, u1_0, u2_0]
    res2 = minimize(loss_s2, x0_s2, bounds=bounds_s2, method="L-BFGS-B",
                    options={"maxiter": 8000, "ftol": 1e-15, "gtol": 1e-11})
    t0_off, rp, a, inc, u1, u2 = res2.x
    dur = transit_duration_days(period, rp, a, inc)

    # Sanity check: if Stage 2 made things worse OR duration is unphysical, fall back
    if res2.fun > res1.fun * 1.05 or dur < 0.005:
        print("    Stage 2 degraded fit — keeping Stage 1 t0/inc, TEPCat Rp/Rs & a/Rs")
        # Use inc_s1 only if it gives a physical transit; otherwise revert to inc0
        inc_use = inc_s1 if transit_duration_days(period, rp0, a0, inc_s1) > 0.005 else inc0
        t0_off, rp, a, inc, u1, u2 = t0_s1, rp0, a0, inc_use, u1_0, u2_0
        dur = transit_duration_days(period, rp, a, inc)
        if dur < 0.005:
            # Last resort: use inc0 directly and compute geometric duration
            inc = inc0
            dur = transit_duration_days(period, rp, a, inc)

    return dict(t0_offset_days=t0_off, rp_rs=rp, a_rs=a, inc=inc,
                u1=u1, u2=u2, duration_days=dur,
                loss=res2.fun, n_bins=len(fl_bin))


# ── Step 3: Plot master phase-fold fit ────────────────────────────────────────

def plot_phase_fold(ph_bin, fl_bin, fe_bin, fit, period, planet_name, out_dir):
    t_bat  = ph_bin           # already in days from center
    t0_off = fit["t0_offset_days"]
    mod    = batman_flux(t_bat, period, t0_off,
                         fit["rp_rs"], fit["a_rs"], fit["inc"],
                         fit["u1"], fit["u2"])
    t_h    = t_bat * 24   # hours

    dur_h  = fit["duration_days"] * 24
    depth  = fit["rp_rs"] ** 2
    y_lo   = 1.0 - 6 * depth
    y_hi   = 1.0 + 4 * depth
    resid  = fl_bin - mod
    rms_ppm = np.std(resid) * 1e6

    # ── Full-phase plot (left panel) + zoomed transit (right panel) ──
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, 2, width_ratios=[2, 1],
                           height_ratios=[3, 1], hspace=0.07, wspace=0.25)
    ax1      = fig.add_subplot(gs[0, 0])
    ax2      = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_zoom  = fig.add_subplot(gs[0, 1])
    ax_zresid= fig.add_subplot(gs[1, 1], sharex=ax_zoom)

    # Full phase — plot without error bars (they swamp the signal)
    ax1.plot(t_h, fl_bin, "o", ms=4, color="steelblue", alpha=0.7,
             label="Phase-folded TESS (binned, 200 bins)", zorder=3)
    ax1.plot(t_h, mod, "r-", lw=2.2, zorder=5, label="Batman Mandel-Agol fit")
    for sign in (-1, 1):
        ax1.axvline(sign * dur_h / 2, color="gray", ls=":", lw=1, alpha=0.7)
    ax1.set_ylabel("Normalized Flux", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_title(
        f"{planet_name}  ·  P = {period:.5f} d  ·  Rp/Rs = {fit['rp_rs']:.4f}"
        f"  ·  a/Rs = {fit['a_rs']:.2f}  ·  i = {fit['inc']:.2f}°"
        f"  ·  Dur = {dur_h:.2f} h  ·  u1={fit['u1']:.3f}, u2={fit['u2']:.3f}",
        fontsize=9, pad=8)
    ax1.set_ylim(y_lo, y_hi)

    ax2.plot(t_h, resid, "o", ms=3, color="dimgray", alpha=0.75)
    ax2.axhline(0, color="r", lw=1.2)
    for sign in (-1, 1):
        ax2.axvline(sign * dur_h / 2, color="gray", ls=":", lw=1, alpha=0.7)
    ax2.set_xlabel("Time from transit center (hours)", fontsize=12)
    ax2.set_ylabel("Residuals", fontsize=11)
    ax2.set_title(f"RMS = {rms_ppm:.0f} ppm", fontsize=9, pad=3)

    # ── Zoomed transit panel — ±3× duration ──
    zoom_h = max(3 * dur_h, 0.5)
    zm = np.abs(t_h) <= zoom_h
    # Clip outlier bins in zoom window before plotting
    fl_zm, t_zm = fl_bin[zm], t_h[zm]
    if zm.sum() > 5:
        med_zm = np.median(fl_zm)
        mad_zm = np.median(np.abs(fl_zm - med_zm)) + 1e-9
        ok_zm  = np.abs(fl_zm - med_zm) < 5 * mad_zm
        fl_zm, t_zm = fl_zm[ok_zm], t_zm[ok_zm]
    ax_zoom.plot(t_zm, fl_zm, "o", ms=5, color="steelblue", alpha=0.85, zorder=3)
    t_mod_zoom = np.linspace(-zoom_h / 24, zoom_h / 24, 500)
    mod_zoom = batman_flux(t_mod_zoom, period, fit["t0_offset_days"],
                           fit["rp_rs"], fit["a_rs"], fit["inc"],
                           fit["u1"], fit["u2"])
    ax_zoom.plot(t_mod_zoom * 24, mod_zoom, "r-", lw=2.2, zorder=5)
    for sign in (-1, 1):
        ax_zoom.axvline(sign * dur_h / 2, color="gray", ls=":", lw=1, alpha=0.7)
    ax_zoom.set_xlim(-zoom_h, zoom_h)
    ax_zoom.set_ylim(y_lo, y_hi)
    ax_zoom.set_ylabel("Normalized Flux", fontsize=10)
    ax_zoom.set_title(f"Transit zoom  (depth = {depth*1e6:.0f} ppm)", fontsize=10)

    resid_zm = resid[zm]
    ax_zresid.plot(t_zm, resid_zm[ok_zm] if zm.sum() > 5 else resid_zm,
                   "o", ms=4, color="dimgray", alpha=0.8)
    ax_zresid.axhline(0, color="r", lw=1.2)
    rz_std = np.std(resid_zm) if len(resid_zm) > 1 else 1e-3
    ax_zresid.set_ylim(-5 * rz_std, 5 * rz_std)
    for sign in (-1, 1):
        ax_zresid.axvline(sign * dur_h / 2, color="gray", ls=":", lw=1, alpha=0.7)
    ax_zresid.set_xlabel("Time from transit center (hours)", fontsize=10)
    ax_zresid.set_ylabel("Residuals", fontsize=9)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "phase_fold_fit.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: phase_fold_fit.png")
    return path


# ── Step 4: Individual transit identification ─────────────────────────────────

def find_transit_centers(time, t0_abs, period, duration_days):
    """
    Return transit midtimes that fall fully within the observed timespan,
    with enough room on both sides for the pre and post baseline windows.
    """
    pad      = 0.1 * duration_days   # 10% padding at segment edges
    half_win = duration_days + pad    # center → edge of triplet window

    n0 = int(np.floor((time[0] - t0_abs) / period))
    n1 = int(np.ceil( (time[-1] - t0_abs) / period))

    centers = []
    for n in range(n0, n1 + 1):
        tc = t0_abs + n * period
        if tc - half_win >= time[0] and tc + half_win <= time[-1]:
            centers.append(tc)
    return np.array(centers)


# ── Step 5: Triplet extraction ────────────────────────────────────────────────

def extract_segment(time, flux, ferr, t1, t2, min_points=5):
    m = (time >= t1) & (time <= t2)
    if m.sum() < min_points:
        return None
    return time[m], flux[m], ferr[m]


def extract_triplet(time, flux, ferr, tc, duration_days):
    """
    Extract three equal-duration segments of length D around transit center tc:
      pre:  [tc - 2D, tc - D]
      in:   [tc - D,  tc + D]  (full transit window)
      post: [tc + D,  tc + 2D]
    where D = duration_days / 2 (half-duration).
    """
    D = duration_days / 2
    pre  = extract_segment(time, flux, ferr, tc - 2*D, tc - D)
    intr = extract_segment(time, flux, ferr, tc - D,   tc + D)
    post = extract_segment(time, flux, ferr, tc + D,   tc + 2*D)
    return pre, intr, post


# ── Step 6: Per-transit fit ───────────────────────────────────────────────────

def fit_single_transit(t_in, f_in, fe_in, tc_nom, fit_params):
    """
    Fixed shape (Rp/Rs, a/Rs, inc, u1, u2) from master fit.
    Free: t0 (search ±5% of period around tc_nom), slope, intercept.
    Returns dict with corrected flux, model, residuals, t0_fit.
    """
    period = fit_params["period"]
    rp     = fit_params["rp_rs"]
    a      = fit_params["a_rs"]
    inc    = fit_params["inc"]
    u1     = fit_params["u1"]
    u2     = fit_params["u2"]
    search = 0.05 * period

    def loss(params):
        t0, slope, intercept = params
        baseline = 1.0 + slope * (t_in - tc_nom) + intercept
        f_corr   = f_in / np.clip(baseline, 0.85, 1.15)
        try:
            mod = batman_flux(t_in, period, t0, rp, a, inc, u1, u2)
            return np.sum(((f_corr - mod) / (fe_in + 1e-8)) ** 2)
        except Exception:
            return 1e12

    bounds = [
        (tc_nom - search, tc_nom + search),
        (-0.005, 0.005),    # slope (per day)
        (-0.03,  0.03),     # intercept
    ]
    res = minimize(loss, [tc_nom, 0.0, 0.0], bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 400})
    t0_fit, slope, intercept = res.x
    baseline   = 1.0 + slope * (t_in - tc_nom) + intercept
    f_corr     = f_in / np.clip(baseline, 0.85, 1.15)
    model_flux = batman_flux(t_in, period, t0_fit, rp, a, inc, u1, u2)
    residuals  = f_corr - model_flux
    rchi2      = res.fun / max(len(t_in) - 3, 1)

    return dict(t=t_in, f_corr=f_corr, model=model_flux,
                residuals=residuals, t0_fit=t0_fit, rchi2=rchi2)


# ── Step 7: Transit grid plot ─────────────────────────────────────────────────

def _clip_segment(t, f, fe, n_sigma=5):
    """Remove outliers from a flux segment using MAD-based sigma clipping."""
    if len(f) < 5:
        return t, f, fe
    med = np.median(f)
    mad = np.median(np.abs(f - med)) + 1e-9
    ok  = np.abs(f - med) < n_sigma * mad
    return t[ok], f[ok], fe[ok]


def plot_transit_grid(transit_results, fit_params, planet_name, out_dir, n_show=20, rp_rs=None):
    """
    Grid of individual transit fits.
    Each transit occupies 2 rows: [data + model] on top, [residuals] below.
    """
    n     = min(n_show, len(transit_results))
    ncols = 5
    nrows_tile = int(np.ceil(n / ncols))
    dur_h = fit_params["duration_days"] * 24

    # Y-axis limits based on transit depth
    rp = rp_rs if rp_rs is not None else fit_params.get("rp", 0.05)
    depth = rp ** 2
    y_lo = 1.0 - 6.0 * depth
    y_hi = 1.0 + 4.0 * depth

    fig_w = ncols * 3.2
    fig_h = nrows_tile * 3.8
    fig   = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        f"{planet_name}  —  Individual transit fits  (first {n} shown)\n"
        f"Blue dots: baseline + in-transit flux  ·  Red line: Batman model"
        f"  ·  Dashed verticals: transit contacts",
        fontsize=10, y=1.005)

    for i, res in enumerate(transit_results[:n]):
        tc  = res["tc"]
        tr  = res["transit"]
        pre = res["pre"]
        post= res["post"]

        col      = i % ncols
        tile_row = i // ncols
        # Two sub-rows per tile
        row_data  = tile_row * 2
        row_resid = tile_row * 2 + 1

        total_rows = nrows_tile * 2
        ax_data  = fig.add_subplot(total_rows, ncols, row_data  * ncols + col + 1)
        ax_resid = fig.add_subplot(total_rows, ncols, row_resid * ncols + col + 1,
                                   sharex=ax_data)

        # ── Full triplet: pre + in-transit + post (no error bars — noise-dominated) ──
        all_t, all_f = [], []
        for seg in (pre, post):
            if seg is not None:
                ts, fs, _ = _clip_segment(seg[0], seg[1], seg[2])
                all_t.append((ts - tc) * 24)
                all_f.append(fs)
        t_in_h = (tr["t"] - tc) * 24
        t_in_c, f_in_c, _ = _clip_segment(tr["t"], tr["f_corr"],
                                           np.full_like(tr["f_corr"], 1e-4))
        t_in_c_h = (t_in_c - tc) * 24

        for th, fh in zip(all_t, all_f):
            ax_data.plot(th, fh, ".", ms=1.5, color="steelblue", alpha=0.4)
        ax_data.plot(t_in_c_h, f_in_c, ".", ms=2, color="steelblue", alpha=0.85)
        ax_data.plot(t_in_h, tr["model"], "r-", lw=1.5, zorder=5)

        # Zoom x-axis to ±3× transit duration so the dip fills the panel
        x_half = max(3 * dur_h, 0.3)
        ax_data.set_xlim(-x_half, x_half)

        # Y: data-driven within the zoom window, floored to transit depth range
        in_view = np.abs(t_in_c_h) <= x_half
        if in_view.sum() > 3:
            fv = f_in_c[in_view]
            pad = max(3 * depth, 5e-4)
            ax_data.set_ylim(min(fv.min() - pad, 1.0 - 4 * depth),
                             max(fv.max() + pad, 1.0 + 3 * depth))
        else:
            ax_data.set_ylim(y_lo, y_hi)

        for sign in (-1, 1):
            ax_data.axvline(sign * dur_h / 2, color="gray", ls=":", lw=0.7, alpha=0.6)

        dt_min = (tr["t0_fit"] - tc) * 24 * 60
        ax_data.set_title(f"T#{i+1}  ΔT₀={dt_min:+.1f} min  χ²={tr['rchi2']:.2f}",
                           fontsize=6.5)
        ax_data.tick_params(labelsize=5.5)
        plt.setp(ax_data.get_xticklabels(), visible=False)

        # ── Residuals (in-transit window only) ──
        resid_mmag = tr["residuals"] * 1e3
        resid_std  = np.std(resid_mmag) if len(resid_mmag) > 1 else 1.0
        ax_resid.plot(t_in_h, resid_mmag, ".", ms=2, color="dimgray", alpha=0.8)
        ax_resid.axhline(0, color="r", lw=0.9)
        ax_resid.set_ylim(-6 * resid_std, 6 * resid_std)
        for sign in (-1, 1):
            ax_resid.axvline(sign * dur_h / 2, color="gray", ls=":", lw=0.7, alpha=0.6)
        ax_resid.tick_params(labelsize=5.5)

        # x-label only on bottom row panels
        if tile_row == nrows_tile - 1 or i + ncols >= n:
            ax_resid.set_xlabel("Δt from center (h)", fontsize=6)

    # Shared y-labels via figure text
    fig.text(0.005, 0.75, "Norm. Flux", va="center", rotation="vertical",
             fontsize=9, color="steelblue")
    fig.text(0.005, 0.30, "Resid. (×10⁻³)", va="center", rotation="vertical",
             fontsize=8, color="dimgray")

    plt.tight_layout(rect=[0.015, 0, 1, 1])
    path = os.path.join(out_dir, "individual_transits_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: individual_transits_grid.png  ({n} transits)")
    return path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_planet(row, max_transits=60, n_grid=20):
    name    = str(row["System"]).replace("_", " ")
    host    = str(row["host_star"])
    period  = float(row["Period"])
    rp0     = float(row["Rp_Rs"])
    a0      = float(row["a_Rs"])
    teff    = float(row["Teff"])
    inc0    = 85.0
    u1_0, u2_0 = ld_guess(teff)
    out_dir = os.path.join(RESULTS_BASE, name.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    # Apply unit correction: TEPCat R_b is in R_Jup, R_A is in R_sun
    rp0_raw = rp0
    rp0     = rp0 * R_JUP_IN_RSUN   # corrected Rp/Rs

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  Planet : {name}")
    print(f"  Host   : {host}")
    print(f"  Period : {period:.5f} d")
    print(f"  Rp/Rs  : {rp0:.5f}  (catalog {rp0_raw:.4f} × {R_JUP_IN_RSUN:.5f} unit fix)")
    print(f"  a/Rs   : {a0:.2f}   transit depth : {rp0**2*1e6:.0f} ppm")
    print(f"  Teff   : {teff:.0f} K   LD init: u1={u1_0:.2f}, u2={u2_0:.2f}")
    print(f"  Est. transits: {row['est_n_transits']:.0f}   Sectors: {row['n_sectors']}")
    print(f"  Output: {out_dir}")
    print(sep)

    # ── 1. Download ──
    lc_raw = download_lc(host)
    time, flux, ferr = lc_arrays(lc_raw)

    # ── 1b. Flatten for T0 search and phase fold ──
    # Savitzky-Golay window ~6 h removes stellar variability without touching the transit
    sg_window = max(201, 2 * int(0.125 / (period / (len(lc_raw) / (time[-1] - time[0])))) + 1)
    sg_window = sg_window if sg_window % 2 == 1 else sg_window + 1
    print(f"  Flattening LC (SG window = {sg_window} points)...")
    lc_flat = lc_raw.flatten(window_length=sg_window)

    # ── 2. Rough T0 from phase minimum of flattened LC ──
    print("  Finding rough T0 from phase-fold minimum...")
    t0_abs = rough_t0(lc_flat, period)
    print(f"  Rough T0 = {t0_abs:.5f} BJD")

    # ── 3. Phase-fold FLAT LC + master fit ──
    print("  Phase-folding and binning (flattened LC)...")
    ph_bin, fl_bin, fe_bin = phase_fold_and_bin(lc_flat, period, t0_abs)
    print(f"  Binned: {len(ph_bin)} non-empty bins")
    print("  Running master batman fit...")
    fit = master_fit(ph_bin, fl_bin, fe_bin, period, rp0, a0, inc0, u1_0, u2_0)
    fit["period"] = period

    # Adjust absolute T0 by the fitted phase offset
    t0_abs += fit["t0_offset_days"]
    fit["t0_abs"] = t0_abs

    dur_h = fit["duration_days"] * 24
    print(f"  Master fit: Rp/Rs={fit['rp_rs']:.4f}  a/Rs={fit['a_rs']:.2f}"
          f"  i={fit['inc']:.2f}°  dur={dur_h:.2f} h")

    # ── 4. Phase-fold plot ──
    plot_phase_fold(ph_bin, fl_bin, fe_bin, fit, period, name, out_dir)

    # ── 5. Find transit centers (use raw LC time array) ──
    centers = find_transit_centers(time, t0_abs, period, fit["duration_days"])
    print(f"  Found {len(centers)} transit centers with full triplet window")

    if max_transits and len(centers) > max_transits:
        # Sample evenly across the time baseline so we get diversity
        idx    = np.round(np.linspace(0, len(centers) - 1, max_transits)).astype(int)
        centers = centers[idx]
        print(f"  Subsampled to {len(centers)} transits (evenly spaced across baseline)")

    # ── 6. Per-transit fitting ──
    transit_results = []
    skipped = 0
    for i, tc in enumerate(centers):
        pre, intr, post = extract_triplet(time, flux, ferr, tc,
                                           fit["duration_days"])
        if intr is None:
            skipped += 1
            continue
        tr = fit_single_transit(*intr, tc, fit)
        transit_results.append(dict(tc=tc, pre=pre, transit=tr, post=post))

        if (i + 1) % 20 == 0:
            print(f"    Fitted {i+1}/{len(centers)} transits...")

    print(f"  Fitted {len(transit_results)} transits ({skipped} skipped, insufficient data)")

    # ── 7. Save CSV summary ──
    rows_out = []
    for res in transit_results:
        tr = res["transit"]
        rows_out.append({
            "tc_nominal_bjd": res["tc"],
            "t0_fitted_bjd":  tr["t0_fit"],
            "dt0_min":        (tr["t0_fit"] - res["tc"]) * 24 * 60,
            "rchi2":          tr["rchi2"],
            "n_in_points":    len(tr["t"]),
            "has_pre":        res["pre"] is not None,
            "has_post":       res["post"] is not None,
        })
    csv_path = os.path.join(out_dir, "transit_fits.csv")
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)
    print(f"  Saved: transit_fits.csv  ({len(rows_out)} rows)")

    # ── 8. Transit grid plot ──
    plot_transit_grid(transit_results, fit, name, out_dir, n_show=n_grid, rp_rs=rp0)

    print(f"\n  Done. Results in: {out_dir}")
    return fit, transit_results


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="TESS transit triplet preprocessor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--test",   action="store_true",
                    help="Use the planet with the most estimated transits")
    ap.add_argument("--planet", type=str, default=None,
                    help="Planet system name (e.g. 'WASP-100' or 'WASP-100_b')")
    ap.add_argument("--max",    type=int, default=60,
                    help="Maximum individual transits to fit (evenly sampled)")
    ap.add_argument("--grid",   type=int, default=20,
                    help="How many transits to show in the image grid")
    args = ap.parse_args()

    if not args.test and not args.planet:
        ap.print_help()
        return

    df  = load_catalog()
    row = pick_planet(df, name=args.planet)

    print(f"\nSelected: {row['System']}")
    print(f"  est. {row['est_n_transits']:.0f} transits  |  {row['n_sectors']} sectors")

    run_planet(row, max_transits=args.max, n_grid=args.grid)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
export_tess_focused.py

For each selected planet, downloads TESS SPOC PDC-SAP LC per sector and
produces zoomed 5T-wide transit images:

  |<--- 2T pre-baseline --->|<-- T transit -->|<--- 2T post-baseline --->|

Improvements over v1:
  - Stacking-based t0 refinement per sector (replaces sector_report t0)
  - Robust per-transit parabola fit with ±0.3T clamp (avoids runaway shifts)
  - Batman model Y-anchored to out-of-transit data baseline
  - Gap detection: skips transits with >10-min gaps inside in-transit window
  - Local baseline detrending for sinusoidal/variable stars
  - --reexport flag to only process planets with known issues

Usage:
    python export_tess_focused.py                    # all planets
    python export_tess_focused.py --planet KELT-09
    python export_tess_focused.py --reexport         # only flagged planets
"""

import argparse
import os
import re
import time
import warnings
warnings.filterwarnings("ignore")

import batman
import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

BJD_OFFSET  = 2457000.0
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT    = os.path.join(BASE_DIR, "TESS_focused", "current")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# Planets flagged for re-export (has offset/cutoff/yaxis issues)
REEXPORT_PLANETS = [
    "KELT-09", "KELT-18", "KELT-23", "WASP-019", "WASP-045",
    "HAT-P-39", "KELT-17", "TOI-2154", "TOI-4153",
    "TOI-2152", "TOI-1268",
    "HAT-P-37",   # sinusoidal — needs detrending
    "TOI-2025",   # semiok / offsetLeft
]

LD_TABLE = [
    (7000, 0.20, 0.20),
    (6000, 0.33, 0.25),
    (5000, 0.44, 0.25),
    (4000, 0.55, 0.22),
    (0,    0.65, 0.15),
]

GAP_THRESHOLD_DAYS = 10 / 60 / 24   # 10 minutes


# ── Utilities ──────────────────────────────────────────────────────────────────

def robust_normalize(lc):
    flux_vals = np.array(lc.flux)
    if hasattr(flux_vals.flat[0], "value"):
        flux_vals = np.array([v.value for v in flux_vals])
    divisor = np.nanpercentile(flux_vals, 75)
    if divisor == 0 or np.isnan(divisor):
        divisor = 1.0
    return lk.LightCurve(time=lc.time, flux=flux_vals / divisor)


def download_sector_lc(host, sector, retries=4):
    delays = [10, 20, 30]
    for attempt in range(retries):
        try:
            sr = lk.search_lightcurve(
                host, mission="TESS", sector=sector,
                author="SPOC", exptime=120
            )
            if sr is None or len(sr) == 0:
                return None
            lc = sr[0].download(flux_column="pdcsap_flux")
            if lc is None:
                return None
            lc = lc.remove_nans()
            return robust_normalize(lc)
        except Exception as e:
            if attempt < retries - 1:
                wait = delays[min(attempt, len(delays) - 1)]
                print(f"    Retry {attempt+1} ({e}) — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"    Download failed: {e}")
                return None


def refine_t0_stacking(time, flux, t0_init, period, half_dur, n_bins=40):
    """
    Stack all in-sector transit windows in phase space and fit a parabola
    to the binned minimum to get a better t0.
    """
    phase = (time - t0_init) % period
    phase[phase > period / 2] -= period
    mask = np.abs(phase) <= 2 * half_dur
    if mask.sum() < 20:
        return t0_init

    p_in = phase[mask]
    f_in = flux[mask]

    bins = np.linspace(-2 * half_dur, 2 * half_dur, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bt, bf = [], []
    for i in range(n_bins):
        sel = (p_in >= bins[i]) & (p_in < bins[i + 1])
        if sel.sum() > 0:
            bt.append(bin_centers[i])
            bf.append(np.median(f_in[sel]))

    if len(bt) < 5:
        return t0_init

    bt, bf = np.array(bt), np.array(bf)
    inner = np.abs(bt) <= half_dur
    if inner.sum() < 3:
        inner = np.ones(len(bt), dtype=bool)

    try:
        a, b, c = np.polyfit(bt[inner], bf[inner], 2)
        if a <= 0:
            return t0_init
        offset = -b / (2 * a)
        if abs(offset) > half_dur:
            return t0_init
        return t0_init + offset
    except Exception:
        return t0_init


def refine_tc_batman(time, flux, tc_guess, period, rp, a_rs, inc, ecc, u1, u2,
                     T_days, search_frac=0.6):
    """
    Refine tc by minimizing chi-squared between data and batman model.
    Uses the full transit shape as a template — more robust than parabola.
    search_frac: search window = ±search_frac * T_days
    """
    half_search = search_frac * T_days
    mask = (time >= tc_guess - 1.5 * T_days) & (time <= tc_guess + 1.5 * T_days)
    if mask.sum() < 10:
        return tc_guess
    t_fit = time[mask]
    f_fit = flux[mask]

    def chi2(tc_shift):
        model = get_batman_model(t_fit, tc_guess + tc_shift,
                                 period, rp, a_rs, inc, ecc, 90.0, u1, u2)
        # Anchor model baseline to data before comparing
        out = np.abs(t_fit - (tc_guess + tc_shift)) > 0.6 * T_days
        if out.sum() > 3:
            scale = np.median(f_fit[out]) / np.median(model[out])
            model = model * scale
        return np.sum((f_fit - model) ** 2)

    try:
        result = minimize_scalar(chi2, bounds=(-half_search, half_search),
                                 method="bounded")
        shift = np.clip(result.x, -half_search, half_search)
        return tc_guess + shift
    except Exception:
        return tc_guess


def has_gap(time, tc, T_days):
    """Return True if the in-transit window has a data gap > GAP_THRESHOLD_DAYS."""
    mask = (time >= tc - 0.5 * T_days) & (time <= tc + 0.5 * T_days)
    t_in = np.sort(time[mask])
    if len(t_in) < 2:
        return True
    gaps = np.diff(t_in)
    return bool(np.any(gaps > GAP_THRESHOLD_DAYS))


def detrend_baseline(time, flux, tc, T_days, deg=2):
    """
    Fit a polynomial to out-of-transit data and divide it out.
    Removes sinusoidal/slope trends from the baseline.
    """
    out_mask = (np.abs(time - tc) > 0.6 * T_days) & \
               (np.abs(time - tc) <= 2.5 * T_days)
    if out_mask.sum() < 6:
        return flux
    t_rel = time - tc
    try:
        coeffs = np.polyfit(t_rel[out_mask], flux[out_mask], deg)
        trend  = np.polyval(coeffs, t_rel)
        # Divide out trend, re-anchor to 1.0
        detrended = flux / trend
        return detrended
    except Exception:
        return flux


def get_batman_model(time_arr, tc, period, rp, a, inc, ecc, w_deg, u1, u2):
    p = batman.TransitParams()
    p.t0        = tc
    p.per       = period
    p.rp        = abs(rp)
    p.a         = abs(a)
    p.inc       = inc
    p.ecc       = max(0.0, ecc)
    p.w         = w_deg
    p.u         = [u1, u2]
    p.limb_dark = "quadratic"
    try:
        m = batman.TransitModel(p, time_arr)
        return m.light_curve(p)
    except Exception:
        return np.ones_like(time_arr)


def anchor_batman_to_data(bat, t_win, tc_refined, T_days, f_win):
    """Shift batman baseline to match the out-of-transit data median."""
    out_mask = np.abs(t_win - tc_refined) > 0.6 * T_days
    if out_mask.sum() < 5:
        return bat
    data_baseline  = np.median(f_win[out_mask])
    model_baseline = np.median(bat[out_mask])
    if model_baseline == 0:
        return bat
    # Scale so model baseline matches data baseline
    return bat * (data_baseline / model_baseline)


def load_manual_offsets(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "tc_manual_offsets.csv")
    """
    Load user-specified per-transit tc offsets (in minutes).
    CSV columns: planet, sector (int or 'all'), transit_n (int or 'all'), offset_min
    Returns dict keyed by (planet, sector, transit_n) -> offset_days
    """
    offsets = {}
    if not os.path.exists(path):
        return offsets
    df = pd.read_csv(path, comment="#")
    for _, row in df.iterrows():
        planet    = str(row["planet"]).strip()
        sector    = row["sector"]    # int or 'all'
        transit_n = row["transit_n"] # int or 'all'
        offset_d  = float(row["offset_min"]) / 60 / 24
        offsets[(planet, str(sector), str(transit_n))] = offset_d
    return offsets


def get_manual_offset(offsets, planet, sector, transit_n):
    """Look up offset with fallback: specific transit > all transits in sector > all sectors."""
    for s_key in (str(sector), "all"):
        for t_key in (str(transit_n), "all"):
            key = (planet, s_key, t_key)
            if key in offsets:
                return offsets[key]
    return 0.0


def btjd_to_utc(btjd):
    try:
        t = Time(btjd + BJD_OFFSET, format="jd", scale="tdb")
        return t.iso[:10]
    except Exception:
        return "?"


def get_ld(teff):
    for thr, u1, u2 in LD_TABLE:
        if teff >= thr:
            return u1, u2
    return 0.44, 0.25


# ── Per-transit plot ───────────────────────────────────────────────────────────

def plot_transit(time, flux, tc_refined, T_days, batman_flux,
                 planet, sector, transit_n, tc_utc, out_path):
    T_h   = T_days * 24
    t_rel = (time - tc_refined) * 24

    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.axvspan(-2.5 * T_h, -0.5 * T_h, color="#cce5ff", alpha=0.35, label="Pre-baseline (2T)")
    ax.axvspan(-0.5 * T_h,  0.5 * T_h, color="#ffd6cc", alpha=0.35, label="Transit (T)")
    ax.axvspan( 0.5 * T_h,  2.5 * T_h, color="#ccffe5", alpha=0.35, label="Post-baseline (2T)")

    ax.scatter(t_rel, flux, s=4, color="#555555", zorder=3, label="PDC-SAP flux")

    if batman_flux is not None:
        ax.plot(t_rel, batman_flux, color="crimson", lw=1.5, zorder=4, label="Batman model")

    ax.axvline(-0.5 * T_h, color="navy", lw=1.0, ls="--", alpha=0.7, label="Ingress / Egress")
    ax.axvline( 0.5 * T_h, color="navy", lw=1.0, ls="--", alpha=0.7, label="_nolegend_")
    for x in [-2.5 * T_h, 2.5 * T_h]:
        ax.axvline(x, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax.axvline(0, color="red", lw=0.9, ls="-.", alpha=0.6, label="Refined tc")

    ax.set_xlim(-2.6 * T_h, 2.6 * T_h)
    ax.set_xlabel("Time from transit midpoint (hours)", fontsize=10)
    ax.set_ylabel("Normalised flux", fontsize=10)
    ax.set_title(
        f"{planet}  |  Sector {sector}  |  Transit #{transit_n}  ({tc_utc})\n"
        f"tc (BTJD) = {tc_refined:.5f}   T = {T_h:.2f} h",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet",   type=str,  default=None)
    parser.add_argument("--sector",   type=int,  default=None)
    parser.add_argument("--reexport", action="store_true",
                        help="Only process planets flagged with issues")
    args = parser.parse_args()

    plan = pd.read_csv(os.path.join(RESULTS_DIR, "summary", "transit_plan_detailed.csv"))
    pri  = pd.read_csv(os.path.join(DATA_DIR, "planets_priority.csv"))

    if args.planet:
        plan = plan[plan["planet"] == args.planet]
    elif args.reexport:
        plan = plan[plan["planet"].isin(REEXPORT_PLANETS)]
    if args.sector:
        plan = plan[plan["sector"] == args.sector]

    manual_offsets = load_manual_offsets()
    if manual_offsets:
        print(f"Loaded {len(manual_offsets)} manual tc offset(s) from tc_manual_offsets.csv")

    planets = plan["planet"].unique()
    total_images = len(plan)
    done = 0

    for planet in planets:
        p_plan = plan[plan["planet"] == planet]
        prow   = pri[pri["System"] == planet]
        if prow.empty:
            print(f"[{planet}] No parameter row — skipping")
            continue
        prow = prow.iloc[0]

        host   = re.sub(r"\s+[A-D]$", "", str(prow.get("host_star", planet))).strip()
        period = float(prow["Period"])
        T_days = float(prow["pl_trandur"]) / 24.0
        half_T = T_days / 2.0
        rp     = float(prow["pl_ratror"])
        a      = float(prow["pl_ratdor"])
        inc    = float(prow["pl_orbincl"])
        ecc_v  = prow["pl_orbeccen"]
        ecc    = float(ecc_v) if pd.notna(ecc_v) and float(ecc_v) >= 0 else 0.0
        teff   = float(prow["st_teff"]) if pd.notna(prow.get("st_teff")) else 5500.0
        u1, u2 = get_ld(teff)

        # Sinusoidal flag: apply local detrending
        do_detrend = planet in ("HAT-P-37",)

        sectors = p_plan["sector"].unique()

        for sector in sectors:
            s_plan  = p_plan[p_plan["sector"] == sector]
            n_take  = len(s_plan)
            out_dir = os.path.join(OUT_ROOT, planet, f"sector_{sector:03d}")
            os.makedirs(out_dir, exist_ok=True)

            print(f"[{planet}] Sector {sector} — downloading LC...", end=" ", flush=True)
            lc = download_sector_lc(host, sector)
            if lc is None:
                print("FAILED — skipping sector")
                continue

            time_arr = np.array(lc.time.value) if hasattr(lc.time, "value") else np.array(lc.time)
            flux_arr = np.array(lc.flux.value) if hasattr(lc.flux, "value") else np.array(lc.flux)
            print(f"{len(time_arr)} pts")

            # Load t0 from sector_report, then refine via stacking
            rep_path = os.path.join(RESULTS_DIR, "survey", planet, "sector_report.csv")
            t0_sec = None
            if os.path.exists(rep_path):
                rep = pd.read_csv(rep_path)
                rep = rep[rep["sector"] != "Total"]
                rep["sector"] = rep["sector"].astype(int)
                rep_row = rep[rep["sector"] == sector]
                if not rep_row.empty:
                    t0_sec = float(rep_row["t0_sec_btjd"].values[0])
            if t0_sec is None:
                t0_sec = float(s_plan.iloc[0]["tc_btjd"])

            # Refine t0 by stacking all transits in this sector's data
            t0_sec = refine_t0_stacking(time_arr, flux_arr, t0_sec, period, half_T)

            # Find all valid transit times (all 3 windows ≥10 pts, no gap in transit)
            t_lo = time_arr.min()
            t_hi = time_arr.max()
            n_lo = int(np.ceil((t_lo - t0_sec) / period))
            n_hi = int(np.floor((t_hi - t0_sec) / period))
            all_tc = [t0_sec + n * period for n in range(n_lo, n_hi + 1)]

            valid_tcs = []
            for tc_c in all_tc:
                pre  = (time_arr >= tc_c - 1.5*T_days) & (time_arr < tc_c - 0.5*T_days)
                in_t = (time_arr >= tc_c - 0.5*T_days) & (time_arr <= tc_c + 0.5*T_days)
                post = (time_arr >  tc_c + 0.5*T_days) & (time_arr <= tc_c + 1.5*T_days)
                if pre.sum() < 10 or in_t.sum() < 10 or post.sum() < 10:
                    continue
                if has_gap(time_arr, tc_c, T_days):
                    continue
                valid_tcs.append(tc_c)

            if not valid_tcs:
                print(f"  No valid transits found in LC data for sector {sector}")
                continue

            selected_tcs = valid_tcs[:n_take]
            print(f"  Found {len(valid_tcs)} valid (no-gap) transits, taking {len(selected_tcs)}")

            for t_idx, tc_pred in enumerate(selected_tcs):
                transit_n = t_idx + 1

                # Per-transit midpoint refinement via batman template chi-sq fit
                tc_refined = refine_tc_batman(
                    time_arr, flux_arr, tc_pred,
                    period, rp, a, inc, ecc, u1, u2, T_days
                )

                # Apply manual offset if specified by user
                manual_off = get_manual_offset(manual_offsets, planet, sector, transit_n)
                if manual_off != 0.0:
                    tc_refined += manual_off
                    print(f"    Manual offset applied: {manual_off*24*60:+.1f} min")

                # 5T display window
                win_lo   = tc_refined - 2.5 * T_days
                win_hi   = tc_refined + 2.5 * T_days
                mask_win = (time_arr >= win_lo) & (time_arr <= win_hi)
                t_win    = time_arr[mask_win]
                f_win    = flux_arr[mask_win]

                # Optional detrending for sinusoidal stars
                if do_detrend:
                    f_win = detrend_baseline(t_win, f_win, tc_refined, T_days)

                # Batman model + Y-anchor to data baseline
                bat = get_batman_model(t_win, tc_refined, period, rp, a, inc, ecc, 90.0, u1, u2)
                bat = anchor_batman_to_data(bat, t_win, tc_refined, T_days, f_win)

                tc_utc   = btjd_to_utc(tc_refined)
                out_path = os.path.join(out_dir, f"transit_{transit_n:02d}.png")
                plot_transit(t_win, f_win, tc_refined, T_days, bat,
                             planet, sector, transit_n, tc_utc, out_path)
                done += 1
                print(f"  Transit {transit_n:02d}  tc={tc_refined:.5f}  -> {out_path}  [{done}/{total_images}]")

    print(f"\nDone. {done} images saved to {OUT_ROOT}/")


if __name__ == "__main__":
    main()

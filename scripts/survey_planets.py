#!/usr/bin/env python3
"""
survey_planets.py — Download & survey all 42 priority planet light curves

For each planet in planets_priority.csv:
  1. Download all TESS PDC-SAP sectors via lightkurve
  2. Generate a sector grid image (like Cell 3 in inspect_lightcurves.ipynb)
  3. Count usable transits per sector
  4. Save a per-planet sector report CSV

Outputs per planet (under results/{planet}/):
  sector_grid.png       — all sectors plotted with transit markers
  sector_report.csv     — usable transit count per sector

A master summary is also saved to:
  results/survey_summary.csv  — all planets × all sectors

Supports smart resume — already-processed planets are skipped.

Usage:
    python survey_planets.py                    # all 42 planets
    python survey_planets.py --planet HAT-P-12  # single planet test
    python survey_planets.py --min-pts 10       # change usability threshold
"""

import argparse
import os
import re
import time
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PRIORITY_CSV  = "planets_priority.csv"
LOOKUP_CSV    = "sector_lookup.csv"
RESULTS_DIR   = "results"
SUMMARY_CSV   = os.path.join(RESULTS_DIR, "survey_summary.csv")
BJD_OFFSET    = 2457000.0
MJD_TO_BTJD  = 2400000.5 - BJD_OFFSET
SLEEP         = 2.0
MIN_PTS       = 10       # minimum in-transit points to count as usable


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_host(row):
    host = str(row.get("host_star", row.get("System", ""))).strip()
    return re.sub(r'\s+[A-D]$', '', host).strip()


def robust_normalize(lc):
    """
    Normalize by 75th percentile and strip flux units.
    Stripping units prevents 'electron/s vs ppm' mismatch errors when
    stitching sectors that were processed with different TESS pipelines.
    """
    flux_vals = np.array(lc.flux.value, dtype=float)
    p75 = np.nanpercentile(flux_vals, 75)
    divisor = p75 if abs(p75) > 1e-10 else (float(np.nanmedian(flux_vals)) or 1.0)
    norm_flux = flux_vals / divisor
    # Return a plain LightCurve with no units attached
    return lk.LightCurve(time=lc.time, flux=norm_flux)


def predict_transit_times(period, t0, t_min, t_max):
    n_lo = int(np.ceil((t_min - t0) / period))
    n_hi = int(np.floor((t_max - t0) / period))
    return t0 + np.arange(n_lo, n_hi + 1) * period


def fit_transit_midpoint(t_sec, f_sec, tc_nominal, half_dur):
    """
    Fit the midpoint of a single transit using minimize_scalar.
    Returns the fitted tc, or tc_nominal if fitting fails.
    """
    from scipy.optimize import minimize_scalar
    mask = np.abs(t_sec - tc_nominal) <= half_dur
    if mask.sum() < 5:
        return tc_nominal
    t_in = t_sec[mask]
    f_in = f_sec[mask]
    # Simple minimum: fit a parabola around the flux minimum
    mi = np.argmin(f_in)
    lo, hi = max(0, mi - 4), min(len(t_in), mi + 5)
    if hi - lo < 3:
        return tc_nominal
    try:
        coeffs = np.polyfit(t_in[lo:hi], f_in[lo:hi], 2)
        if coeffs[0] > 0:  # parabola opens upward — valid minimum
            tc_fit = -coeffs[1] / (2 * coeffs[0])
            # Sanity: must stay within the window
            if abs(tc_fit - tc_nominal) <= half_dur:
                return float(tc_fit)
    except Exception:
        pass
    return tc_nominal


def refine_t0(time_arr, flux_arr, period, t0_init, half_dur, min_pts=5):
    """
    Stack all transit windows in phase space, bin, fit a parabola to the
    flux minimum. Returns the t0 correction in days.
    More robust than using the raw phase-fold minimum bin.
    """
    phase_all, flux_all = [], []
    predicted = predict_transit_times(period, t0_init, time_arr.min(), time_arr.max())
    for tc in predicted:
        mask = np.abs(time_arr - tc) <= half_dur
        if mask.sum() >= min_pts:
            phase_all.extend((time_arr[mask] - tc).tolist())
            flux_all.extend(flux_arr[mask].tolist())

    if len(phase_all) < 20:
        return 0.0

    ph = np.array(phase_all)
    fl = np.array(flux_all)

    edges = np.linspace(-half_dur, half_dur, 41)
    bc, bf = [], []
    for i in range(40):
        m = (ph >= edges[i]) & (ph < edges[i+1])
        if m.sum() >= 2:
            bc.append(0.5 * (edges[i] + edges[i+1]))
            bf.append(float(np.median(fl[m])))

    if len(bc) < 5:
        return 0.0

    bc = np.array(bc)
    bf = np.array(bf)
    mi = np.argmin(bf)
    lo, hi = max(0, mi - 5), min(len(bc), mi + 6)
    coeffs = np.polyfit(bc[lo:hi], bf[lo:hi], 2)
    if abs(coeffs[0]) > 1e-12:
        return float(-coeffs[1] / (2 * coeffs[0]))
    return float(bc[mi])


def get_sector_time_ranges(host_star):
    """Returns {sector: (t_min_btjd, t_max_btjd)} from MAST search table."""
    results = lk.search_lightcurve(host_star, mission="TESS", exptime=120, author="SPOC")
    if results is None or len(results) == 0:
        return {}
    table = results.table
    ranges = {}
    for row in table:
        sec  = int(row["sequence_number"])
        tmin = float(row["t_min"]) + MJD_TO_BTJD
        tmax = float(row["t_max"]) + MJD_TO_BTJD
        if sec not in ranges:
            ranges[sec] = [tmin, tmax]
        else:
            ranges[sec][0] = min(ranges[sec][0], tmin)
            ranges[sec][1] = max(ranges[sec][1], tmax)
    return {s: tuple(v) for s, v in ranges.items()}


# ── Per-planet processing ─────────────────────────────────────────────────────

def process_planet(row, min_pts):
    name    = row["System"]
    host    = get_host(row)
    period  = float(row["Period"])
    t0_bjd  = float(row["pl_tranmid"])
    t0      = t0_bjd - BJD_OFFSET          # convert to BTJD
    trandur = float(row["pl_trandur"])     # hours
    half_dur = trandur / 24 / 2           # days

    planet_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(planet_dir, exist_ok=True)

    print(f"  Downloading {name} (host={host})...")
    for attempt in range(1, 5):
        try:
            results    = lk.search_lightcurve(host, mission="TESS", exptime=120, author="SPOC")
            collection = results.download_all(quality_bitmask="default",
                                              flux_column="pdcsap_flux")
            lc_full    = collection.stitch(corrector_func=robust_normalize)
            lc_full    = lc_full.remove_nans()
            lc_full    = lc_full.remove_outliers(sigma_lower=1e6, sigma_upper=4.0)
            break
        except Exception as e:
            print(f"  Download attempt {attempt}/4 failed: {e}")
            if attempt == 4:
                raise
            wait = attempt * 10
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    # Sanity-check normalization
    flux_median = float(np.nanmedian(lc_full.flux.value))
    if not (0.5 < flux_median < 2.0):
        lc_full = lc_full / flux_median

    time_arr = lc_full.time.value
    flux_arr = lc_full.flux.value

    # Auto-refine t0: stack all transits, fit parabola to minimum
    try:
        offset = refine_t0(time_arr, flux_arr, period, t0, half_dur)
        t0 = t0 + offset
        print(f"  t0 refined by {offset*24*60:+.1f} min")
    except Exception as e:
        print(f"  t0 refinement failed ({e}) — using catalog t0")

    # Get real sector time ranges from MAST
    sector_ranges = get_sector_time_ranges(host)

    # Split light curve by sector
    sector_lcs = {}
    for sec, (smin, smax) in sector_ranges.items():
        mask = (time_arr >= smin) & (time_arr <= smax)
        if mask.sum() > 0:
            sector_lcs[sec] = (time_arr[mask], flux_arr[mask])

    # ── Count usable transits per sector (with per-sector t0 refinement) ────────
    # A transit is USABLE only if all three windows have >= min_pts:
    #   pre-baseline  [tc - 3*half_dur, tc - half_dur]   width = T
    #   in-transit    [tc - half_dur,   tc + half_dur]   width = T
    #   post-baseline [tc + half_dur,   tc + 3*half_dur] width = T
    sector_rows = []
    sector_t0   = {}   # store refined t0 per sector for use in image generation

    for sec in sorted(sector_lcs.keys()):
        t_sec, f_sec = sector_lcs[sec]

        # Per-sector t0 refinement on top of the global correction
        try:
            sec_offset = refine_t0(t_sec, f_sec, period, t0, half_dur)
            t0_sec = t0 + sec_offset
        except Exception:
            t0_sec = t0
            sec_offset = 0.0
        sector_t0[sec] = t0_sec

        predicted = predict_transit_times(period, t0_sec, t_sec.min(), t_sec.max())
        n_pred = len(predicted)
        n_obs  = 0
        n_use  = 0
        pts_in = []

        for tc in predicted:
            n_pre  = int(np.sum((t_sec >= tc - 3*half_dur) & (t_sec <  tc - half_dur)))
            n_in   = int(np.sum(np.abs(t_sec - tc) <= half_dur))
            n_post = int(np.sum((t_sec >  tc + half_dur)   & (t_sec <= tc + 3*half_dur)))

            if n_in >= 1:
                n_obs += 1
            if n_pre >= min_pts and n_in >= min_pts and n_post >= min_pts:
                n_use += 1
                pts_in.append(n_in)

        med_pts = float(np.median(pts_in)) if pts_in else 0.0
        sector_rows.append({
            "sector":      sec,
            "t0_sec_btjd": round(t0_sec, 6),
            "t0_offset_min": round(sec_offset * 24 * 60, 2),
            "n_predicted": n_pred,
            "n_observed":  n_obs,
            "n_usable":    n_use,
            "median_pts":  round(med_pts, 1),
        })

    df_report = pd.DataFrame(sector_rows)

    # Add Total row at the bottom
    total_row = {
        "sector":          "Total",
        "t0_sec_btjd":     "",
        "t0_offset_min":   "",
        "n_predicted":     int(df_report["n_predicted"].sum()),
        "n_observed":      int(df_report["n_observed"].sum()),
        "n_usable":        int(df_report["n_usable"].sum()),
        "median_pts":      round(float(df_report["median_pts"].mean()), 1),
    }
    df_report = pd.concat([df_report, pd.DataFrame([total_row])], ignore_index=True)
    df_report.to_csv(os.path.join(planet_dir, "sector_report.csv"), index=False)

    total_usable = int(df_report.loc[df_report["sector"] == "Total", "n_usable"].iloc[0])
    print(f"  Sectors={len(sector_lcs)}  Total usable transits={total_usable}")

    # ── One image per sector (uses per-sector refined t0) ─────────────────────
    n_imgs = 0
    for sec in sorted(sector_lcs.keys()):
        t_sec, f_sec = sector_lcs[sec]
        t0_sec    = sector_t0.get(sec, t0)   # use per-sector refined t0
        predicted = predict_transit_times(period, t0_sec, t_sec.min(), t_sec.max())

        n_use_sec = 0
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t_sec, f_sec, ".", ms=1.2, color="#555555", alpha=0.6, rasterized=True)

        for tc in predicted:
            n_pre  = int(np.sum((t_sec >= tc - 3*half_dur) & (t_sec <  tc - half_dur)))
            n_in   = int(np.sum(np.abs(t_sec - tc)         <= half_dur))
            n_post = int(np.sum((t_sec >  tc + half_dur)   & (t_sec <= tc + 3*half_dur)))

            # Fit each transit's midpoint individually so the line tracks the
            # actual flux minimum rather than the (period-drifted) prediction
            tc_fit = fit_transit_midpoint(t_sec, f_sec, tc, half_dur)

            if n_pre >= min_pts and n_in >= min_pts and n_post >= min_pts:
                # Full triplet usable — shade windows around the fitted midpoint
                ax.axvspan(tc_fit - 3*half_dur, tc_fit - half_dur,
                           alpha=0.10, color="#3498DB", zorder=0)   # pre
                ax.axvspan(tc_fit - half_dur,   tc_fit + half_dur,
                           alpha=0.18, color="#E74C3C", zorder=0)   # transit
                ax.axvspan(tc_fit + half_dur,   tc_fit + 3*half_dur,
                           alpha=0.10, color="#2ECC71", zorder=0)   # post
                ax.axvline(tc_fit, color="#E74C3C", lw=0.8, alpha=0.9)
                n_use_sec += 1
            elif n_in >= min_pts:
                # In-transit has data but pre or post missing — partial
                ax.axvspan(tc_fit - half_dur, tc_fit + half_dur,
                           alpha=0.10, color="#F39C12", zorder=0)
                ax.axvline(tc_fit, color="#F39C12", lw=0.6, alpha=0.7, ls="--")

        ax.set_xlabel("BTJD (days)", fontsize=9)
        ax.set_ylabel("Norm. Flux", fontsize=9)
        sec_offset_min = (t0_sec - t0) * 24 * 60
        ax.set_title(
            f"{name}  —  Sector {sec:02d}  |  "
            f"{n_use_sec} usable  |  "
            f"P={period:.4f}d  dur={trandur:.2f}h  "
            f"t0 offset={sec_offset_min:+.1f} min",
            fontsize=8, fontweight="bold"
        )
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
        plt.tight_layout()

        img_path = os.path.join(planet_dir, f"sector_s{sec:03d}.png")
        plt.savefig(img_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        n_imgs += 1

    print(f"  Saved {n_imgs} sector images → {planet_dir}/")

    return {
        "System":        name,
        "n_sectors":     len(sector_lcs),
        "total_usable":  int(total_usable),
        "t0_btjd":       round(t0, 6),
        "status":        "ok",
        **{f"S{r['sector']:03d}": r["n_usable"] for r in sector_rows}
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet",   type=str, default=None)
    parser.add_argument("--min-pts",  type=int, default=MIN_PTS)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(PRIORITY_CSV)
    if args.planet:
        df = df[df["System"].str.contains(args.planet, case=False)]
        if df.empty:
            print(f"Planet '{args.planet}' not found.")
            return

    # Smart resume — skip planets that already have a sector_report.csv
    done = set()
    if os.path.exists(SUMMARY_CSV):
        existing = pd.read_csv(SUMMARY_CSV)
        done = set(existing[existing["status"] == "ok"]["System"].tolist())
        print(f"Resuming — {len(done)} planets already done.")

    all_rows = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        name = row["System"]
        if name in done:
            print(f"[{i+1}/{total}] {name} — already done, skipping")
            continue

        print(f"\n[{i+1}/{total}] {name}")
        try:
            result = process_planet(row, args.min_pts)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "System":       name,
                "n_sectors":    0,
                "total_usable": 0,
                "t0_btjd":      float("nan"),
                "status":       f"error: {e}",
            }

        all_rows.append(result)

        # Save summary after every planet
        batch = pd.DataFrame(all_rows)
        if os.path.exists(SUMMARY_CSV) and len(done) > 0:
            existing = pd.read_csv(SUMMARY_CSV)
            combined = pd.concat([existing, batch], ignore_index=True)
            combined = combined.drop_duplicates(subset="System", keep="last")
        else:
            combined = batch
        combined.to_csv(SUMMARY_CSV, index=False)

        time.sleep(SLEEP)

    # Final sorted summary — add a Total row
    final = pd.read_csv(SUMMARY_CSV)
    ok = final[final["status"] == "ok"].sort_values("total_usable", ascending=False)

    # Build Total row: sum all numeric columns
    sector_cols = [c for c in final.columns if c.startswith("S") and c[1:].isdigit()]
    total_row = {"System": "Total", "status": "summary"}
    total_row["n_sectors"]    = int(ok["n_sectors"].sum())
    total_row["total_usable"] = int(ok["total_usable"].sum())
    for col in sector_cols:
        if col in ok.columns:
            total_row[col] = int(ok[col].sum())

    final_with_total = pd.concat(
        [ok, pd.DataFrame([total_row])],
        ignore_index=True
    )
    final_with_total.to_csv(SUMMARY_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Survey complete — {len(ok)} planets ok, "
          f"{len(final)-len(ok)} errors")
    print(f"\nTop 10 by total usable transits:")
    print(ok[["System", "n_sectors", "total_usable"]].head(10).to_string(index=False))
    print(f"\nGrand total usable transits: {int(ok['total_usable'].sum())}")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Summary CSV:      {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

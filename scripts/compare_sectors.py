#!/usr/bin/env python3
"""
compare_sectors.py — Compare out-of-transit baseline scatter between two TESS sectors.

For each planet present in both sectors, downloads the (cached) light curve,
extracts out-of-transit baseline regions, and computes the RMS scatter.
Lower RMS = less noise = better sector for that planet.

Usage:
    python compare_sectors.py
    python compare_sectors.py --sec-a 75 --sec-b 41
"""

import argparse
import os
import re
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIORITY_CSV = "planets_priority.csv"
RESULTS_DIR  = "results"
BJD_OFFSET   = 2457000.0
MJD_TO_BTJD  = 2400000.5 - BJD_OFFSET
MIN_PTS      = 10


def get_host(row):
    host = str(row.get("host_star", row.get("System", ""))).strip()
    return re.sub(r"\s+[A-D]$", "", host).strip()


def robust_normalize(lc):
    flux_vals = np.array(lc.flux.value, dtype=float)
    p75 = np.nanpercentile(flux_vals, 75)
    divisor = p75 if abs(p75) > 1e-10 else (float(np.nanmedian(flux_vals)) or 1.0)
    return lk.LightCurve(time=lc.time, flux=flux_vals / divisor)


def predict_transit_times(period, t0, t_min, t_max):
    n_lo = int(np.ceil((t_min - t0) / period))
    n_hi = int(np.floor((t_max - t0) / period))
    return t0 + np.arange(n_lo, n_hi + 1) * period


def get_sector_time_ranges(host_star):
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


def baseline_rms(t_sec, f_sec, period, t0, half_dur):
    """
    Compute RMS of out-of-transit flux points.
    Masks out all predicted transit windows (±half_dur) before computing.
    Returns (rms, n_points).
    """
    predicted = predict_transit_times(period, t0, t_sec.min(), t_sec.max())
    in_transit = np.zeros(len(t_sec), dtype=bool)
    for tc in predicted:
        in_transit |= np.abs(t_sec - tc) <= half_dur * 1.5   # slightly wider mask

    baseline = f_sec[~in_transit]
    baseline = baseline[np.isfinite(baseline)]

    if len(baseline) < 20:
        return np.nan, 0

    # Remove slow trends with a linear fit before computing scatter
    t_base = t_sec[~in_transit][np.isfinite(f_sec[~in_transit])]
    c = np.polyfit(t_base - t_base.mean(), baseline, 1)
    detrended = baseline - np.polyval(c, t_base - t_base.mean())

    return float(np.std(detrended)), len(detrended)


def process_planet(row, sec_a, sec_b):
    name    = row["System"]
    host    = get_host(row)
    period  = float(row["Period"])
    t0_bjd  = float(row["pl_tranmid"])
    t0      = t0_bjd - BJD_OFFSET
    trandur = float(row["pl_trandur"])
    half_dur = trandur / 24 / 2

    # Check that this planet has usable transits in both sectors
    report_path = os.path.join(RESULTS_DIR, name, "sector_report.csv")
    if not os.path.exists(report_path):
        return None

    rep = pd.read_csv(report_path)
    rep = rep[rep["sector"] != "Total"].copy()
    rep["sector"] = rep["sector"].astype(int)

    def usable_in(sec):
        r = rep[rep["sector"] == sec]
        return int(r["n_usable"].values[0]) if len(r) else 0

    n_a = usable_in(sec_a)
    n_b = usable_in(sec_b)

    if n_a == 0 and n_b == 0:
        return None   # planet not in either sector

    # Download (from cache)
    try:
        results    = lk.search_lightcurve(host, mission="TESS", exptime=120, author="SPOC")
        collection = results.download_all(quality_bitmask="default", flux_column="pdcsap_flux")
        lc_full    = collection.stitch(corrector_func=robust_normalize)
        lc_full    = lc_full.remove_nans()
        lc_full    = lc_full.remove_outliers(sigma_lower=1e6, sigma_upper=4.0)
    except Exception as e:
        print(f"  {name}: download failed — {e}")
        return None

    flux_median = float(np.nanmedian(lc_full.flux.value))
    if not (0.5 < flux_median < 2.0):
        lc_full = lc_full / flux_median

    time_arr = lc_full.time.value
    flux_arr = lc_full.flux.value

    sector_ranges = get_sector_time_ranges(host)

    def get_sector_arrays(sec):
        if sec not in sector_ranges:
            return None, None
        smin, smax = sector_ranges[sec]
        mask = (time_arr >= smin) & (time_arr <= smax)
        if mask.sum() == 0:
            return None, None
        return time_arr[mask], flux_arr[mask]

    t_a, f_a = get_sector_arrays(sec_a)
    t_b, f_b = get_sector_arrays(sec_b)

    rms_a, n_pts_a = (baseline_rms(t_a, f_a, period, t0, half_dur)
                      if t_a is not None else (np.nan, 0))
    rms_b, n_pts_b = (baseline_rms(t_b, f_b, period, t0, half_dur)
                      if t_b is not None else (np.nan, 0))

    return {
        "planet":     name,
        f"S{sec_a:02d}_usable":   n_a,
        f"S{sec_b:02d}_usable":   n_b,
        f"S{sec_a:02d}_rms":      round(rms_a * 1e6, 1) if np.isfinite(rms_a) else np.nan,
        f"S{sec_b:02d}_rms":      round(rms_b * 1e6, 1) if np.isfinite(rms_b) else np.nan,
        f"S{sec_a:02d}_pts":      n_pts_a,
        f"S{sec_b:02d}_pts":      n_pts_b,
        "winner":     (f"S{sec_a:02d}" if rms_a < rms_b
                       else f"S{sec_b:02d}" if rms_b < rms_a
                       else "tie") if np.isfinite(rms_a) and np.isfinite(rms_b) else "—",
        "rms_diff_ppm": (round((rms_b - rms_a) * 1e6, 1)
                         if np.isfinite(rms_a) and np.isfinite(rms_b) else np.nan),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec-a", type=int, default=75)
    parser.add_argument("--sec-b", type=int, default=41)
    args = parser.parse_args()

    sec_a, sec_b = args.sec_a, args.sec_b
    print(f"Comparing Sector {sec_a:02d} vs Sector {sec_b:02d}")
    print(f"(RMS in ppm of detrended out-of-transit baseline)\n")

    df = pd.read_csv(PRIORITY_CSV)
    rows = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        name = row["System"]
        print(f"[{i+1}/{total}] {name}...", end=" ", flush=True)
        result = process_planet(row, sec_a, sec_b)
        if result is None:
            print("skipped (not in both sectors or missing report)")
            continue
        rows.append(result)
        rms_a = result[f"S{sec_a:02d}_rms"]
        rms_b = result[f"S{sec_b:02d}_rms"]
        print(f"S{sec_a:02d}={rms_a} ppm  S{sec_b:02d}={rms_b} ppm  → {result['winner']}")

    if not rows:
        print("No planets found in both sectors.")
        return

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, f"sector_compare_S{sec_a:02d}_vs_S{sec_b:02d}.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── Summary ──────────────────────────────────────────────────────────────
    both = df_out.dropna(subset=[f"S{sec_a:02d}_rms", f"S{sec_b:02d}_rms"])
    wins_a = (both["winner"] == f"S{sec_a:02d}").sum()
    wins_b = (both["winner"] == f"S{sec_b:02d}").sum()
    ties   = (both["winner"] == "tie").sum()

    print(f"\n{'='*55}")
    print(f"Summary ({len(both)} planets with data in both sectors):")
    print(f"  S{sec_a:02d} wins (lower RMS): {wins_a} planets")
    print(f"  S{sec_b:02d} wins (lower RMS): {wins_b} planets")
    print(f"  Ties:                    {ties} planets")
    print(f"\n  Median RMS  S{sec_a:02d}: {both[f'S{sec_a:02d}_rms'].median():.1f} ppm")
    print(f"  Median RMS  S{sec_b:02d}: {both[f'S{sec_b:02d}_rms'].median():.1f} ppm")
    print(f"\n  Median diff (S{sec_b:02d}−S{sec_a:02d}): {both['rms_diff_ppm'].median():+.1f} ppm")
    print(f"  (positive = S{sec_a:02d} is quieter on average)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(both) * 0.32)))

    # Left: side-by-side RMS per planet
    planets = both["planet"].tolist()
    rms_a_vals = both[f"S{sec_a:02d}_rms"].values
    rms_b_vals = both[f"S{sec_b:02d}_rms"].values
    y = np.arange(len(planets))
    h = 0.35

    ax = axes[0]
    ax.barh(y + h/2, rms_a_vals, h, color="#2C7BB6", label=f"S{sec_a:02d}", alpha=0.85)
    ax.barh(y - h/2, rms_b_vals, h, color="#D7191C", label=f"S{sec_b:02d}", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(planets, fontsize=8)
    ax.set_xlabel("Baseline RMS (ppm)", fontsize=9)
    ax.set_title(f"Out-of-transit Baseline Scatter\nS{sec_a:02d} vs S{sec_b:02d}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: scatter plot S_a RMS vs S_b RMS (diagonal = equal noise)
    ax2 = axes[1]
    ax2.scatter(rms_a_vals, rms_b_vals, color="#555555", s=40, zorder=3)
    for planet, xa, xb in zip(planets, rms_a_vals, rms_b_vals):
        ax2.annotate(planet, (xa, xb), fontsize=6.5,
                     textcoords="offset points", xytext=(4, 2))
    lim_max = max(np.nanmax(rms_a_vals), np.nanmax(rms_b_vals)) * 1.1
    ax2.plot([0, lim_max], [0, lim_max], "k--", lw=1, alpha=0.4, label="Equal noise")
    ax2.fill_between([0, lim_max], [0, 0], [0, lim_max],
                     alpha=0.06, color="#2C7BB6", label=f"S{sec_a:02d} quieter")
    ax2.fill_between([0, lim_max], [0, lim_max], [lim_max, lim_max],
                     alpha=0.06, color="#D7191C", label=f"S{sec_b:02d} quieter")
    ax2.set_xlabel(f"S{sec_a:02d} RMS (ppm)", fontsize=9)
    ax2.set_ylabel(f"S{sec_b:02d} RMS (ppm)", fontsize=9)
    ax2.set_title(f"S{sec_a:02d} vs S{sec_b:02d} — Noise Comparison\n"
                  f"(below diagonal = S{sec_a:02d} quieter)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, lim_max); ax2.set_ylim(0, lim_max)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR, f"sector_compare_S{sec_a:02d}_vs_S{sec_b:02d}.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_png}")

    # ── Recommendation ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    winner_sec = sec_a if wins_a >= wins_b else sec_b
    loser_sec  = sec_b if winner_sec == sec_a else sec_a
    med_a = both[f"S{sec_a:02d}_rms"].median()
    med_b = both[f"S{sec_b:02d}_rms"].median()
    print(f"RECOMMENDATION: Focus on Sector {winner_sec:02d}")
    print(f"  Lower baseline RMS in {wins_a if winner_sec==sec_a else wins_b}/{len(both)} planets")
    print(f"  Median RMS S{sec_a:02d}={med_a:.1f}  S{sec_b:02d}={med_b:.1f} ppm")
    print(f"\nPer-planet recommendation:")
    for _, r in df_out.iterrows():
        if r["winner"] not in ("—", "tie"):
            ra = r[f"S{sec_a:02d}_rms"]
            rb = r[f"S{sec_b:02d}_rms"]
            if np.isfinite(ra) and np.isfinite(rb):
                print(f"  {r['planet']:<22} → S{r['winner'][1:]}  "
                      f"(S{sec_a:02d}={ra:.0f}  S{sec_b:02d}={rb:.0f} ppm  "
                      f"Δ={abs(ra-rb):.0f} ppm)")


if __name__ == "__main__":
    main()

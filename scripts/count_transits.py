#!/usr/bin/env python3
"""
count_transits.py — Count actual usable TESS transits for priority planets

For each planet in planets_priority.csv:
  1. Download TESS PDC-SAP light curve via lightkurve
  2. Use pl_tranmid + Period to compute all predicted transit times
  3. Count how many predicted transits have enough in-transit data points
  4. Save summary to transit_counts.csv

Supports smart resume — already-processed planets are skipped on re-run.

Usage:
    python count_transits.py                  # all 42 priority planets
    python count_transits.py --planet TOI-157 # single planet
    python count_transits.py --min-points 10  # change minimum points threshold
"""

import argparse
import ast
import os
import time
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import numpy as np
import pandas as pd

CATALOG_PATH  = "planets_priority.csv"
OUTPUT_PATH   = "transit_counts.csv"
BJD_OFFSET    = 2457000.0   # BTJD = BJD - 2457000
SLEEP_BETWEEN = 2.0         # seconds between MAST queries
MIN_POINTS    = 10          # minimum in-transit data points to count as usable


def load_catalog():
    df = pd.read_csv(CATALOG_PATH)
    # Convert pl_tranmid from BJD to BTJD
    df["t0_btjd"] = df["pl_tranmid"] - BJD_OFFSET
    return df


def get_host(row):
    """Use NEA hostname if available, else fall back to host_star from TEPCat.
    Strips binary star suffixes like ' A', ' B', ' C' that lightkurve cannot resolve.
    """
    import re
    if pd.notna(row.get("hostname")):
        host = str(row["hostname"]).strip()
    else:
        host = str(row["host_star"]).strip()
    # Strip trailing single-letter component labels: 'TOI-1259 A' → 'TOI-1259'
    host = re.sub(r'\s+[A-D]$', '', host).strip()
    return host


def download_lc(host_star):
    results = lk.search_lightcurve(host_star, mission="TESS", exptime=120)
    if results is None or len(results) == 0:
        raise ValueError(f"No TESS 2-min data found for '{host_star}'")
    collection = results.download_all(quality_bitmask="default")
    lc = collection.stitch(corrector_func=lambda x: x.normalize())
    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma_lower=1e6, sigma_upper=4.0)
    return lc


def count_usable_transits(time_arr, period, t0_btjd, trandur_hours, min_points=MIN_POINTS):
    """
    Returns (n_predicted, n_observed, n_usable, median_points_in_window).

    n_predicted  — transit times falling within the light curve time span
    n_observed   — predicted transits with at least 1 data point in window
    n_usable     — predicted transits with >= min_points in window
    """
    half_dur = (trandur_hours / 24.0) / 2.0  # days

    t_min, t_max = time_arr.min(), time_arr.max()

    # Compute all transit epochs within the light curve span
    n_lo = int(np.ceil((t_min - t0_btjd) / period))
    n_hi = int(np.floor((t_max - t0_btjd) / period))

    if n_hi < n_lo:
        return 0, 0, 0, 0.0

    epochs       = np.arange(n_lo, n_hi + 1)
    transit_times = t0_btjd + epochs * period

    n_predicted = len(transit_times)
    n_observed  = 0
    n_usable    = 0
    point_counts = []

    for tc in transit_times:
        mask = np.abs(time_arr - tc) <= half_dur
        n_pts = int(mask.sum())
        point_counts.append(n_pts)
        if n_pts >= 1:
            n_observed += 1
        if n_pts >= min_points:
            n_usable += 1

    observed_counts = [p for p in point_counts if p > 0]
    median_pts = float(np.median(observed_counts)) if observed_counts else 0.0
    return n_predicted, n_observed, n_usable, median_pts


def process_planet(row, min_points):
    host   = get_host(row)
    name   = row["System"]
    period = float(row["Period"])
    t0     = float(row["t0_btjd"])
    dur    = float(row["pl_trandur"])   # hours

    print(f"  [{name}]  host={host}  P={period:.3f}d  dur={dur:.2f}h")

    lc = download_lc(host)
    time_arr = lc.time.value
    span = time_arr[-1] - time_arr[0]
    n_lc_points = len(time_arr)

    n_pred, n_obs, n_use, med_pts = count_usable_transits(
        time_arr, period, t0, dur, min_points
    )

    print(f"    LC span={span:.1f}d  points={n_lc_points:,}")
    print(f"    predicted={n_pred}  observed={n_obs}  usable(>={min_points}pts)={n_use}  median_pts={med_pts:.0f}")

    return {
        "System":          name,
        "host_star":       host,
        "n_sectors":       int(row["n_sectors"]),
        "Period":          period,
        "pl_trandur":      dur,
        "lc_span_days":    round(span, 1),
        "lc_n_points":     n_lc_points,
        "n_predicted":     n_pred,
        "n_observed":      n_obs,
        "n_usable":        n_use,
        "median_pts_in_window": round(med_pts, 1),
        "est_transits":    int(row["n_sectors"] * 27.4 / period),
        "status":          "ok",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet",     type=str,  default=None,
                        help="Process a single planet by System name")
    parser.add_argument("--min-points", type=int,  default=MIN_POINTS,
                        help=f"Minimum in-transit points to count as usable (default {MIN_POINTS})")
    args = parser.parse_args()

    df = load_catalog()

    if args.planet:
        df = df[df["System"].str.contains(args.planet, case=False)]
        if df.empty:
            print(f"Planet '{args.planet}' not found in {CATALOG_PATH}")
            return

    # Smart resume — load existing results and skip already-done planets
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done = set(existing["System"].tolist())
        print(f"Resuming — {len(done)} planets already processed, skipping them.")
    else:
        existing = pd.DataFrame()
        done = set()

    rows = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        name = row["System"]
        if name in done:
            print(f"  [{i+1}/{total}] {name} — already done, skipping")
            continue

        print(f"\n[{i+1}/{total}] Processing {name} ...")

        try:
            result = process_planet(row, args.min_points)
        except Exception as e:
            print(f"    ERROR: {e}")
            result = {
                "System":               name,
                "host_star":            get_host(row),
                "n_sectors":            int(row["n_sectors"]),
                "Period":               float(row["Period"]),
                "pl_trandur":           float(row["pl_trandur"]),
                "lc_span_days":         None,
                "lc_n_points":          None,
                "n_predicted":          None,
                "n_observed":           None,
                "n_usable":             None,
                "median_pts_in_window": None,
                "est_transits":         int(row["n_sectors"] * 27.4 / row["Period"]),
                "status":               f"error: {e}",
            }

        rows.append(result)

        # Save after every planet so progress is never lost
        batch = pd.DataFrame(rows)
        combined = pd.concat([existing, batch], ignore_index=True) if not existing.empty else batch
        for _attempt in range(5):
            try:
                combined.to_csv(OUTPUT_PATH, index=False)
                break
            except PermissionError:
                print(f"    [save] transit_counts.csv is locked — retrying in 3s (close Excel if open)")
                time.sleep(3)

        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. Results saved to {OUTPUT_PATH}")

    # Print summary table
    final = pd.read_csv(OUTPUT_PATH)
    ok = final[final["status"] == "ok"].sort_values("n_usable", ascending=False)
    print("\nUsable transit counts (sorted):")
    print(ok[["System", "n_sectors", "Period", "est_transits",
              "n_usable", "median_pts_in_window"]].to_string(index=False))


if __name__ == "__main__":
    main()

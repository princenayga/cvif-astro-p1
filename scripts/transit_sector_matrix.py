#!/usr/bin/env python3
"""
transit_sector_matrix.py — Build a planet x TESS-sector transit count matrix

For each priority planet:
  1. Gets actual TESS sector numbers + time boundaries from lightkurve search table
  2. Generates transit ephemeris (t0 + n*Period)
  3. Assigns each predicted transit to its sector
  4. Counts transits per sector

Outputs:
  transit_sector_matrix.csv  — wide format: planets x sectors, values = transit count
  transit_sector_matrix.png  — heatmap of the matrix

No light curve downloads needed — uses only the MAST search metadata.

Usage:
    python transit_sector_matrix.py
"""

import re
import time
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import numpy as np
import pandas as pd

PRIORITY_CSV  = "planets_priority.csv"
LOOKUP_CSV    = "sector_lookup.csv"
OUT_CSV       = "transit_sector_matrix.csv"
BJD_OFFSET    = 2457000.0
MJD_TO_BTJD  = 2400000.5 - BJD_OFFSET   # MJD + this = BTJD
SLEEP         = 1.0
MIN_POINTS    = 10    # minimum points to count transit as usable (approx: sector coverage implies data)


def get_host(row):
    host = str(row.get("host_star", row.get("System", ""))).strip()
    return re.sub(r'\s+[A-D]$', '', host).strip()


def get_sector_time_ranges(host_star):
    """
    Returns dict: {sector_number: (t_min_btjd, t_max_btjd)}
    Uses MAST search table only — no light curve download.
    """
    results = lk.search_lightcurve(host_star, mission="TESS", exptime=120)
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


def count_transits_per_sector(period, t0_btjd, sector_ranges):
    """
    For each TESS sector, count how many transit times fall within its window.
    Returns dict: {sector: count}
    """
    if not sector_ranges:
        return {}

    all_t_min = min(v[0] for v in sector_ranges.values())
    all_t_max = max(v[1] for v in sector_ranges.values())

    n_lo = int(np.ceil((all_t_min - t0_btjd) / period))
    n_hi = int(np.floor((all_t_max - t0_btjd) / period))

    transit_times = t0_btjd + np.arange(n_lo, n_hi + 1) * period

    counts = {s: 0 for s in sector_ranges}
    for tc in transit_times:
        for sec, (smin, smax) in sector_ranges.items():
            if smin <= tc <= smax:
                counts[sec] += 1
                break

    return counts


def main():
    df_priority = pd.read_csv(PRIORITY_CSV)
    df_priority["t0_btjd"] = df_priority["pl_tranmid"] - BJD_OFFSET

    # Load sector lookup if available
    try:
        lookup = pd.read_csv(LOOKUP_CSV)
        lookup_map = dict(zip(lookup["System"], lookup["host_star"]))
        print(f"Loaded sector_lookup.csv ({len(lookup)} planets)")
    except FileNotFoundError:
        lookup_map = {}
        print("sector_lookup.csv not found — will use host_star from priority CSV")

    all_results = []   # list of dicts: {System, sector, n_transits}
    all_sectors = set()

    total = len(df_priority)
    for i, (_, row) in enumerate(df_priority.iterrows()):
        name   = row["System"]
        host   = lookup_map.get(name, get_host(row))
        period = float(row["Period"])
        t0     = float(row["t0_btjd"])

        print(f"[{i+1}/{total}] {name} ({host})...", end=" ", flush=True)
        try:
            ranges = get_sector_time_ranges(host)
            counts = count_transits_per_sector(period, t0, ranges)
            total_transits = sum(counts.values())
            all_sectors.update(counts.keys())
            all_results.append({"System": name, "counts": counts,
                                 "total": total_transits})
            print(f"{len(counts)} sectors, {total_transits} predicted transits")
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({"System": name, "counts": {}, "total": 0})

        time.sleep(SLEEP)

    # ── Build wide-format matrix ────────────────────────────────────────────────
    sector_cols = sorted(all_sectors)

    rows = []
    for r in all_results:
        row_dict = {"System": r["System"]}
        for s in sector_cols:
            row_dict[f"S{s:03d}"] = r["counts"].get(s, 0)
        row_dict["Total"] = r["total"]
        rows.append(row_dict)

    matrix = pd.DataFrame(rows)
    matrix = matrix.sort_values("Total", ascending=False).reset_index(drop=True)
    matrix.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}  ({len(matrix)} planets x {len(sector_cols)} sectors)")

    # ── Print summary table ────────────────────────────────────────────────────
    print("\nTransit counts per sector (non-zero only):")
    print(f"{'Planet':<20} {'Total':>6}  Sector breakdown")
    print("-" * 80)
    for _, row in matrix.iterrows():
        breakdown = {
            sector_cols[i]: int(row[f"S{sector_cols[i]:03d}"])
            for i in range(len(sector_cols))
            if int(row[f"S{sector_cols[i]:03d}"]) > 0
        }
        print(f"{row['System']:<20} {int(row['Total']):>6}  {breakdown}")


if __name__ == "__main__":
    main()

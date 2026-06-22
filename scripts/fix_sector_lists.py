#!/usr/bin/env python3
"""
fix_sector_lists.py — Replace 0-indexed sector_list with real TESS sector numbers

Queries lightkurve for each planet in planets_priority.csv, retrieves actual
TESS sector numbers from sequence_number column, then patches sector_list and
n_sectors in all relevant CSV files.

Saves real sector numbers to sector_lookup.csv so this only needs to run once.

Usage:
    python fix_sector_lists.py
"""

import re
import time
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import pandas as pd

PRIORITY_CSV = "planets_priority.csv"
LOOKUP_CSV   = "sector_lookup.csv"     # cache so we don't re-query
SLEEP        = 1.5

CSVS_TO_PATCH = [
    "planets_ready_for_modeling.csv",
    "tess_coverage_raw.csv",
    "planets_merged.csv",
    "planets_batman_complete.csv",
    "planets_priority.csv",
    "planets_conference_shortlist.csv",
]


def get_host(row):
    host = str(row.get("host_star", row.get("System", ""))).strip()
    host = re.sub(r'\s+[A-D]$', '', host).strip()
    return host


def query_sectors(host_star):
    results = lk.search_lightcurve(host_star, mission="TESS", exptime=120)
    if results is None or len(results) == 0:
        return []
    table = results.table
    return sorted(set(int(s) for s in table["sequence_number"] if s is not None))


def build_lookup():
    """Query lightkurve for every priority planet and cache results."""
    df = pd.read_csv(PRIORITY_CSV)

    # Load existing lookup if available (smart resume)
    try:
        existing = pd.read_csv(LOOKUP_CSV)
        done = set(existing["System"].tolist())
        rows = existing.to_dict("records")
        print(f"Resuming — {len(done)} planets already in lookup cache.")
    except FileNotFoundError:
        done = set()
        rows = []

    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        name = row["System"]
        if name in done:
            print(f"  [{i+1}/{total}] {name} — cached, skipping")
            continue

        host = get_host(row)
        print(f"  [{i+1}/{total}] {name} ({host})...", end=" ", flush=True)
        try:
            sectors = query_sectors(host)
            print(f"{sectors}")
        except Exception as e:
            sectors = []
            print(f"ERROR: {e}")

        rows.append({
            "System":      name,
            "host_star":   host,
            "sector_list": str(sectors),
            "n_sectors":   len(sectors),
        })
        done.add(name)

        # Save after every planet
        pd.DataFrame(rows).to_csv(LOOKUP_CSV, index=False)
        time.sleep(SLEEP)

    return pd.read_csv(LOOKUP_CSV)


def patch_csv(path, lookup):
    """Replace sector_list and n_sectors in a CSV using the lookup table."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  Skipping (not found): {path}")
        return

    if "sector_list" not in df.columns:
        print(f"  Skipping (no sector_list column): {path}")
        return

    # Match on System column if present, else host_star
    match_col = "System" if "System" in df.columns else "host_star"

    before = df["sector_list"].copy()
    updated = 0

    for _, lrow in lookup.iterrows():
        name = lrow["System"]
        mask = df[match_col] == name
        if mask.sum() == 0:
            # Try host_star match
            if "host_star" in df.columns:
                mask = df["host_star"] == lrow["host_star"]
        if mask.sum() > 0:
            df.loc[mask, "sector_list"] = lrow["sector_list"]
            df.loc[mask, "n_sectors"]   = lrow["n_sectors"]
            updated += mask.sum()

    df.to_csv(path, index=False)
    changed = (df["sector_list"] != before).sum()
    print(f"  Patched {path}: {updated} rows updated, {changed} sector_lists changed")


def main():
    print("=== Step 1: Build sector lookup from lightkurve ===")
    lookup = build_lookup()
    print(f"\nLookup complete: {len(lookup)} planets in {LOOKUP_CSV}")

    print("\n=== Step 2: Patch all CSV files ===")
    for csv_path in CSVS_TO_PATCH:
        patch_csv(csv_path, lookup)

    print("\nDone. Showing first 5 rows of updated planets_priority.csv:")
    df = pd.read_csv(PRIORITY_CSV)
    print(df[["System", "n_sectors", "sector_list"]].head())


if __name__ == "__main__":
    main()

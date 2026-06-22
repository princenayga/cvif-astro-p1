#!/usr/bin/env python3
"""
tess_best_planets.py

1. Filters the 33 TESS-suitable planets (excludes Kepler, K2, WD) from
   planets_priority.csv and saves them to planets_tess_best.csv

2. Reads already-computed survey results (results/{planet}/sector_report.csv)
   to report usable transit counts per planet and per sector.

No light curve downloads needed — reads from existing survey output.

Usage:
    python tess_best_planets.py
"""

import os
import pandas as pd

PRIORITY_CSV  = "planets_priority.csv"
OUT_CSV       = "planets_tess_best.csv"
RESULTS_DIR   = "results"

POOR_PREFIXES    = ("Kepler", "K2")
SPECIAL_PREFIXES = ("WD",)

# ── Step 1: Filter best planets ───────────────────────────────────────────────
df = pd.read_csv(PRIORITY_CSV)

def category(name):
    if any(name.startswith(p) for p in POOR_PREFIXES):
        return "poor"
    if any(name.startswith(p) for p in SPECIAL_PREFIXES):
        return "special"
    return "best"

df["tess_category"] = df["System"].apply(category)
df_best = df[df["tess_category"] == "best"].drop(columns=["tess_category"]).copy()
df_best = df_best.reset_index(drop=True)
df_best.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df_best)} best-for-TESS planets -> {OUT_CSV}")

excluded = df[df["tess_category"] != "best"]["System"].tolist()
print(f"Excluded ({len(excluded)}): {excluded}\n")

# ── Step 2: Load usable transit counts from survey results ────────────────────
rows = []
missing = []

for name in df_best["System"]:
    report_path = os.path.join(RESULTS_DIR, name, "sector_report.csv")
    if not os.path.exists(report_path):
        missing.append(name)
        rows.append({"planet": name, "n_sectors": 0, "total_usable": 0,
                     "sector_breakdown": "no survey data"})
        continue

    rep = pd.read_csv(report_path)
    data_rows = rep[rep["sector"] != "Total"]
    total_row = rep[rep["sector"] == "Total"]

    total_usable = int(total_row["n_usable"].values[0]) if len(total_row) else int(data_rows["n_usable"].sum())
    n_sectors    = int((data_rows["n_usable"] > 0).sum())

    breakdown = {
        f"S{int(r['sector']):02d}": int(r["n_usable"])
        for _, r in data_rows.iterrows()
        if int(r["n_usable"]) > 0
    }
    rows.append({
        "planet":           name,
        "n_sectors":        n_sectors,
        "total_usable":     total_usable,
        "sector_breakdown": str(breakdown),
    })

df_counts = pd.DataFrame(rows).sort_values("total_usable", ascending=False).reset_index(drop=True)

# ── Step 3: Print report ──────────────────────────────────────────────────────
print(f"{'Rank':<5} {'Planet':<22} {'Sectors':>8} {'Usable':>8}  Sector breakdown")
print("-" * 90)
for i, row in df_counts.iterrows():
    print(f"{i+1:<5} {row['planet']:<22} {row['n_sectors']:>8} {row['total_usable']:>8}  {row['sector_breakdown']}")

print("-" * 90)
ok = df_counts[df_counts["total_usable"] > 0]
print(f"{'TOTAL':<5} {'':<22} {'':>8} {int(df_counts['total_usable'].sum()):>8}")
print(f"\nPlanets with >=1 usable transit : {len(ok)} / {len(df_counts)}")
print(f"Planets with >=10 usable transits: {len(df_counts[df_counts['total_usable']>=10])} / {len(df_counts)}")
print(f"Planets with >=30 usable transits: {len(df_counts[df_counts['total_usable']>=30])} / {len(df_counts)}")

if missing:
    print(f"\nMissing survey data (run survey_planets.py first): {missing}")

# ── Step 4: Save count summary ────────────────────────────────────────────────
summary_path = os.path.join(RESULTS_DIR, "tess_best_transit_counts.csv")
df_counts.to_csv(summary_path, index=False)
print(f"\nCount summary saved -> {summary_path}")

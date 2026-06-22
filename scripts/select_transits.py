#!/usr/bin/env python3
"""
select_transits.py — Automatic transit selection plan for the 33 TESS-best planets.

Rules (in priority order):
  1. Find the single sector with the most usable transits for that planet.
     If it has >= TARGET, take TARGET transits from there. Done.
  2. If no single sector has >= TARGET, combine sectors starting from the
     richest, preferring same TESS year, until TARGET is reached.
  3. Skip planets with fewer than MIN_USABLE total usable transits.

Outputs:
  transit_selection_plan.csv   — one row per planet: source sector(s), how many
                                 transits to take from each, total
  transit_selection_plan.png   — visual summary

Usage:
    python select_transits.py
    python select_transits.py --target 10 --min-usable 10
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

RESULTS_DIR  = "results"
BEST_CSV     = "planets_tess_best.csv"
TARGET       = 10
MIN_USABLE   = 10

def sector_year(s):
    if s <= 13: return 1
    if s <= 26: return 2
    if s <= 39: return 3
    if s <= 55: return 4
    if s <= 69: return 5
    if s <= 83: return 6
    return 7

year_colors = {1:"#1f77b4",2:"#ff7f0e",3:"#2ca02c",
               4:"#d62728",5:"#9467bd",6:"#8c564b",7:"#e377c2"}


def load_sector_report(planet):
    path = os.path.join(RESULTS_DIR, planet, "sector_report.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    rep = pd.read_csv(path)
    rep = rep[rep["sector"] != "Total"].copy()
    rep["sector"] = rep["sector"].astype(int)
    rep = rep[rep["n_usable"] > 0].sort_values("n_usable", ascending=False)
    return rep


def select_sectors(rep, target):
    """
    Returns list of dicts: [{sector, n_take, year, n_usable, median_pts}]
    following the priority rules described in the module docstring.
    """
    if rep.empty:
        return []

    # Rule 1: single sector covers target
    best = rep.iloc[0]
    if best["n_usable"] >= target:
        return [{
            "sector":     int(best["sector"]),
            "n_take":     target,
            "year":       sector_year(int(best["sector"])),
            "n_usable":   int(best["n_usable"]),
            "median_pts": float(best["median_pts"]),
        }]

    # Rule 2: combine sectors — prefer same year as richest sector
    primary_year = sector_year(int(best["sector"]))
    remaining    = target
    plan         = []

    # Sort candidates: same year first (by n_usable desc), then other years
    same_year  = rep[rep["sector"].apply(sector_year) == primary_year]
    other_year = rep[rep["sector"].apply(sector_year) != primary_year].sort_values("n_usable", ascending=False)
    candidates = pd.concat([same_year, other_year])

    for _, row in candidates.iterrows():
        if remaining <= 0:
            break
        take = min(int(row["n_usable"]), remaining)
        plan.append({
            "sector":     int(row["sector"]),
            "n_take":     take,
            "year":       sector_year(int(row["sector"])),
            "n_usable":   int(row["n_usable"]),
            "median_pts": float(row["median_pts"]),
        })
        remaining -= take

    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     type=int, default=TARGET)
    parser.add_argument("--min-usable", type=int, default=MIN_USABLE)
    args = parser.parse_args()

    target     = args.target
    min_usable = args.min_usable

    df_best = pd.read_csv(BEST_CSV)
    planets = df_best["System"].tolist()

    rows      = []
    skip_list = []

    for planet in planets:
        rep = load_sector_report(planet)
        total_usable = int(rep["n_usable"].sum()) if not rep.empty else 0

        if total_usable < min_usable:
            skip_list.append((planet, total_usable))
            continue

        plan = select_sectors(rep, target)
        total_take   = sum(p["n_take"] for p in plan)
        sectors_used = [p["sector"] for p in plan]
        n_sectors    = len(plan)
        years_used   = sorted(set(p["year"] for p in plan))
        single       = (n_sectors == 1)

        # Sector breakdown string e.g. "S41:10" or "S41:8 + S75:2"
        breakdown = " + ".join(f"S{p['sector']:02d}:{p['n_take']}" for p in plan)

        rows.append({
            "planet":         planet,
            "total_usable":   total_usable,
            "transits_taken": total_take,
            "n_sectors_used": n_sectors,
            "single_sector":  single,
            "years_used":     str(years_used),
            "primary_sector": sectors_used[0],
            "selection":      breakdown,
            "median_pts":     round(np.mean([p["median_pts"] for p in plan]), 1),
            "plan":           plan,   # kept for plotting, dropped before CSV save
        })

    df_plan = pd.DataFrame(rows).sort_values("total_usable", ascending=False)

    # ── Console report ────────────────────────────────────────────────────────
    print(f"Target transits per planet : {target}")
    print(f"Min usable threshold       : {min_usable}")
    print(f"Planets included           : {len(df_plan)}")
    print(f"Planets skipped            : {len(skip_list)}")
    if skip_list:
        for p, n in skip_list:
            print(f"  {p} ({n} usable — below threshold)")
    print()

    single = df_plan[df_plan["single_sector"]]
    multi  = df_plan[~df_plan["single_sector"]]
    print(f"Single-sector selections   : {len(single)}")
    print(f"Multi-sector selections    : {len(multi)}")
    print()

    print(f"{'Planet':<22} {'Total':>6} {'Take':>5} {'Selection'}")
    print("-" * 70)
    for _, row in df_plan.iterrows():
        flag = "" if row["single_sector"] else " *"
        print(f"{row['planet']:<22} {row['total_usable']:>6} {row['transits_taken']:>5}  {row['selection']}{flag}")
    print("-" * 70)
    print(f"  * = multi-sector selection")
    print(f"\nTotal transits to process  : {int(df_plan['transits_taken'].sum())}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS_DIR, "transit_selection_plan.csv")
    df_plan.drop(columns=["plan"]).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plan_col = df_plan["plan"].tolist()
    planet_names = df_plan["planet"].tolist()
    n = len(planet_names)

    fig, ax = plt.subplots(figsize=(13, max(6, n * 0.38)))

    bottoms = np.zeros(n)
    # Collect all years present for legend
    all_years = sorted(set(
        p["year"] for row_plan in plan_col for p in row_plan
    ))
    year_legend_done = set()

    for row_plan, planet in zip(plan_col, planet_names):
        idx = planet_names.index(planet)
        for seg in row_plan:
            yr    = seg["year"]
            color = year_colors[yr]
            label = f"TESS Year {yr} (S{(yr-1)*13+1:02d}–S{yr*13:02d})" \
                    if yr not in year_legend_done else "_nolegend_"
            ax.barh(n - 1 - idx, seg["n_take"],
                    left=bottoms[idx],
                    color=color, edgecolor="white", linewidth=0.4,
                    alpha=0.88, label=label)
            # Label the segment with sector number if wide enough
            if seg["n_take"] >= 2:
                ax.text(bottoms[idx] + seg["n_take"] / 2,
                        n - 1 - idx,
                        f"S{seg['sector']:02d}",
                        ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")
            bottoms[idx] += seg["n_take"]
            year_legend_done.add(yr)

    # Annotate total and flag multi-sector
    for idx, (row_plan, name) in enumerate(zip(plan_col, planet_names)):
        total = sum(p["n_take"] for p in row_plan)
        multi_flag = "" if len(row_plan) == 1 else " *"
        ax.text(bottoms[idx] + 0.15, n - 1 - idx,
                f"{total}{multi_flag}", va="center", fontsize=8)

    ax.set_yticks(range(n))
    ax.set_yticklabels(planet_names[::-1], fontsize=8.5)
    ax.set_xlabel(f"Number of transits selected (target = {target})", fontsize=10)
    ax.set_title(
        f"Transit Selection Plan — {n} TESS-Best Planets\n"
        f"Target: {target} transits per planet  |  "
        f"* = multi-sector  |  color = TESS observing year",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(0, target * 1.25)
    ax.axvline(target, color="black", lw=1.0, ls="--", alpha=0.4)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR, "transit_selection_plan.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

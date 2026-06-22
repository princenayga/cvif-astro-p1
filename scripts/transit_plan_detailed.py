#!/usr/bin/env python3
"""
transit_plan_detailed.py

Produces a row-per-transit selection plan:
  For planet X, use Sector Y, Transit N (tc = BTJD, date = UTC approx)

Uses only MAST metadata (no LC download) + already-computed sector_report.csv.

Output:
  results/transit_plan_detailed.csv
  results/transit_plan_detailed.png

Usage:
    python transit_plan_detailed.py
    python transit_plan_detailed.py --target 10 --min-usable 10
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

RESULTS_DIR  = "results"
BEST_CSV     = "planets_tess_best.csv"
PRIORITY_CSV = "planets_priority.csv"
BJD_OFFSET   = 2457000.0
MJD_TO_BTJD  = 2400000.5 - BJD_OFFSET
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


def get_host(row):
    import re
    host = str(row.get("host_star", row.get("System", ""))).strip()
    return re.sub(r"\s+[A-D]$", "", host).strip()


def get_sector_time_ranges(host):
    """MAST metadata only — no download."""
    results = lk.search_lightcurve(host, mission="TESS", exptime=120, author="SPOC")
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


def predict_usable_in_sector(period, t0_sec, trandur, tmin, tmax, min_pts=10):
    """
    List all predicted transit midpoints inside [tmin, tmax] in time order.
    Returns list of tc values (BTJD) — these are the usable transit midpoints.
    """
    half_dur = trandur / 24 / 2
    n_lo = int(np.ceil((tmin - t0_sec) / period))
    n_hi = int(np.floor((tmax - t0_sec) / period))
    times = [t0_sec + n * period for n in range(n_lo, n_hi + 1)]
    # Keep only those whose midpoint is within the sector window
    return [tc for tc in times if tmin + half_dur <= tc <= tmax - half_dur]


def btjd_to_utc(btjd):
    """Convert BTJD to approximate UTC date string."""
    try:
        bjd = btjd + BJD_OFFSET
        t   = Time(bjd, format="jd", scale="tdb")
        return t.iso[:10]
    except Exception:
        return "?"


def load_sector_report(planet):
    path = os.path.join(RESULTS_DIR, planet, "sector_report.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    rep = pd.read_csv(path)
    rep = rep[rep["sector"] != "Total"].copy()
    rep["sector"] = rep["sector"].astype(int)
    return rep


def select_sectors(rep, target):
    """Same logic as select_transits.py — returns list of {sector, n_take}."""
    usable = rep[rep["n_usable"] > 0].sort_values("n_usable", ascending=False)
    if usable.empty:
        return []

    best = usable.iloc[0]
    if best["n_usable"] >= target:
        return [{"sector": int(best["sector"]), "n_take": target}]

    primary_year = sector_year(int(best["sector"]))
    same  = usable[usable["sector"].apply(sector_year) == primary_year]
    other = usable[usable["sector"].apply(sector_year) != primary_year]
    candidates = pd.concat([same, other])

    remaining = target
    plan = []
    for _, row in candidates.iterrows():
        if remaining <= 0:
            break
        take = min(int(row["n_usable"]), remaining)
        plan.append({"sector": int(row["sector"]), "n_take": take})
        remaining -= take
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",     type=int, default=TARGET)
    parser.add_argument("--min-usable", type=int, default=MIN_USABLE)
    args   = parser.parse_args()
    target     = args.target
    min_usable = args.min_usable

    df_best = pd.read_csv(BEST_CSV)
    df_pri  = pd.read_csv(PRIORITY_CSV)
    df_pri["t0_btjd"] = df_pri["pl_tranmid"] - BJD_OFFSET

    detail_rows = []
    skipped     = []
    total       = len(df_best)

    for i, (_, brow) in enumerate(df_best.iterrows()):
        planet = brow["System"]
        print(f"[{i+1}/{total}] {planet}...", end=" ", flush=True)

        prow   = df_pri[df_pri["System"] == planet].iloc[0]
        host   = get_host(prow)
        period = float(prow["Period"])
        trandur= float(prow["pl_trandur"])

        rep    = load_sector_report(planet)
        total_usable = int(rep["n_usable"].sum()) if not rep.empty else 0

        if total_usable < min_usable:
            skipped.append((planet, total_usable))
            print(f"skipped ({total_usable} usable < {min_usable})")
            continue

        sector_plan = select_sectors(rep, target)

        # Get sector time ranges from MAST (metadata only)
        try:
            sec_ranges = get_sector_time_ranges(host)
        except Exception as e:
            print(f"MAST query failed: {e}")
            skipped.append((planet, total_usable))
            continue

        # Build per-transit rows.
        # sector_report already verified n_usable transits have complete windows,
        # so no edge filtering needed — trust the count.
        # If a segment delivers fewer transits than n_take (ephemeris vs data gap),
        # pull the deficit from the next-richest unused sector.

        # Build a pool of all sectors sorted by n_usable descending
        all_usable_secs = rep[rep["n_usable"] > 0].sort_values(
            "n_usable", ascending=False)
        used_secs = {seg["sector"] for seg in sector_plan}

        transit_counter = 0
        remaining_debt  = 0   # shortfall carried from a previous segment

        for seg in sector_plan:
            sec    = seg["sector"]
            n_take = seg["n_take"] + remaining_debt
            remaining_debt = 0

            rep_row = rep[rep["sector"] == sec]
            if rep_row.empty or sec not in sec_ranges:
                remaining_debt = n_take
                continue

            t0_sec  = float(rep_row["t0_sec_btjd"].values[0])
            med_pts = float(rep_row["median_pts"].values[0])
            tmin, tmax = sec_ranges[sec]

            transit_times = predict_usable_in_sector(
                period, t0_sec, trandur, tmin, tmax
            )
            selected = transit_times[:n_take]

            # If we got fewer than planned, record the shortfall
            if len(selected) < n_take:
                remaining_debt = n_take - len(selected)

            for t_idx, tc in enumerate(selected):
                transit_counter += 1
                detail_rows.append({
                    "planet":       planet,
                    "sector":       sec,
                    "transit_n":    t_idx + 1,
                    "export_n":     t_idx + 1,
                    "overall_n":    transit_counter,
                    "tc_btjd":      round(tc, 6),
                    "tc_utc":       btjd_to_utc(tc),
                    "tess_year":    sector_year(sec),
                    "median_pts":   med_pts,
                    "single_sector": len(sector_plan) == 1,
                })

        # Supplement from unused sectors if still short
        if remaining_debt > 0:
            fallback_secs = all_usable_secs[
                ~all_usable_secs["sector"].isin(used_secs)
            ]
            for _, fb_row in fallback_secs.iterrows():
                if remaining_debt <= 0:
                    break
                fb_sec = int(fb_row["sector"])
                if fb_sec not in sec_ranges:
                    continue
                t0_fb   = float(fb_row["t0_sec_btjd"])
                med_fb  = float(fb_row["median_pts"])
                tmin_fb, tmax_fb = sec_ranges[fb_sec]
                fb_times = predict_usable_in_sector(
                    period, t0_fb, trandur, tmin_fb, tmax_fb
                )
                fb_selected = fb_times[:remaining_debt]
                for t_idx, tc in enumerate(fb_selected):
                    transit_counter += 1
                    detail_rows.append({
                        "planet":        planet,
                        "sector":        fb_sec,
                        "transit_n":     t_idx + 1,
                        "export_n":      t_idx + 1,
                        "overall_n":     transit_counter,
                        "tc_btjd":       round(tc, 6),
                        "tc_utc":        btjd_to_utc(tc),
                        "tess_year":     sector_year(fb_sec),
                        "median_pts":    med_fb,
                        "single_sector": False,
                    })
                remaining_debt -= len(fb_selected)
                used_secs.add(fb_sec)

        n_got = sum(1 for r in detail_rows if r["planet"] == planet)
        print(f"{n_got} transits from sectors used for {planet}")

    df_detail = pd.DataFrame(detail_rows)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS_DIR, "transit_plan_detailed.csv")
    df_detail.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── Console table ─────────────────────────────────────────────────────────
    print(f"\n{'Planet':<22} {'Sec':>4} {'#':>3} {'tc (BTJD)':>12} {'UTC Date':>12} {'Pts':>5}")
    print("-" * 65)
    prev_planet = None
    for _, row in df_detail.iterrows():
        sep = "" if row["planet"] == prev_planet else "\n" if prev_planet else ""
        print(f"{sep}{row['planet']:<22} {row['sector']:>4}  {row['transit_n']:>2}  "
              f"{row['tc_btjd']:>12.5f}  {row['tc_utc']:>12}  {row['median_pts']:>5.0f}")
        prev_planet = row["planet"]

    print("-" * 65)
    print(f"Total: {len(df_detail)} transits across "
          f"{df_detail['planet'].nunique()} planets")
    if skipped:
        print(f"\nSkipped ({len(skipped)}): "
              + ", ".join(f"{p}({n})" for p, n in skipped))

    # ── Plot: timeline of selected transits ───────────────────────────────────
    planets_ordered = df_detail["planet"].unique().tolist()
    n_planets = len(planets_ordered)

    year_colors = {1:"#1f77b4",2:"#ff7f0e",3:"#2ca02c",
                   4:"#d62728",5:"#9467bd",6:"#8c564b",7:"#e377c2"}

    fig, ax = plt.subplots(figsize=(14, max(6, n_planets * 0.42)))

    year_labeled = set()
    for pi, planet in enumerate(planets_ordered):
        pdata = df_detail[df_detail["planet"] == planet]
        for _, row in pdata.iterrows():
            yr    = row["tess_year"]
            color = year_colors[yr]
            label = f"TESS Year {yr}" if yr not in year_labeled else "_nolegend_"
            ax.scatter(row["tc_btjd"], pi, color=color, s=55,
                       marker="|", linewidths=2.5, label=label, zorder=3)
            year_labeled.add(yr)

    ax.set_yticks(range(n_planets))
    ax.set_yticklabels(planets_ordered, fontsize=8.5)
    ax.set_xlabel("Transit midpoint (BTJD)", fontsize=10)
    ax.set_title(
        f"Selected Transits — {n_planets} TESS-Best Planets  "
        f"({len(df_detail)} total transits)\n"
        "Each tick = one selected transit | color = TESS observing year",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR, "transit_plan_detailed.png")
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

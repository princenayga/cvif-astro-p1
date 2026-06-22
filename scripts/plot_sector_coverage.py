#!/usr/bin/env python3
"""
plot_sector_coverage.py — Visualize TESS sector coverage for priority planets

Queries MAST via lightkurve for each planet, retrieves actual TESS sector
numbers from sequence_number column, and plots a grid showing which sectors
each planet was observed in.

Usage:
    python plot_sector_coverage.py               # all ok planets in transit_counts.csv
    python plot_sector_coverage.py --top 10      # top 10 by usable transits only
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import re
import time

COUNTS_PATH = "transit_counts.csv"
SLEEP       = 1.5


def get_host(row):
    host = str(row.get("host_star", row.get("System", ""))).strip()
    host = re.sub(r'\s+[A-D]$', '', host).strip()
    return host


def query_sectors(host_star):
    results = lk.search_lightcurve(host_star, mission="TESS", exptime=120)
    if results is None or len(results) == 0:
        return []
    table = results.table
    sectors = sorted(set(int(s) for s in table["sequence_number"] if s is not None))
    return sectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=None,
                        help="Only show top N planets by usable transits")
    args = parser.parse_args()

    df = pd.read_csv(COUNTS_PATH)
    df = df[df["status"] == "ok"].sort_values("n_usable", ascending=False)
    if args.top:
        df = df.head(args.top)

    planets = df["System"].tolist()
    print(f"Querying sector coverage for {len(planets)} planets...\n")

    planet_sectors = {}
    for i, (_, row) in enumerate(df.iterrows()):
        name = row["System"]
        host = get_host(row)
        print(f"  [{i+1}/{len(planets)}] {name} ({host})...", end=" ", flush=True)
        try:
            sectors = query_sectors(host)
            planet_sectors[name] = sectors
            print(f"sectors: {sectors}")
        except Exception as e:
            planet_sectors[name] = []
            print(f"ERROR: {e}")
        time.sleep(SLEEP)

    # ── Build the plot ──────────────────────────────────────────────────────────
    all_sectors = sorted(set(s for ss in planet_sectors.values() for s in ss))
    if not all_sectors:
        print("No sector data retrieved.")
        return

    max_sector = max(all_sectors)
    sector_range = list(range(1, max_sector + 1))

    n_planets = len(planets)
    n_sectors = len(sector_range)

    fig, ax = plt.subplots(figsize=(max(14, n_sectors * 0.18), max(5, n_planets * 0.45)))

    # Color by usable transit count
    usable_map = dict(zip(df["System"], df["n_usable"]))
    max_usable = df["n_usable"].max()

    cmap = plt.cm.YlOrRd

    for y, name in enumerate(reversed(planets)):
        secs = set(planet_sectors.get(name, []))
        u = usable_map.get(name, 0)
        color = cmap(0.2 + 0.7 * (u / max_usable)) if max_usable > 0 else cmap(0.5)

        for x, s in enumerate(sector_range):
            if s in secs:
                ax.add_patch(mpatches.Rectangle(
                    (x, y + 0.1), 0.85, 0.8,
                    facecolor=color, edgecolor="white", linewidth=0.4
                ))

    # Axis labels
    ax.set_xlim(0, n_sectors)
    ax.set_ylim(0, n_planets)
    ax.set_yticks(np.arange(n_planets) + 0.5)
    ax.set_yticklabels(
        [f"{name}  ({int(usable_map.get(name,0))} transits)"
         for name in reversed(planets)],
        fontsize=8
    )

    # X-axis: show every 5th sector label
    tick_positions = [i for i, s in enumerate(sector_range) if s % 5 == 0 or s == 1]
    tick_labels    = [str(sector_range[i]) for i in tick_positions]
    ax.set_xticks([p + 0.4 for p in tick_positions])
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=0)

    ax.set_xlabel("TESS Sector Number", fontsize=10)
    ax.set_title("TESS PDC-SAP Sector Coverage — Priority Exoplanets\n"
                 "(color intensity = usable transit count)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.15, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=max_usable))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cbar.set_label("Usable transits", fontsize=9)

    plt.tight_layout()
    out = "sector_coverage.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.show()

    # Also print a text table
    print("\nSector lists:")
    for name in planets:
        secs = planet_sectors.get(name, [])
        print(f"  {name:<20}: {secs}")


if __name__ == "__main__":
    main()

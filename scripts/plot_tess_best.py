#!/usr/bin/env python3
"""
plot_tess_best.py — Visualize usable transit counts for the 33 TESS-best planets.

Reads from results/tess_best_transit_counts.csv and per-planet sector_report.csv.

Produces:
  results/tess_best_total.png        — bar chart: total usable transits per planet
  results/tess_best_by_sector.png    — stacked bar: sector breakdown per planet
  results/tess_best_sector_grid.png  — grid: which sectors each planet appears in

Usage:
    python plot_tess_best.py
"""

import ast
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR   = "results"
COUNTS_CSV    = os.path.join(RESULTS_DIR, "tess_best_transit_counts.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(COUNTS_CSV)
df = df[df["total_usable"] > 0].sort_values("total_usable", ascending=False).reset_index(drop=True)

# Build planet x sector matrix from per-planet sector_report.csv files
all_sectors = set()
planet_sector_map = {}   # {planet: {sector: n_usable}}

for name in df["planet"]:
    path = os.path.join(RESULTS_DIR, name, "sector_report.csv")
    if not os.path.exists(path):
        planet_sector_map[name] = {}
        continue
    rep = pd.read_csv(path)
    rep = rep[rep["sector"] != "Total"].copy()
    rep["sector"] = rep["sector"].astype(int)
    sec_map = {int(r["sector"]): int(r["n_usable"])
               for _, r in rep.iterrows() if int(r["n_usable"]) > 0}
    planet_sector_map[name] = sec_map
    all_sectors.update(sec_map.keys())

all_sectors = sorted(all_sectors)
planets     = df["planet"].tolist()
totals      = df["total_usable"].values

# ── Plot 1: Total usable transits per planet ──────────────────────────────────
fig, ax = plt.subplots(figsize=(11, max(6, len(planets) * 0.34)))

cmap   = plt.cm.RdYlGn
colors = cmap(np.linspace(0.2, 0.85, len(planets))[::-1])

bars = ax.barh(planets[::-1], totals[::-1], color=colors,
               edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, totals[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va="center", fontsize=8)

med = np.median(totals)
ax.axvline(med, color="navy", lw=1.3, ls="--", label=f"Median = {med:.0f}")
ax.set_xlabel("Total usable transits (all sectors combined)", fontsize=10)
ax.set_title("TESS-Best Planets — Total Usable Transits\n"
             "(pre + in-transit + post windows each >= 10 pts)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, totals.max() * 1.14)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out1 = os.path.join(RESULTS_DIR, "tess_best_total.png")
plt.savefig(out1, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── Plot 2: Stacked bar — sector breakdown per planet ─────────────────────────
# Group sectors by TESS year to keep the legend manageable
# Year 1: S01-13, Year 2: S14-26, Year 3: S27-39, Year 4: S40-55,
# Year 5: S56-69, Year 6: S70-83, Year 7+: S84+
def sector_year(s):
    if s <= 13:  return 1
    if s <= 26:  return 2
    if s <= 39:  return 3
    if s <= 55:  return 4
    if s <= 69:  return 5
    if s <= 83:  return 6
    return 7

year_colors = {
    1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c",
    4: "#d62728", 5: "#9467bd", 6: "#8c564b", 7: "#e377c2",
}
year_labels = {
    1: "Y1 S01-13", 2: "Y2 S14-26", 3: "Y3 S27-39",
    4: "Y4 S40-55", 5: "Y5 S56-69", 6: "Y6 S70-83", 7: "Y7+ S84+",
}

fig, ax = plt.subplots(figsize=(13, max(6, len(planets) * 0.34)))

bottoms = np.zeros(len(planets))
year_plotted = set()

for sec in all_sectors:
    yr    = sector_year(sec)
    color = year_colors[yr]
    vals  = np.array([planet_sector_map[p].get(sec, 0) for p in planets], dtype=float)

    label = year_labels[yr] if yr not in year_plotted else "_nolegend_"
    ax.barh(range(len(planets)), vals[::-1],
            left=bottoms[::-1],
            color=color, edgecolor="white", linewidth=0.2,
            alpha=0.85, label=label)
    year_plotted.add(yr)
    bottoms += vals

# Annotate total
for i, total in enumerate(totals[::-1]):
    if total > 0:
        ax.text(total + 1, i, str(int(total)), va="center", fontsize=7.5)

ax.set_yticks(range(len(planets)))
ax.set_yticklabels(planets[::-1], fontsize=8.5)
ax.set_xlabel("Usable transits", fontsize=10)
ax.set_title("TESS-Best Planets — Usable Transits by TESS Year/Sector\n"
             "(each color = one TESS observing year)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", title="TESS Year")
ax.set_xlim(0, totals.max() * 1.14)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
out2 = os.path.join(RESULTS_DIR, "tess_best_by_sector.png")
plt.savefig(out2, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── Plot 3: Grid — which sectors each planet appears in ───────────────────────
# Matrix: rows = planets, columns = sectors; cell = usable count (0 = grey)
mat = np.zeros((len(planets), len(all_sectors)), dtype=int)
for r, planet in enumerate(planets):
    for c, sec in enumerate(all_sectors):
        mat[r, c] = planet_sector_map[planet].get(sec, 0)

fig, ax = plt.subplots(figsize=(max(16, len(all_sectors) * 0.21),
                                 max(6,  len(planets) * 0.38)))

cmap_grid = plt.cm.YlOrRd.copy()
cmap_grid.set_under("#eeeeee")
vmax = mat.max() if mat.max() > 0 else 1

im = ax.imshow(mat, aspect="auto", cmap=cmap_grid,
               vmin=0.5, vmax=vmax, interpolation="nearest")

# Annotate non-zero cells
for r in range(mat.shape[0]):
    for c in range(mat.shape[1]):
        v = mat[r, c]
        if v > 0:
            ax.text(c, r, str(v), ha="center", va="center",
                    fontsize=5.5,
                    color="white" if v > vmax * 0.6 else "black")

# X-axis: sector labels, colored by year
ax.set_xticks(range(len(all_sectors)))
xlabels = [f"S{s:02d}" for s in all_sectors]
ax.set_xticklabels(xlabels, rotation=90, fontsize=6.5)

# Color x-tick labels by TESS year
for tick, sec in zip(ax.get_xticklabels(), all_sectors):
    tick.set_color(year_colors[sector_year(sec)])

ax.set_yticks(range(len(planets)))
ax.set_yticklabels(planets, fontsize=8.5)
ax.set_xlabel("TESS Sector  (color = TESS year: blue=Y1 orange=Y2 green=Y3 red=Y4 purple=Y5 brown=Y6)",
              fontsize=8)
ax.set_title("TESS-Best Planets — Sector Coverage Map\n"
             "(cell value = usable transits; grey = not observed or no usable transits)",
             fontsize=11, fontweight="bold")

cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
cb.set_label("Usable transits", fontsize=9)

plt.tight_layout()
out3 = os.path.join(RESULTS_DIR, "tess_best_sector_grid.png")
plt.savefig(out3, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\n{len(planets)} planets  |  {int(totals.sum())} total usable transits")
print(f"Median per planet: {np.median(totals):.0f}  |  Max: {totals.max()} ({planets[0]})")
print(f"Sectors covered: {len(all_sectors)}  (S{min(all_sectors):02d} - S{max(all_sectors):02d})")

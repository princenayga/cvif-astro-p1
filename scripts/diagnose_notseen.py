#!/usr/bin/env python3
"""
diagnose_notseen.py

Plots full sector light curves for planets flagged as transitNotSeen,
with predicted transit midpoints overlaid.
Saves one PNG per planet-sector to TESS_focused/diagnostics/

Usage:
    python diagnose_notseen.py
"""

import os, time, warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

BJD_OFFSET = 2457000.0
OUT_DIR    = os.path.join("TESS_focused", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

NOT_SEEN = {
    "HAT-P-20":  [44, 45],
    "HATS-06":   [98],
    "TOI-157":   [98],
    "TOI-169":   [13],
    "WASP-064":  [33],
    "WASP-170":  [89],
}

plan = pd.read_csv("results/transit_plan_detailed.csv")
pri  = pd.read_csv("planets_priority.csv")

for planet, sectors in NOT_SEEN.items():
    prow = pri[pri["System"] == planet]
    if prow.empty:
        print(f"{planet}: not found in priority CSV"); continue
    prow   = prow.iloc[0]
    host   = re.sub(r"\s+[A-D]$", "", str(prow.get("host_star", planet))).strip()
    period = float(prow["Period"])
    T_days = float(prow["pl_trandur"]) / 24.0
    depth  = float(prow["pl_trandep"]) / 1e6 if pd.notna(prow.get("pl_trandep")) else None

    for sector in sectors:
        print(f"[{planet}] S{sector} downloading...", end=" ", flush=True)
        try:
            sr = lk.search_lightcurve(host, mission="TESS", sector=sector,
                                      author="SPOC", exptime=120)
            if not sr or len(sr) == 0:
                print("not found"); continue
            lc = sr[0].download(flux_column="pdcsap_flux").remove_nans()
        except Exception as e:
            print(f"failed: {e}"); continue

        t = np.array(lc.time.value)
        f = np.array(lc.flux.value if hasattr(lc.flux, "value") else lc.flux)
        div = np.nanpercentile(f, 75)
        f   = f / div
        print(f"{len(t)} pts  flux range {f.min():.5f}–{f.max():.5f}")

        # Predicted transit times
        rep_path = os.path.join("results", planet, "sector_report.csv")
        t0_sec = None
        if os.path.exists(rep_path):
            rep = pd.read_csv(rep_path)
            rep = rep[rep["sector"] != "Total"]
            rep["sector"] = rep["sector"].astype(int)
            rrow = rep[rep["sector"] == sector]
            if not rrow.empty:
                t0_sec = float(rrow["t0_sec_btjd"].values[0])
        if t0_sec is None:
            p_plan = plan[(plan["planet"]==planet) & (plan["sector"]==sector)]
            t0_sec = float(p_plan["tc_btjd"].iloc[0]) if len(p_plan) else t.min()

        n_lo = int(np.ceil((t.min() - t0_sec) / period))
        n_hi = int(np.floor((t.max() - t0_sec) / period))
        tcs  = [t0_sec + n * period for n in range(n_lo, n_hi + 1)
                if t.min() <= t0_sec + n * period <= t.max()]

        fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=False)

        # Top: full sector
        ax = axes[0]
        ax.scatter(t, f, s=1, color="#444444", zorder=2)
        for tc in tcs:
            ax.axvline(tc, color="red", lw=0.7, alpha=0.5)
        ax.set_ylabel("Norm. flux"); ax.set_xlabel("BTJD")
        noise = np.std(f) * 1e6
        title = f"{planet} S{sector} — full sector  (σ={noise:.0f} ppm"
        if depth:
            title += f",  expected depth={depth*1e6:.0f} ppm"
        title += ")"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(t.min(), t.max())

        # Bottom: phase-folded on predicted period
        phase = ((t - t0_sec) % period)
        phase[phase > period/2] -= period
        ax2 = axes[1]
        ax2.scatter(phase * 24, f, s=1, color="#444444", zorder=2, alpha=0.5)
        ax2.axvline(-T_days/2*24, color="navy", lw=1, ls="--", alpha=0.7)
        ax2.axvline( T_days/2*24, color="navy", lw=1, ls="--", alpha=0.7)
        ax2.axvline(0, color="red", lw=1, ls="-.", alpha=0.5)
        ax2.set_xlim(-period/2*24, period/2*24)
        ax2.set_xlabel("Phase (hours)"); ax2.set_ylabel("Norm. flux")
        ax2.set_title("Phase-folded  (navy dashes = transit window)", fontsize=9)

        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"{planet}_S{sector:03d}_diagnostic.png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")

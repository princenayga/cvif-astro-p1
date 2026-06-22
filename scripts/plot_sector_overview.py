#!/usr/bin/env python3
"""
plot_sector_overview.py

For a given planet, downloads the full sector LC and plots it with all
transit windows shaded:
  Blue   = pre-baseline  [tc - 1.5T, tc - 0.5T]
  Pink   = in-transit    [tc - 0.5T, tc + 0.5T]
  Green  = post-baseline [tc + 0.5T, tc + 1.5T]

Also applies any manual offsets from tc_manual_offsets.csv before plotting.
Saves to results/sector_lcs/{planet}_S{sec}_overview.png

After inspecting, either:
  - Run again with updated tc_manual_offsets.csv to confirm alignment
  - Run export_transit_csv.py --planet X to extract CSVs

Usage:
    python scripts/plot_sector_overview.py --planet TOI-157
    python scripts/plot_sector_overview.py --planet KELT-20 --sector 40
    python scripts/plot_sector_overview.py --planet KELT-20 --all-sectors
"""

import argparse
import os
import re
import time as time_mod
import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

BJD_OFFSET = 2457000.0
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RESULTS_DIR= os.path.join(BASE_DIR, "results")
OUT_DIR    = os.path.join(RESULTS_DIR, "sector_lcs")

LD_TABLE = [
    (7000, 0.20, 0.20), (6000, 0.33, 0.25), (5000, 0.44, 0.25),
    (4000, 0.55, 0.22), (0,    0.65, 0.15),
]


def robust_normalize(lc):
    f = np.array(lc.flux)
    if hasattr(f.flat[0], "value"):
        f = np.array([v.value for v in f])
    div = np.nanpercentile(f, 75)
    return lk.LightCurve(time=lc.time, flux=f / (div if div else 1.0))


def download_lc(host, sector, retries=4):
    delays = [10, 20, 30]
    for attempt in range(retries):
        try:
            sr = lk.search_lightcurve(host, mission="TESS", sector=sector,
                                      author="SPOC", exptime=120)
            if not sr or len(sr) == 0:
                return None
            lc = sr[0].download(flux_column="pdcsap_flux")
            if lc is None:
                return None
            return robust_normalize(lc.remove_nans())
        except Exception as e:
            if attempt < retries - 1:
                time_mod.sleep(delays[min(attempt, 2)])
            else:
                print(f"    Download failed: {e}")
                return None


def get_ld(teff):
    for thr, u1, u2 in LD_TABLE:
        if teff >= thr:
            return u1, u2
    return 0.44, 0.25


def refine_t0_stacking(time, flux, t0_init, period, half_dur, n_bins=40):
    phase = (time - t0_init) % period
    phase[phase > period / 2] -= period
    mask = np.abs(phase) <= 2 * half_dur
    if mask.sum() < 20:
        return t0_init
    p_in, f_in = phase[mask], flux[mask]
    bins = np.linspace(-2*half_dur, 2*half_dur, n_bins+1)
    bct  = 0.5*(bins[:-1]+bins[1:])
    bt, bf = [], []
    for i in range(n_bins):
        sel = (p_in >= bins[i]) & (p_in < bins[i+1])
        if sel.sum() > 0:
            bt.append(bct[i]); bf.append(np.median(f_in[sel]))
    if len(bt) < 5:
        return t0_init
    bt, bf = np.array(bt), np.array(bf)
    inner = np.abs(bt) <= half_dur
    if inner.sum() < 3:
        inner = np.ones(len(bt), dtype=bool)
    try:
        a, b, _ = np.polyfit(bt[inner], bf[inner], 2)
        if a <= 0:
            return t0_init
        offset = -b / (2*a)
        return t0_init + offset if abs(offset) <= half_dur else t0_init
    except Exception:
        return t0_init


def load_manual_offsets(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "tc_manual_offsets.csv")
    offsets = {}
    if not os.path.exists(path):
        return offsets
    df = pd.read_csv(path, comment="#")
    for _, row in df.iterrows():
        key = (str(row["planet"]).strip(), str(row["sector"]), str(row["transit_n"]))
        offsets[key] = float(row["offset_min"]) / 60 / 24
    return offsets


def get_manual_offset(offsets, planet, sector, transit_n):
    for s in (str(sector), "all"):
        for t in (str(transit_n), "all"):
            v = offsets.get((planet, s, t))
            if v is not None:
                return v
    return 0.0


def plot_overview(planet, sector, time_arr, flux_arr, tcs, T_days, out_path,
                  depth_ppm=None, offsets_applied=False):
    """
    Full-sector LC with pre/transit/post windows shaded for every transit.
    """
    half_T = T_days / 2.0
    noise_ppm = np.std(flux_arr) * 1e6

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.scatter(time_arr, flux_arr, s=1, color="#333333", zorder=2, rasterized=True)

    legend_done = set()
    for i, tc in enumerate(tcs):
        # Pre-baseline
        ax.axvspan(tc - 1.5*T_days, tc - half_T,
                   color="#cce5ff", alpha=0.45, zorder=1,
                   label="Pre-baseline" if "pre" not in legend_done else "_")
        # Transit
        ax.axvspan(tc - half_T, tc + half_T,
                   color="#ffcccc", alpha=0.55, zorder=1,
                   label="Transit" if "tr" not in legend_done else "_")
        # Post-baseline
        ax.axvspan(tc + half_T, tc + 1.5*T_days,
                   color="#ccffdd", alpha=0.45, zorder=1,
                   label="Post-baseline" if "post" not in legend_done else "_")
        # tc marker
        ax.axvline(tc, color="red", lw=0.8, alpha=0.7, zorder=3)
        ax.text(tc, flux_arr.max() + 0.0005, str(i+1),
                color="red", fontsize=7, ha="center", va="bottom")
        legend_done.update({"pre", "tr", "post"})

    ax.set_xlim(time_arr.min(), time_arr.max())
    ax.set_xlabel("BTJD", fontsize=11)
    ax.set_ylabel("Normalised flux", fontsize=11)

    title = f"{planet}  |  Sector {sector}  |  Full LC\n"
    title += f"σ = {noise_ppm:.0f} ppm"
    if depth_ppm:
        title += f"   Expected depth = {depth_ppm:.0f} ppm   SNR ~ {depth_ppm/noise_ppm:.1f}"
    title += f"   T = {T_days*24:.2f} h   {len(tcs)} transits marked"
    if offsets_applied:
        title += "   [manual offsets applied]"
    ax.set_title(title, fontsize=10, fontweight="bold")

    patches = [
        mpatches.Patch(color="#cce5ff", alpha=0.7, label="Pre-baseline"),
        mpatches.Patch(color="#ffcccc", alpha=0.7, label="Transit"),
        mpatches.Patch(color="#ccffdd", alpha=0.7, label="Post-baseline"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet",      required=True, type=str)
    parser.add_argument("--sector",      type=int, default=None,
                        help="Specific sector (default: best sector from plan)")
    parser.add_argument("--all-sectors", action="store_true",
                        help="Plot all sectors for this planet")
    args = parser.parse_args()

    plan = pd.read_csv(os.path.join(RESULTS_DIR, "summary", "transit_plan_detailed.csv"))
    pri  = pd.read_csv(os.path.join(DATA_DIR, "planets_priority.csv"))
    os.makedirs(OUT_DIR, exist_ok=True)

    p_plan = plan[plan["planet"] == args.planet]
    if p_plan.empty:
        print(f"Planet '{args.planet}' not found in transit plan."); return

    prow = pri[pri["System"] == args.planet]
    if prow.empty:
        print(f"'{args.planet}' not in planets_priority.csv"); return
    prow = prow.iloc[0]

    host   = re.sub(r"\s+[A-D]$", "", str(prow.get("host_star", args.planet))).strip()
    period = float(prow["Period"])
    T_days = float(prow["pl_trandur"]) / 24.0
    half_T = T_days / 2.0
    depth  = float(prow["pl_trandep"]) if pd.notna(prow.get("pl_trandep")) else None

    manual_offsets = load_manual_offsets()
    has_offsets = any(k[0] == args.planet for k in manual_offsets)

    # Which sectors to plot
    if args.all_sectors:
        sectors = p_plan["sector"].unique()
    elif args.sector:
        sectors = [args.sector]
    else:
        sectors = p_plan["sector"].unique()   # all sectors in plan

    for sector in sectors:
        s_plan = p_plan[p_plan["sector"] == sector]

        print(f"[{args.planet}] Sector {sector} — downloading...", end=" ", flush=True)
        lc = download_lc(host, sector)
        if lc is None:
            print("FAILED"); continue

        time_arr = np.array(lc.time.value) if hasattr(lc.time, "value") else np.array(lc.time)
        flux_arr = np.array(lc.flux.value) if hasattr(lc.flux, "value") else np.array(lc.flux)
        print(f"{len(time_arr)} pts")

        # Load & stack-refine t0
        rep_path = os.path.join(RESULTS_DIR, "survey", args.planet, "sector_report.csv")
        t0_sec = None
        if os.path.exists(rep_path):
            rep = pd.read_csv(rep_path)
            rep = rep[rep["sector"] != "Total"]
            rep["sector"] = rep["sector"].astype(int)
            rr = rep[rep["sector"] == sector]
            if not rr.empty:
                t0_sec = float(rr["t0_sec_btjd"].values[0])
        if t0_sec is None:
            t0_sec = float(s_plan.iloc[0]["tc_btjd"])

        t0_sec = refine_t0_stacking(time_arr, flux_arr, t0_sec, period, half_T)

        # All predicted tc values within sector data range
        n_lo = int(np.ceil((time_arr.min()  - t0_sec) / period))
        n_hi = int(np.floor((time_arr.max() - t0_sec) / period))
        all_tc = [t0_sec + n * period for n in range(n_lo, n_hi + 1)
                  if time_arr.min() <= t0_sec + n*period <= time_arr.max()]

        # Apply manual offsets per transit
        tcs_plot = []
        for i, tc in enumerate(all_tc):
            off = get_manual_offset(manual_offsets, args.planet, sector, i+1)
            tcs_plot.append(tc + off)

        out_path = os.path.join(OUT_DIR,
                                f"{args.planet}_S{sector:03d}_overview.png")
        plot_overview(args.planet, sector, time_arr, flux_arr, tcs_plot,
                      T_days, out_path, depth_ppm=depth,
                      offsets_applied=has_offsets)

    print(f"\nDone. Check results/sector_lcs/ for overview plots.")
    print("If offsets needed, edit data/tc_manual_offsets.csv then re-run.")
    print(f"When satisfied: python scripts/export_transit_csv.py --planet {args.planet}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
rank_sectors.py

For a given planet, scores and ranks ALL available TESS sectors using:

  1. Transit SNR       = expected_depth / baseline_rms  (higher = better)
  2. Baseline RMS      = out-of-transit scatter in ppm   (lower = better)
  3. Baseline stability= ratio of detrended vs raw std   (closer to 1 = flatter)
  4. Usable transits   = from sector_report.csv          (higher = better)
  5. Data completeness = actual pts / expected pts       (higher = better)

Outputs:
  results/sector_rankings/{planet}_sector_ranking.csv   — scored table
  results/sector_rankings/{planet}_sector_ranking.png   — bar chart

Usage:
    python scripts/rank_sectors.py --planet TOI-157
    python scripts/rank_sectors.py --planet TOI-157 --top 3
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
import numpy as np
import pandas as pd

BJD_OFFSET  = 2457000.0
MJD_TO_BTJD = 2400000.5 - BJD_OFFSET
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR     = os.path.join(RESULTS_DIR, "sector_rankings")


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
                return None


def baseline_rms(time, flux, t0, period, half_T):
    """Out-of-transit RMS in ppm after linear detrend of each inter-transit gap."""
    phase = (time - t0) % period
    phase[phase > period/2] -= period
    out_mask = np.abs(phase) > half_T * 1.2
    if out_mask.sum() < 20:
        return np.nan, np.nan
    t_out = time[out_mask]
    f_out = flux[out_mask]
    # Detrend with linear fit to remove slow drift
    try:
        coeffs = np.polyfit(t_out, f_out, 1)
        f_detrended = f_out - np.polyval(coeffs, t_out)
    except Exception:
        f_detrended = f_out - np.median(f_out)
    raw_rms      = float(np.std(f_out) * 1e6)
    detrended_rms= float(np.std(f_detrended) * 1e6)
    stability    = detrended_rms / raw_rms if raw_rms > 0 else 1.0
    return detrended_rms, stability


def completeness(time, sector_duration_days):
    """Fraction of expected 2-min cadence points actually present."""
    expected = sector_duration_days * 24 * 60 / 2
    return min(len(time) / expected, 1.0)


def score_sector(rms, stability, snr, n_usable, complete):
    """
    Composite score (higher = better).
    Weights: SNR 40%, RMS 30%, stability 15%, usable 10%, completeness 5%
    Each component normalised to [0,1] before weighting — done across sectors
    so this function returns the raw components; normalisation happens in main.
    """
    return dict(rms=rms, stability=stability, snr=snr,
                n_usable=n_usable, completeness=complete)


def sector_year(s):
    if s <= 13: return 1
    if s <= 26: return 2
    if s <= 39: return 3
    if s <= 55: return 4
    if s <= 69: return 5
    if s <= 83: return 6
    return 7


YEAR_COLORS = {1:"#1f77b4",2:"#ff7f0e",3:"#2ca02c",
               4:"#d62728",5:"#9467bd",6:"#8c564b",7:"#e377c2"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet", required=True)
    parser.add_argument("--top", type=int, default=None,
                        help="Print top N sectors (default: all)")
    args = parser.parse_args()

    pri  = pd.read_csv(os.path.join(DATA_DIR, "planets_priority.csv"))
    prow = pri[pri["System"] == args.planet]
    if prow.empty:
        print(f"'{args.planet}' not found."); return
    prow = prow.iloc[0]

    host   = re.sub(r"\s+[A-D]$", "", str(prow.get("host_star", args.planet))).strip()
    period = float(prow["Period"])
    T_days = float(prow["pl_trandur"]) / 24.0
    half_T = T_days / 2.0
    # pl_trandep is in percent; convert to ppm. Fall back to (Rp/Rs)^2 * 1e6.
    if pd.notna(prow.get("pl_trandep")) and float(prow["pl_trandep"]) > 0:
        depth = float(prow["pl_trandep"]) * 10000   # % -> ppm
    elif pd.notna(prow.get("pl_ratror")) and float(prow["pl_ratror"]) > 0:
        depth = float(prow["pl_ratror"]) ** 2 * 1e6
    else:
        depth = None

    # Load sector_report for usable transit counts
    rep_path = os.path.join(RESULTS_DIR, "survey", args.planet, "sector_report.csv")
    usable_map = {}
    t0_map = {}
    if os.path.exists(rep_path):
        rep = pd.read_csv(rep_path)
        rep = rep[rep["sector"] != "Total"]
        rep["sector"] = rep["sector"].astype(int)
        rep = rep[rep["n_usable"] > 0]
        for _, r in rep.iterrows():
            usable_map[int(r["sector"])] = int(r["n_usable"])
            t0_map[int(r["sector"])]     = float(r["t0_sec_btjd"])

    if not usable_map:
        print(f"No sector_report.csv found for {args.planet}. Run survey_planets.py first.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    sectors = sorted(usable_map.keys())
    print(f"\n{args.planet} — ranking {len(sectors)} sectors with usable transits\n")

    for sec in sectors:
        n_usable = usable_map[sec]
        t0_sec   = t0_map[sec]
        print(f"  S{sec:02d}  downloading...", end=" ", flush=True)
        lc = download_lc(host, sec)
        if lc is None:
            print("FAILED"); continue

        time_arr = np.array(lc.time.value) if hasattr(lc.time, "value") else np.array(lc.time)
        flux_arr = np.array(lc.flux.value) if hasattr(lc.flux, "value") else np.array(lc.flux)

        rms, stab = baseline_rms(time_arr, flux_arr, t0_sec, period, half_T)
        snr       = (depth / rms) if (depth and not np.isnan(rms) and rms > 0) else np.nan
        dur       = time_arr.max() - time_arr.min()
        comp      = completeness(time_arr, dur)

        rows.append({
            "sector":      sec,
            "year":        sector_year(sec),
            "n_usable":    n_usable,
            "rms_ppm":     round(rms, 1)  if not np.isnan(rms)  else None,
            "stability":   round(stab, 3) if not np.isnan(stab) else None,
            "snr":         round(snr, 2)  if not np.isnan(snr)  else None,
            "completeness":round(comp, 3),
            "duration_d":  round(dur, 1),
        })
        print(f"rms={rms:.0f} ppm  snr={snr:.1f}  usable={n_usable}  complete={comp:.0%}")

    if not rows:
        print("No sectors scored."); return

    df = pd.DataFrame(rows)

    # Composite score — normalise each metric to [0,1] then weight
    def norm(series, invert=False):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([0.5]*len(series), index=series.index)
        n = (series - mn) / (mx - mn)
        return 1 - n if invert else n

    df["score"] = (
        norm(df["snr"].fillna(0))          * 0.40 +
        norm(df["rms_ppm"].fillna(df["rms_ppm"].max()), invert=True) * 0.30 +
        norm(df["stability"].fillna(1),    invert=True) * 0.15 +
        norm(df["n_usable"])               * 0.10 +
        norm(df["completeness"])           * 0.05
    )
    df["score"] = df["score"].round(3)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, f"{args.planet}_sector_ranking.csv")
    df.to_csv(csv_path, index=False)

    # Print table
    top_n = args.top or len(df)
    print(f"\n{'Rank':<5} {'Sector':<8} {'Year':<5} {'RMS(ppm)':<10} {'SNR':<7} "
          f"{'Stability':<11} {'Usable':<8} {'Complete':<10} {'Score'}")
    print("-" * 75)
    for _, r in df.head(top_n).iterrows():
        print(f"  {int(r['rank']):<4} S{int(r['sector']):<6} "
              f"Y{int(r['year']):<4} "
              f"{r['rms_ppm'] or 'N/A':<10} "
              f"{r['snr'] or 'N/A':<7} "
              f"{r['stability'] or 'N/A':<11} "
              f"{int(r['n_usable']):<8} "
              f"{r['completeness']:.0%}{'':6}"
              f"{r['score']:.3f}")
    print(f"\n  Best sector: S{int(df.iloc[0]['sector'])}  "
          f"(score={df.iloc[0]['score']:.3f}, "
          f"SNR={df.iloc[0]['snr']}, "
          f"RMS={df.iloc[0]['rms_ppm']} ppm, "
          f"{int(df.iloc[0]['n_usable'])} usable transits)")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, max(4, len(df)*0.45+2)))
    metrics = [
        ("rms_ppm",    "Baseline RMS (ppm)",  True,  "Lower is better"),
        ("snr",        "Transit SNR",          False, "Higher is better"),
        ("stability",  "Baseline stability\n(1=flat, <1=variable)", True, "Closer to 1 = flatter"),
        ("score",      "Composite score",      False, "Higher is better"),
    ]
    sec_labels = [f"S{int(r['sector'])} Y{int(r['year'])}" for _, r in df.iterrows()]
    colors     = [YEAR_COLORS[int(r['year'])] for _, r in df.iterrows()]

    for ax, (col, label, invert, hint) in zip(axes, metrics):
        vals = df[col].fillna(0).values
        bars = ax.barh(range(len(df)), vals, color=colors, edgecolor="white", linewidth=0.4)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(sec_labels, fontsize=8)
        ax.set_xlabel(label, fontsize=9)
        ax.set_title(hint, fontsize=8, style="italic")
        ax.invert_yaxis()
        if invert:
            ax.invert_xaxis()
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f" {val:.1f}", va="center", fontsize=7)

    # Legend for TESS years
    from matplotlib.patches import Patch
    legend_els = [Patch(color=YEAR_COLORS[y], label=f"Year {y}") for y in sorted(YEAR_COLORS) if y in df["year"].values]
    fig.legend(handles=legend_els, loc="lower center", ncol=len(legend_els),
               fontsize=8, title="TESS Year", bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{args.planet} — Sector Quality Ranking\n"
                 f"Score weights: SNR 40%  RMS 30%  Stability 15%  Usable 10%  Complete 5%",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, f"{args.planet}_sector_ranking.png")
    plt.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()

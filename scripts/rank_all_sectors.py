#!/usr/bin/env python3
"""
rank_all_sectors.py

Runs sector ranking for ALL planets in the transit plan that have
multiple available sectors, then produces a master summary.

For single-sector planets, just records the sector as-is (no choice needed).

Outputs:
  results/sector_rankings/{planet}_sector_ranking.csv  — per planet
  results/sector_rankings/{planet}_sector_ranking.png  — per planet
  results/sector_rankings/master_sector_selection.csv  — one row per planet:
      planet, recommended_sector, score, snr, rms_ppm, n_usable, n_sectors_available

Usage:
    python scripts/rank_all_sectors.py
    python scripts/rank_all_sectors.py --skip-single   # only rank multi-sector planets
"""

import argparse
import os
import re
import subprocess
import sys
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
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_DIR      = os.path.join(RESULTS_DIR, "sector_rankings")
PER_PLANET_DIR = os.path.join(OUT_DIR, "per_planet")


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
    phase = (time - t0) % period
    phase[phase > period / 2] -= period
    out_mask = np.abs(phase) > half_T * 1.2
    if out_mask.sum() < 20:
        return np.nan, np.nan
    t_out, f_out = time[out_mask], flux[out_mask]
    try:
        coeffs = np.polyfit(t_out, f_out, 1)
        f_det  = f_out - np.polyval(coeffs, t_out)
    except Exception:
        f_det  = f_out - np.median(f_out)
    raw_rms = float(np.std(f_out) * 1e6)
    det_rms = float(np.std(f_det) * 1e6)
    stab    = det_rms / raw_rms if raw_rms > 0 else 1.0
    return det_rms, stab


def norm_series(series, invert=False):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    n = (series - mn) / (mx - mn)
    return 1 - n if invert else n


def rank_planet(planet, host, period, T_days, depth, sectors, usable_map, t0_map):
    half_T = T_days / 2.0
    rows = []
    for sec in sectors:
        n_usable = usable_map.get(sec, 0)
        t0_sec   = t0_map.get(sec, None)
        if t0_sec is None:
            continue
        lc = download_lc(host, sec)
        if lc is None:
            print(f"    S{sec:02d} download failed — skipping")
            continue
        time_arr = np.array(lc.time.value) if hasattr(lc.time, "value") else np.array(lc.time)
        flux_arr = np.array(lc.flux.value) if hasattr(lc.flux, "value") else np.array(lc.flux)
        rms, stab = baseline_rms(time_arr, flux_arr, t0_sec, period, half_T)
        snr  = (depth / rms) if (depth and not np.isnan(rms) and rms > 0) else np.nan
        dur  = time_arr.max() - time_arr.min()
        comp = min(len(time_arr) / (dur * 24 * 60 / 2), 1.0)
        rows.append(dict(sector=sec, n_usable=n_usable,
                         rms_ppm=round(rms,1) if not np.isnan(rms) else None,
                         stability=round(stab,3) if not np.isnan(stab) else None,
                         snr=round(snr,2) if not np.isnan(snr) else None,
                         completeness=round(comp,3)))
        print(f"    S{sec:02d}  rms={rms:.0f} ppm  snr={snr:.1f}  usable={n_usable}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["score"] = (
        norm_series(df["snr"].fillna(0))                                    * 0.40 +
        norm_series(df["rms_ppm"].fillna(df["rms_ppm"].max()), invert=True) * 0.30 +
        norm_series(df["stability"].fillna(1), invert=True)                 * 0.15 +
        norm_series(df["n_usable"])                                         * 0.10 +
        norm_series(df["completeness"])                                     * 0.05
    )
    df["score"] = df["score"].round(3)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df.insert(0, "planet", planet)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-single", action="store_true",
                        help="Skip planets with only one available sector")
    args = parser.parse_args()

    plan = pd.read_csv(os.path.join(RESULTS_DIR, "summary", "transit_plan_detailed.csv"))
    pri  = pd.read_csv(os.path.join(DATA_DIR, "planets_priority.csv"))
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PER_PLANET_DIR, exist_ok=True)

    planets = plan["planet"].unique()
    master_rows = []

    for i, planet in enumerate(planets):
        prow = pri[pri["System"] == planet]
        if prow.empty:
            print(f"[{i+1}/{len(planets)}] {planet} — not in priority CSV, skipping")
            continue
        prow = prow.iloc[0]

        host   = re.sub(r"\s+[A-D]$", "", str(prow.get("host_star", planet))).strip()
        period = float(prow["Period"])
        T_days = float(prow["pl_trandur"]) / 24.0

        if pd.notna(prow.get("pl_trandep")) and float(prow["pl_trandep"]) > 0:
            depth = float(prow["pl_trandep"]) * 10000
        elif pd.notna(prow.get("pl_ratror")) and float(prow["pl_ratror"]) > 0:
            depth = float(prow["pl_ratror"]) ** 2 * 1e6
        else:
            depth = None

        # Load sector_report
        rep_path = os.path.join(RESULTS_DIR, "survey", planet, "sector_report.csv")
        usable_map, t0_map = {}, {}
        if os.path.exists(rep_path):
            rep = pd.read_csv(rep_path)
            rep = rep[rep["sector"] != "Total"]
            rep["sector"] = rep["sector"].astype(int)
            rep = rep[rep["n_usable"] > 0]
            for _, r in rep.iterrows():
                usable_map[int(r["sector"])] = int(r["n_usable"])
                t0_map[int(r["sector"])]     = float(r["t0_sec_btjd"])

        sectors = sorted(usable_map.keys())
        n_avail = len(sectors)

        if n_avail == 0:
            print(f"[{i+1}/{len(planets)}] {planet} — no sector_report, skipping")
            continue

        if n_avail == 1 and args.skip_single:
            sec = sectors[0]
            print(f"[{i+1}/{len(planets)}] {planet} — single sector S{sec}, skipping rank")
            master_rows.append(dict(
                planet=planet, recommended_sector=sec,
                score="N/A", snr="N/A", rms_ppm="N/A",
                n_usable=usable_map[sec], n_sectors_available=1, note="single sector"
            ))
            continue

        print(f"\n[{i+1}/{len(planets)}] {planet} — {n_avail} sectors: {sectors}")
        df = rank_planet(planet, host, period, T_days, depth,
                         sectors, usable_map, t0_map)

        if df.empty:
            print(f"  No data — skipping")
            continue

        # Save per-planet CSV and PNG
        csv_path = os.path.join(PER_PLANET_DIR, f"{planet}_sector_ranking.csv")
        df.to_csv(csv_path, index=False)

        # Per-planet bar chart
        fig, axes = plt.subplots(1, 4, figsize=(16, max(4, len(df)*0.45+2)))
        metrics = [
            ("rms_ppm",   "Baseline RMS (ppm)",   True),
            ("snr",       "Transit SNR",            False),
            ("stability", "Stability (1=flat)",     True),
            ("score",     "Composite score",         False),
        ]
        sec_labels = [f"S{int(r['sector'])}" for _, r in df.iterrows()]
        for ax, (col, label, invert) in zip(axes, metrics):
            vals = df[col].fillna(0).values
            bars = ax.barh(range(len(df)), vals, color="#2196F3",
                           edgecolor="white", linewidth=0.4)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(sec_labels, fontsize=9)
            ax.set_xlabel(label, fontsize=9)
            ax.invert_yaxis()
            if invert:
                ax.invert_xaxis()
            ax.spines[["top", "right"]].set_visible(False)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f" {val:.1f}", va="center", fontsize=7)
        fig.suptitle(f"{planet} — Sector Quality Ranking",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        png_path = os.path.join(PER_PLANET_DIR, f"{planet}_sector_ranking.png")
        plt.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.close()

        best = df.iloc[0]
        master_rows.append(dict(
            planet=planet,
            recommended_sector=int(best["sector"]),
            score=best["score"],
            snr=best["snr"],
            rms_ppm=best["rms_ppm"],
            n_usable=int(best["n_usable"]),
            n_sectors_available=n_avail,
            note="ranked"
        ))
        print(f"  -> Best: S{int(best['sector'])}  "
              f"score={best['score']}  SNR={best['snr']}  "
              f"RMS={best['rms_ppm']} ppm  usable={int(best['n_usable'])}")

    # Master summary
    df_master = pd.DataFrame(master_rows)
    master_path = os.path.join(OUT_DIR, "master_sector_selection.csv")
    df_master.to_csv(master_path, index=False)

    # Summary plot
    ranked = df_master[df_master["note"] == "ranked"].copy()
    if not ranked.empty:
        ranked["snr_f"]   = pd.to_numeric(ranked["snr"], errors="coerce")
        ranked["rms_f"]   = pd.to_numeric(ranked["rms_ppm"], errors="coerce")
        ranked["score_f"] = pd.to_numeric(ranked["score"], errors="coerce")
        ranked = ranked.sort_values("score_f", ascending=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, max(5, len(ranked)*0.42+2)))
        for ax, (col, label, invert) in zip(axes, [
            ("rms_f",   "Best-sector RMS (ppm)", True),
            ("snr_f",   "Best-sector SNR",        False),
            ("score_f", "Composite score",         False),
        ]):
            vals  = ranked[col].fillna(0).values
            names = ranked["planet"].values
            secs  = ranked["recommended_sector"].values
            bars  = ax.barh(range(len(ranked)), vals,
                            color="#2196F3", edgecolor="white", linewidth=0.4)
            ax.set_yticks(range(len(ranked)))
            ax.set_yticklabels([f"{p}  →S{s}" for p, s in zip(names, secs)], fontsize=8)
            ax.set_xlabel(label, fontsize=9)
            if invert:
                ax.invert_xaxis()
            ax.spines[["top","right"]].set_visible(False)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                        f" {val:.1f}", va="center", fontsize=7)

        fig.suptitle("Master Sector Selection — Multi-Sector Planets\n"
                     "Arrow shows recommended sector per planet",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        png_path = os.path.join(OUT_DIR, "master_sector_selection.png")
        plt.savefig(png_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"\nSaved summary plot: {png_path}")

    print(f"\nSaved master CSV: {master_path}")
    print(f"\n{'Planet':<22} {'Best Sector':>12} {'SNR':>7} {'RMS(ppm)':>10} {'Score':>7} {'N avail':>8}")
    print("-" * 72)
    for _, r in df_master.iterrows():
        print(f"{r['planet']:<22} S{str(r['recommended_sector']):<11} "
              f"{str(r['snr']):>7} {str(r['rms_ppm']):>10} "
              f"{str(r['score']):>7} {str(r['n_sectors_available']):>8}")


if __name__ == "__main__":
    main()

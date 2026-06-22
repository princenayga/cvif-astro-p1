#!/usr/bin/env python3
"""
run_all_pipeline.py

Full pipeline for all N30 planets:
  Step 1 — Auto-fill transit_selection.csv for planets not yet visually confirmed
            (uses primary sector from sector_selection.csv + refined t0 from rank cache)
  Step 2 — Export 5 CSVs per transit  (calls export_transit_csv.py)
  Step 3 — Run MSD diagnostics on residuals  (calls run_msd_analysis.py --types resid)

Skips planets already in transit_selection.csv unless --force is given.

Usage:
    python scripts/run_all_pipeline.py               # all planets, skip done ones
    python scripts/run_all_pipeline.py --force        # redo everything
    python scripts/run_all_pipeline.py --planet KELT-09
    python scripts/run_all_pipeline.py --skip-export  # MSD only (CSVs already exist)
    python scripts/run_all_pipeline.py --skip-msd     # export only
    python scripts/run_all_pipeline.py --step1-only   # just auto-fill transit_selection.csv
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
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
N30_DIR     = os.path.join(RESULTS_DIR, "N30")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

SEL_CSV     = os.path.join(N30_DIR, "sector_selection.csv")
TSEL_CSV    = os.path.join(N30_DIR, "transit_selection.csv")
CACHE_CSV   = os.path.join(N30_DIR, "_rank_cache.csv")

GAP_THRESHOLD_DAYS = 10 / 60 / 24   # 10 min


# ── LC utilities ──────────────────────────────────────────────────────────────

def robust_normalize(lc):
    f = np.array(lc.flux)
    if hasattr(f.flat[0], "value"):
        f = np.array([v.value for v in f])
    div = np.nanpercentile(f, 75)
    return lk.LightCurve(time=lc.time, flux=f / (div if div else 1.0))


def download_lc(host, sector, retries=3):
    delays = [15, 30]
    for attempt in range(retries):
        try:
            sr = lk.search_lightcurve(host, mission="TESS", sector=sector,
                                      author="SPOC", exptime=120)
            if not sr or len(sr) == 0:
                return None
            lc = sr[0].download(flux_column="pdcsap_flux")
            return robust_normalize(lc.remove_nans()) if lc else None
        except Exception as e:
            if attempt < retries - 1:
                wait = delays[min(attempt, len(delays)-1)]
                print(f"    retry {attempt+1} ({e}) — wait {wait}s")
                time_mod.sleep(wait)
            else:
                print(f"    download failed: {e}")
    return None


def refine_t0(time, flux, t0_init, period, half_dur, n_bins=40):
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
        inner = np.ones(len(bt), bool)
    try:
        a, b, _ = np.polyfit(bt[inner], bf[inner], 2)
        if a <= 0:
            return t0_init
        off = -b / (2*a)
        return t0_init + off if abs(off) <= half_dur else t0_init
    except Exception:
        return t0_init


def has_gap(time, tc, T_days):
    mask = (time >= tc - 0.5*T_days) & (time <= tc + 0.5*T_days)
    t_in = np.sort(time[mask])
    if len(t_in) < 2:
        return True
    return bool(np.any(np.diff(t_in) > GAP_THRESHOLD_DAYS))


def get_transit_centers(tarr, t0, period, T_days):
    half_T = T_days / 2.0
    n_lo = int(np.ceil((tarr.min()  - t0) / period))
    n_hi = int(np.floor((tarr.max() - t0) / period))
    tcs = []
    for n in range(n_lo, n_hi+1):
        tc = t0 + n * period
        if tarr.min() > tc + 1.5*T_days or tarr.max() < tc - 1.5*T_days:
            continue
        pre  = ((tarr >= tc - 1.5*T_days) & (tarr < tc - half_T)).sum()
        in_t = ((tarr >= tc - half_T)      & (tarr <= tc + half_T)).sum()
        post = ((tarr >  tc + half_T)      & (tarr <= tc + 1.5*T_days)).sum()
        if pre >= 8 and in_t >= 8 and post >= 8 and not has_gap(tarr, tc, T_days):
            tcs.append(round(tc, 6))
    return tcs


# ── Step 1: auto-fill transit_selection.csv ───────────────────────────────────

def auto_fill_transit_selection(planets_todo, sel_df, pri_df, rank_cache, force=False):
    """Download primary sector LC and generate transit_selection rows."""

    existing = pd.read_csv(TSEL_CSV) if os.path.exists(TSEL_CSV) else pd.DataFrame()

    new_rows = []
    for planet in planets_todo:
        if not force and not existing.empty and planet in existing["planet"].values:
            n = (existing["planet"] == planet).sum()
            print(f"  [{planet}] already has {n} transits in transit_selection.csv — skip")
            continue

        psel  = sel_df[sel_df["planet"] == planet]
        sector = int(psel.iloc[0]["sector"])

        prow = pri_df[pri_df["System"] == planet]
        if prow.empty:
            print(f"  [{planet}] not in planets_priority.csv — skip"); continue
        prow = prow.iloc[0]

        host   = re.sub(r"\s+[A-D]$", "", str(prow.get("host_star", planet))).strip()
        period = float(prow["Period"])
        T_days = float(prow["pl_trandur"]) / 24.0
        half_T = T_days / 2.0

        # Best t0 seed: rank cache refined, else sector_report, else lit
        t0_init = None
        if not rank_cache.empty:
            rc = rank_cache[(rank_cache["planet"] == planet) & (rank_cache["sector"] == sector)]
            if not rc.empty:
                if "t0_refined_btjd" in rc.columns and pd.notna(rc.iloc[0]["t0_refined_btjd"]):
                    t0_init = float(rc.iloc[0]["t0_refined_btjd"])
                elif "t0_lit_btjd" in rc.columns:
                    t0_init = float(rc.iloc[0]["t0_lit_btjd"])

        if t0_init is None:
            rep_path = os.path.join(RESULTS_DIR, "survey", planet, "sector_report.csv")
            if os.path.exists(rep_path):
                rep = pd.read_csv(rep_path)
                rep = rep[rep["sector"] != "Total"]
                rep["sector"] = rep["sector"].astype(int)
                rr = rep[rep["sector"] == sector]
                if not rr.empty:
                    t0_init = float(rr.iloc[0]["t0_sec_btjd"])

        print(f"\n  [{planet}] S{sector} — downloading...", end=" ", flush=True)
        lc = download_lc(host, sector)
        if lc is None:
            print("FAILED"); continue

        tarr = np.array(lc.time.value) if hasattr(lc.time, "value") else np.array(lc.time)
        farr = np.array(lc.flux.value) if hasattr(lc.flux, "value") else np.array(lc.flux)
        print(f"{len(tarr)} pts")

        if t0_init is None:
            t0_init = tarr.min()

        t0 = refine_t0(tarr, farr, t0_init, period, half_T)
        tcs = get_transit_centers(tarr, t0, period, T_days)
        print(f"  t0={t0:.6f} BTJD  |  {len(tcs)} usable transits found")

        for i, tc in enumerate(tcs):
            new_rows.append(dict(
                planet           = planet,
                global_n         = i + 1,
                sector           = sector,
                transit_in_sector= i + 1,
                tc_btjd          = tc,
            ))

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if not existing.empty and not force:
            # Remove any stale rows for these planets then append
            planets_new = new_df["planet"].unique()
            existing = existing[~existing["planet"].isin(planets_new)]
            final_df = pd.concat([existing, new_df], ignore_index=True)
        else:
            final_df = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

        os.makedirs(N30_DIR, exist_ok=True)
        final_df.to_csv(TSEL_CSV, index=False)
        print(f"\nStep 1 done. transit_selection.csv updated ({len(new_rows)} new rows).")
    else:
        print("\nStep 1: no new rows added.")


# ── Step 2 & 3: subprocess wrappers ───────────────────────────────────────────

def run_export(planet, force=False):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "export_transit_csv.py"),
           "--planet", planet]
    print(f"\n  [export] {planet}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [export] FAILED for {planet}")
    return result.returncode == 0


def run_msd(planet, types=None, force=False):
    cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "run_msd_analysis.py"),
           "--planet", planet]
    if types:
        cmd += ["--types"] + types
    if force:
        cmd.append("--force")
    print(f"\n  [msd]    {planet}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [msd] FAILED for {planet}")
    return result.returncode == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet",      type=str, default=None,
                        help="Process a single planet only")
    parser.add_argument("--force",       action="store_true",
                        help="Redo all steps even if outputs exist")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip CSV export (use existing transit_csvs/)")
    parser.add_argument("--skip-msd",    action="store_true",
                        help="Skip MSD analysis")
    parser.add_argument("--types",       type=str, nargs="+", default=None,
                        help="MSD data types to process: pre post raw model resid "
                             "(default: all five)")
    parser.add_argument("--step1-only",  action="store_true",
                        help="Only run Step 1 (auto-fill transit_selection.csv)")
    args = parser.parse_args()

    sel_df = pd.read_csv(SEL_CSV)
    pri_df = pd.read_csv(os.path.join(DATA_DIR, "planets_priority.csv"))
    rank_cache = pd.read_csv(CACHE_CSV) if os.path.exists(CACHE_CSV) else pd.DataFrame()

    if args.planet:
        planets = [args.planet]
    else:
        planets = list(sel_df["planet"].values)

    print(f"Pipeline: {len(planets)} planet(s)")
    print(f"  force={args.force}  skip_export={args.skip_export}  skip_msd={args.skip_msd}")
    print("=" * 60)

    # Step 1 — auto-fill transit_selection.csv for any planet not yet done
    print("\n── Step 1: Auto-fill transit selections ──")
    tsel_existing = pd.read_csv(TSEL_CSV) if os.path.exists(TSEL_CSV) else pd.DataFrame()
    already_done  = set(tsel_existing["planet"].unique()) if not tsel_existing.empty else set()

    planets_need_step1 = [p for p in planets
                          if args.force or p not in already_done]

    if planets_need_step1:
        auto_fill_transit_selection(planets_need_step1, sel_df, pri_df,
                                    rank_cache, force=args.force)
    else:
        print("  All planets already have transit selections.")

    if args.step1_only:
        print("\nStep 1 only — done.")
        return

    # Step 2 — export CSVs
    if not args.skip_export:
        print("\n── Step 2: Export transit CSVs ──")
        export_ok = {}
        for planet in planets:
            export_ok[planet] = run_export(planet, force=args.force)
    else:
        export_ok = {p: True for p in planets}

    # Step 3 — MSD on residuals
    if not args.skip_msd:
        print("\n── Step 3: MSD diagnostics (residuals) ──")
        msd_ok = {}
        for planet in planets:
            if not args.skip_export and not export_ok.get(planet, False):
                print(f"  [{planet}] skipping MSD — export failed")
                msd_ok[planet] = False
                continue
            msd_ok[planet] = run_msd(planet, types=args.types, force=args.force)

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    tsel_final = pd.read_csv(TSEL_CSV) if os.path.exists(TSEL_CSV) else pd.DataFrame()
    tsel_counts = tsel_final.groupby("planet").size().to_dict() if not tsel_final.empty else {}

    msd_root = os.path.join(N30_DIR, "msd_diagnostics")
    for planet in planets:
        n_sel   = tsel_counts.get(planet, 0)
        n_csv   = 0
        n_msd   = 0
        planet_csv_root = os.path.join(BASE_DIR, "transit_csvs", planet)
        planet_msd_root = os.path.join(msd_root, planet)
        if os.path.exists(planet_csv_root):
            for sd in os.listdir(planet_csv_root):
                sec_path = os.path.join(planet_csv_root, sd)
                if os.path.isdir(sec_path):
                    for td in os.listdir(sec_path):
                        t_path = os.path.join(sec_path, td)
                        if os.path.isdir(t_path):
                            n_csv += 1
        if os.path.exists(planet_msd_root):
            for sd in os.listdir(planet_msd_root):
                sec_path = os.path.join(planet_msd_root, sd)
                if os.path.isdir(sec_path):
                    n_msd += len([f for f in os.listdir(sec_path) if f.endswith("_resid_msd.png")])
        exp_sym = "✓" if export_ok.get(planet, False) else "✗" if not args.skip_export else "-"
        msd_sym = "✓" if msd_ok.get(planet, False)   else "✗" if not args.skip_msd   else "-"
        print(f"  {planet:<22} sel={n_sel:>3}  csvs={n_csv:>3}  msd={n_msd:>3}  "
              f"export={exp_sym}  msd={msd_sym}")

    print(f"\nDiagnostic images: {msd_root}")
    if os.path.exists(os.path.join(msd_root, "master_msd_summary.csv")):
        print(f"Master MSD summary: {os.path.join(msd_root, 'master_msd_summary.csv')}")
    print("Done.")


if __name__ == "__main__":
    main()

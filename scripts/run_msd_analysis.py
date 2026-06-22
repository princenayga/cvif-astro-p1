#!/usr/bin/env python3
"""
run_msd_analysis.py

Runs the whitenoise SWNA pipeline on every transit CSV exported by
export_transit_csv.py. For each transit and each data type
(pre_baseline, in_transit_raw, in_transit_model, in_transit_resid,
post_baseline), produces a diagnostic image and appends one row to a
summary CSV.

Output structure:
  results/N30/msd_diagnostics/
    {planet}/
      sector_{sec:03d}/
        transit_01_pre_baseline_msd.png
        transit_01_in_transit_raw_msd.png
        transit_01_in_transit_model_msd.png
        transit_01_in_transit_resid_msd.png
        transit_01_post_baseline_msd.png
        ...
      summary.csv          ← one row per (transit, data_type)

Usage:
    python scripts/run_msd_analysis.py --planet WASP-019 --sector 63
    python scripts/run_msd_analysis.py --planet WASP-019 --sector 63 --transit 1
    python scripts/run_msd_analysis.py --planet WASP-019 --sector 63 --model cosine
    python scripts/run_msd_analysis.py --planet WASP-019 --sector 63 --types resid
    python scripts/run_msd_analysis.py --all
"""

import argparse
import os
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")

# Install whitenoise if not available
try:
    import whitenoise as wn
except ModuleNotFoundError:
    print("whitenoise not found — installing from GitHub...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/pnayga/whitenoise.git", "-q"
    ])
    import whitenoise as wn
    print(f"whitenoise {wn.__version__} installed.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_ROOT    = os.path.join(BASE_DIR, "transit_csvs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUT_ROOT    = os.path.join(RESULTS_DIR, "N30", "msd_diagnostics")

# Map CSV filename stem → (value column, display label, short tag)
CSV_TYPES = {
    "pre_baseline":     ("flux",       "Pre-baseline flux",         "pre"),
    "in_transit_raw":   ("flux",       "In-transit raw flux",       "raw"),
    "in_transit_model": ("flux_model", "In-transit batman model",   "model"),
    "in_transit_resid": ("residual",   "In-transit residuals",      "resid"),
    "post_baseline":    ("flux",       "Post-baseline flux",        "post"),
}


def run_one(planet, sector, transit_n, csv_type, model, force=False):
    """Run wn.analyze on one CSV file. Returns a result-summary dict."""
    transit_dir = os.path.join(CSV_ROOT, planet,
                               f"sector_{sector:03d}",
                               f"transit_{transit_n:02d}")
    csv_path = os.path.join(transit_dir, f"{csv_type}.csv")
    if not os.path.exists(csv_path):
        return None

    out_dir = os.path.join(OUT_ROOT, planet, f"sector_{sector:03d}")
    os.makedirs(out_dir, exist_ok=True)

    tag      = CSV_TYPES[csv_type][2]
    img_name = f"transit_{transit_n:02d}_{tag}_msd.png"
    img_path = os.path.join(out_dir, img_name)

    if os.path.exists(img_path) and not force:
        print(f"    {img_name} — exists, skip")
        # Try to read existing summary row from CSV
        return {"planet": planet, "sector": sector, "transit_n": transit_n,
                "data_type": csv_type, "tag": tag, "image": img_name,
                "status": "skipped (cached)"}

    # Load CSV
    df = pd.read_csv(csv_path)
    val_col = CSV_TYPES[csv_type][0]
    if val_col not in df.columns:
        print(f"    {csv_type}: column '{val_col}' not found — skip")
        return None

    values = df[val_col].dropna().values
    if len(values) < 10:
        print(f"    {csv_type}: only {len(values)} points — skip")
        return None

    # Normalize residuals/flux around zero mean for MSD analysis
    values = values - np.mean(values)

    label = f"{planet} S{sector} T{transit_n:02d} — {CSV_TYPES[csv_type][1]}"

    # Run full whitenoise pipeline
    try:
        result = wn.analyze(values, model=model, label=label)
    except Exception as e:
        print(f"    {csv_type}: wn.analyze failed — {e}")
        return {"planet": planet, "sector": sector, "transit_n": transit_n,
                "data_type": csv_type, "tag": tag, "image": img_name,
                "status": f"error: {e}", "mu": None, "r2": None}

    # Extract key parameters
    mu_key = "H" if model == "fbm" else "mu"
    mu_val = result.fit.params.get(mu_key, None)
    r2_val = result.fit.r_squared
    regime = _regime(mu_val, model)

    # Plot diagnostics
    try:
        fig = wn.plot_diagnostics(result)
        fig.suptitle(f"{label}\nmodel={model}  {mu_key}={mu_val:.4f}  R²={r2_val:.4f}  [{regime}]",
                     fontsize=9, y=1.01)
        plt.tight_layout()
        plt.savefig(img_path, dpi=130, bbox_inches="tight")
        plt.close("all")
        print(f"    {img_name}  {mu_key}={mu_val:.4f}  R²={r2_val:.4f}  [{regime}]")
    except Exception as e:
        plt.close("all")
        print(f"    {csv_type}: plot failed — {e}")

    return {
        "planet":    planet,
        "sector":    sector,
        "transit_n": transit_n,
        "data_type": csv_type,
        "tag":       tag,
        "model":     model,
        mu_key:      round(mu_val, 5) if mu_val is not None else None,
        "r2":        round(r2_val, 5),
        "regime":    regime,
        "n_points":  len(values),
        "image":     img_name,
        "status":    "ok",
    }


def _regime(mu, model):
    if model == "fbm":
        if mu is None: return "?"
        if mu < 0.475:  return "Subdiffusive"
        if mu < 0.525:  return "Near-Brownian"
        return "Superdiffusive"
    if mu is None: return "?"
    if mu < 0.95:   return "Subdiffusive"
    if mu <= 1.05:  return "Near-Brownian"
    if mu <= 2.0:   return "Superdiffusive"
    return "Hyperballistic"


def process_planet_sector(planet, sector, transit_filter=None,
                           types_filter=None, model="exponential", force=False):
    """Process all transits for one planet/sector combination."""
    sec_dir = os.path.join(CSV_ROOT, planet, f"sector_{sector:03d}")
    if not os.path.exists(sec_dir):
        print(f"  No CSV directory: {sec_dir}")
        return []

    transit_dirs = sorted(
        d for d in os.listdir(sec_dir)
        if d.startswith("transit_") and os.path.isdir(os.path.join(sec_dir, d))
    )
    if not transit_dirs:
        print(f"  No transit folders found in {sec_dir}")
        return []

    types_to_run = list(CSV_TYPES.keys())
    if types_filter:
        types_to_run = [t for t in types_to_run
                        if any(f in t for f in types_filter)]

    rows = []
    for td in transit_dirs:
        transit_n = int(td.split("_")[1])
        if transit_filter and transit_n not in transit_filter:
            continue

        print(f"  Transit {transit_n:02d}")
        for csv_type in types_to_run:
            row = run_one(planet, sector, transit_n, csv_type, model, force)
            if row:
                rows.append(row)

    # Save/update per-planet summary CSV
    if rows:
        out_dir  = os.path.join(OUT_ROOT, planet, f"sector_{sector:03d}")
        sum_path = os.path.join(out_dir, "summary.csv")
        new_df   = pd.DataFrame(rows)
        if os.path.exists(sum_path) and not force:
            old_df = pd.read_csv(sum_path)
            key_cols = ["planet", "sector", "transit_n", "data_type"]
            old_df = old_df[~old_df.set_index(key_cols).index.isin(
                new_df.set_index(key_cols).index)]
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_csv(sum_path, index=False)
        print(f"  Summary: {sum_path}")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet",  type=str, default=None,
                        help="Planet name (e.g. WASP-019)")
    parser.add_argument("--sector",  type=int, default=None,
                        help="Sector number")
    parser.add_argument("--transit", type=int, nargs="+", default=None,
                        help="Specific transit number(s) to process")
    parser.add_argument("--types",   type=str, nargs="+", default=None,
                        help="Data types to process: pre post raw model resid "
                             "(default: all). Use 'resid' for residuals only.")
    parser.add_argument("--model",   type=str, default="exponential",
                        choices=["exponential", "cosine", "sine", "fbm"],
                        help="whitenoise model to fit (default: exponential)")
    parser.add_argument("--force",   action="store_true",
                        help="Re-run even if output image already exists")
    parser.add_argument("--all",     action="store_true",
                        help="Process all planets/sectors found in transit_csvs/")
    args = parser.parse_args()

    all_rows = []

    if args.all:
        if not os.path.exists(CSV_ROOT):
            print(f"No transit_csvs/ directory found."); return
        planets = sorted(os.listdir(CSV_ROOT))
        for planet in planets:
            planet_dir = os.path.join(CSV_ROOT, planet)
            if not os.path.isdir(planet_dir): continue
            sec_dirs = sorted(d for d in os.listdir(planet_dir)
                              if d.startswith("sector_"))
            for sd in sec_dirs:
                sector = int(sd.split("_")[1])
                print(f"\n[{planet}] Sector {sector}")
                rows = process_planet_sector(
                    planet, sector,
                    transit_filter=args.transit,
                    types_filter=args.types,
                    model=args.model,
                    force=args.force,
                )
                all_rows.extend(rows)
    else:
        if not args.planet:
            parser.error("Provide --planet or use --all")
        if not args.sector:
            # Auto-detect sectors from directory
            planet_dir = os.path.join(CSV_ROOT, args.planet)
            if not os.path.exists(planet_dir):
                print(f"No CSVs found for {args.planet}"); return
            sec_dirs = sorted(d for d in os.listdir(planet_dir)
                              if d.startswith("sector_"))
            sectors = [int(sd.split("_")[1]) for sd in sec_dirs]
        else:
            sectors = [args.sector]

        for sector in sectors:
            print(f"\n[{args.planet}] Sector {sector}")
            rows = process_planet_sector(
                args.planet, sector,
                transit_filter=args.transit,
                types_filter=args.types,
                model=args.model,
                force=args.force,
            )
            all_rows.extend(rows)

    # Master summary across all processed planets
    if all_rows:
        os.makedirs(OUT_ROOT, exist_ok=True)
        master_path = os.path.join(OUT_ROOT, "master_msd_summary.csv")
        master_df   = pd.DataFrame(all_rows)
        if os.path.exists(master_path) and not args.force:
            old = pd.read_csv(master_path)
            key_cols = ["planet", "sector", "transit_n", "data_type"]
            old = old[~old.set_index(key_cols).index.isin(
                master_df.set_index(key_cols).index)]
            master_df = pd.concat([old, master_df], ignore_index=True)
        master_df.to_csv(master_path, index=False)
        print(f"\nMaster summary: {master_path}")

        # Quick console report
        ok = master_df[master_df["status"] == "ok"]
        if not ok.empty:
            mu_col = "H" if args.model == "fbm" else "mu"
            if mu_col in ok.columns:
                print(f"\n{'Planet':<22} {'Sector':>7} {'Transit':>8} {'Type':<22} "
                      f"{mu_col:>6} {'R²':>7} {'Regime'}")
                print("-" * 85)
                for _, r in ok.iterrows():
                    print(f"{r['planet']:<22} S{int(r['sector']):<6} "
                          f"T{int(r['transit_n']):02d}{'':5} "
                          f"{r['data_type']:<22} "
                          f"{r.get(mu_col, '?'):>6} "
                          f"{r['r2']:>7.4f}  {r.get('regime','?')}")

    print(f"\nDone. Images in: {OUT_ROOT}")


if __name__ == "__main__":
    main()

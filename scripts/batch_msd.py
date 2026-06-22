"""
Batch MSD analysis using the whitenoise package — exponential model.

For each OK transit in transit_csvs/, this script:
  1. Writes a whitenoise-compatible CSV  (header: 'time [BTJD], flux []')
  2. Calls wn.analyze(file, model='exponential')
  3. Saves the 4-panel wn.plot_diagnostics(result) figure to
       results/msd_plots/<planet>/<sector>/<transit>_msd.png
  4. Collects mu, beta, R2 into
       results/msd_results.csv
       results/msd_results.xlsx  (sheets: All Transits, Planet Summary)

Skips the 3 CHECK transits from the alignment diagnostic.

Usage:
    python scripts/batch_msd.py
"""

import os, glob, tempfile, traceback, warnings

# Force UTF-8 so whitenoise warning symbols don't crash on Windows cp1252
os.environ['PYTHONUTF8'] = '1'
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import whitenoise as wn

warnings.filterwarnings('ignore')

MODEL = 'cosine'   # whitenoise model to use

BASE_DIR     = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
TRANSIT_ROOT = os.path.join(BASE_DIR, 'transit_csvs')
PLOT_ROOT    = os.path.join(BASE_DIR, 'results', f'msd_plots_{MODEL}')
OUT_DIR      = os.path.join(BASE_DIR, 'results')

SKIP = {
    ('TOI-157',  'sector_011', 'transit_06'),
    ('TOI-157',  'sector_011', 'transit_07'),
    ('WASP-019', 'sector_063', 'transit_17'),
}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Gather all merged_raw.csv files ──────────────────────────────────────────
all_files = []
for pdir in sorted(glob.glob(os.path.join(TRANSIT_ROOT, '*', ''))):
    planet = os.path.basename(pdir.rstrip('/\\'))
    for sdir in sorted(glob.glob(os.path.join(pdir, 'sector_*', ''))):
        sector = os.path.basename(sdir.rstrip('/\\'))
        for tdir in sorted(glob.glob(os.path.join(sdir, 'transit_*', ''))):
            transit = os.path.basename(tdir.rstrip('/\\'))
            if (planet, sector, transit) in SKIP:
                continue
            csv_path = os.path.join(tdir, 'merged_raw.csv')
            if os.path.exists(csv_path):
                all_files.append((planet, sector, transit, tdir, csv_path))

total = len(all_files)
print(f'Model: {MODEL}')
print(f'Found {total} transits to analyse.\n')

rows = []

for idx, (planet, sector, transit, tdir, csv_path) in enumerate(all_files, 1):
    print(f'[{idx}/{total}] {planet:<14} {sector} {transit}', end='  ', flush=True)

    df = pd.read_csv(csv_path)
    if len(df) < 20 or 'flux' not in df.columns:
        print('SKIP (too few rows)')
        continue

    # Write whitenoise-compatible temp CSV
    tmp = tempfile.NamedTemporaryFile(
        suffix='.csv', delete=False, mode='w', encoding='utf-8'
    )
    tmp.write('time [BTJD], flux []\n')
    for _, row in df.iterrows():
        tmp.write(f"{row['time_btjd']},{row['flux']}\n")
    tmp.close()

    mu = nu = r2 = N_coeff = np.nan
    fit_mode = 'N/A'

    try:
        result = wn.analyze(tmp.name, model=MODEL, verbose=False)

        if result is not None and result.fit is not None:
            p = result.fit.params
            mu      = p.get('mu', np.nan)
            nu      = p.get('nu', np.nan)
            r2      = result.fit.r_squared
            N_coeff = p.get('N', np.nan)
            fit_mode = getattr(result.fit, 'fit_mode', 'N/A')

            # Save 4-panel diagnostic plot from wn.plot_diagnostics
            plot_dir = os.path.join(PLOT_ROOT, planet, sector)
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'{transit}_msd.png')

            fig = wn.plot_diagnostics(result)
            fig.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close(fig)

            mu_s  = f'{mu:.4f}'  if np.isfinite(mu)  else 'N/A'
            nu_s  = f'{nu:.6f}'  if np.isfinite(nu)   else 'N/A'
            r2_s  = f'{r2:.4f}'  if np.isfinite(r2)   else 'N/A'
            print(f'mu={mu_s}  nu={nu_s}  R2={r2_s}')
        else:
            print('fit failed (no result)')

    except Exception:
        print('ERROR')
        traceback.print_exc()

    finally:
        os.unlink(tmp.name)

    rows.append({
        'planet':   planet,
        'sector':   sector,
        'transit':  transit,
        'n_points': len(df),
        'mu':       round(float(mu),  6) if np.isfinite(mu)  else np.nan,
        'nu':       round(float(nu),  8) if np.isfinite(nu)   else np.nan,
        'N_coeff':  f'{N_coeff:.4e}'     if np.isfinite(N_coeff) else 'nan',
        'fit_mode': fit_mode,
        'R2':       round(float(r2),  6) if np.isfinite(r2)   else np.nan,
    })

# ── Save results ──────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
csv_out    = os.path.join(OUT_DIR, f'msd_results_{MODEL}.csv')
xlsx_out   = os.path.join(OUT_DIR, f'msd_results_{MODEL}.xlsx')

results_df.to_csv(csv_out, index=False)

with pd.ExcelWriter(xlsx_out, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='All Transits', index=False)

    numeric = results_df[['planet', 'mu', 'nu', 'R2']].dropna(subset=['mu'])
    summary = numeric.groupby('planet').agg(
        n_transits=('mu', 'count'),
        mu_mean=('mu', 'mean'),
        mu_std=('mu', 'std'),
        nu_mean=('nu', 'mean'),
        nu_std=('nu', 'std'),
        R2_mean=('R2', 'mean'),
    ).reset_index().round(6)
    summary.to_excel(writer, sheet_name='Planet Summary', index=False)

print(f'\nDone. {len(rows)} transits analysed.')
print(f'  CSV:  results/msd_results_{MODEL}.csv')
print(f'  XLSX: results/msd_results_{MODEL}.xlsx')
print(f'  Plots: results/msd_plots_{MODEL}/<planet>/<sector>/<transit>_msd.png')

"""
run_swna_pipeline.py
--------------------
Runs the full SWNA whitenoise pipeline on all 2-column pipeline CSVs
for the 6 selected exoplanet transit datasets.

Models: cosine, exponential
Output: results/swna/<model>/<planet>/<transit>_diagnostics.png
        results/swna/swna_summary.csv
        results/swna/swna_summary.xlsx

Usage:
    python scripts/run_swna_pipeline.py
"""

import sys
import io
from pathlib import Path

# Fix Windows console UTF-8 (emoji in whitenoise print statements)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Add whitenoise package to path ───────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
WN_DIR       = ROOT_DIR / 'whitenoise'
PIPELINE_DIR = ROOT_DIR / 'data' / 'selected_transits' / 'pipeline'
OUTPUT_DIR   = ROOT_DIR / 'results' / 'swna'

sys.path.insert(0, str(WN_DIR))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import whitenoise as wn

# ── Config ────────────────────────────────────────────────────────────────────
MODELS  = ['cosine', 'exponential']
PLANETS = ['CoRoT-01', 'HATS-13', 'TOI-4773', 'TrEs-5', 'WASP-072', 'WASP-078']

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'whitenoise {wn.__version__}')
    print(f'Pipeline dir : {PIPELINE_DIR}')
    print(f'Output dir   : {OUTPUT_DIR}')
    print()

    summary_rows = []

    for model in MODELS:
        print(f'=== Model: {model.upper()} ===')

        for planet in PLANETS:
            planet_csv_dir = PIPELINE_DIR / planet
            if not planet_csv_dir.exists():
                print(f'  [SKIP] {planet} -- folder not found')
                continue

            csv_files = sorted(planet_csv_dir.glob('*.csv'))
            if not csv_files:
                print(f'  [SKIP] {planet} -- no CSVs found')
                continue

            print(f'\n  {planet} ({len(csv_files)} transit(s))')
            out_dir = OUTPUT_DIR / model / planet
            out_dir.mkdir(parents=True, exist_ok=True)

            for csv_path in csv_files:
                transit_name = csv_path.stem   # e.g. WASP-078_T01

                try:
                    result = wn.analyze(str(csv_path), model=model, detrend_method=None)
                except Exception as exc:
                    print(f'    [FAIL] {transit_name}: {exc}')
                    continue

                result.summary()

                fig = wn.plot_diagnostics(result)
                png_path = out_dir / f'{transit_name}_diagnostics.png'
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                if result.fit is None:
                    print(f'    [FAIL] {transit_name}: fit returned None')
                    continue

                params = result.fit.params
                r2     = result.fit.r_squared
                regime = result.regime
                mu_val = params.get('mu', params.get('H', float('nan')))

                print(f'    [OK]  {transit_name:<22}  mu={mu_val:.4f}  R2={r2:.4f}  [{regime}]')

                summary_rows.append({
                    'planet':    planet,
                    'transit':   transit_name,
                    'model':     model,
                    'r_squared': r2,
                    'regime':    regime,
                    **params,
                })

        print()

    # ── Save summary CSV and Excel ────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        id_cols    = ['planet', 'transit', 'model', 'r_squared', 'regime']
        param_cols = [c for c in summary_df.columns if c not in id_cols]
        summary_df = summary_df[id_cols + param_cols]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_out  = OUTPUT_DIR / 'swna_summary.csv'
        xlsx_out = OUTPUT_DIR / 'swna_summary.xlsx'

        summary_df.to_csv(csv_out, index=False)
        print(f'Summary CSV  --> {csv_out}')

        try:
            summary_df.to_excel(xlsx_out, index=False, sheet_name='SWNA Results')
            print(f'Summary XLSX --> {xlsx_out}')
        except ImportError:
            print('[WARN] openpyxl not installed -- Excel skipped. Run: pip install openpyxl')

        print()
        mu_col = 'mu' if 'mu' in summary_df.columns else list(param_cols)[0]
        print(summary_df[['planet', 'transit', 'model', mu_col, 'r_squared', 'regime']].to_string(index=False))
    else:
        print('No results to save.')


if __name__ == '__main__':
    main()

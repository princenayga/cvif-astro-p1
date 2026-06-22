"""
For each transit_csvs/<planet>/<sector>/<transit>/ folder:
  - pre_baseline.csv  : flux near 1.0  -> residual = flux - 1.0
  - post_baseline.csv : flux near 1.0  -> residual = flux - 1.0
  - in_transit_resid.csv : residual column already computed

Outputs one merged CSV per transit:
  transit_csvs/<planet>/<sector>/<transit>/merged_residuals.csv

Columns: time_btjd, residual, segment
  segment values: 'pre_baseline', 'in_transit', 'post_baseline'
"""

import pandas as pd
import os

ROOT = os.path.join(os.path.dirname(__file__), '..', 'transit_csvs')
ROOT = os.path.normpath(ROOT)


def load_baseline(path):
    df = pd.read_csv(path)
    df['residual'] = df['flux'] - 1.0
    return df[['time_btjd', 'residual']]


def load_transit_resid(path):
    df = pd.read_csv(path)
    return df[['time_btjd', 'residual']]


def process_transit(transit_dir):
    pre_path   = os.path.join(transit_dir, 'pre_baseline.csv')
    post_path  = os.path.join(transit_dir, 'post_baseline.csv')
    resid_path = os.path.join(transit_dir, 'in_transit_resid.csv')

    missing = [p for p in [pre_path, post_path, resid_path] if not os.path.exists(p)]
    if missing:
        print(f'  SKIP (missing files): {[os.path.basename(m) for m in missing]}')
        return

    pre   = load_baseline(pre_path)
    pre['segment'] = 'pre_baseline'

    post  = load_baseline(post_path)
    post['segment'] = 'post_baseline'

    resid = load_transit_resid(resid_path)
    resid['segment'] = 'in_transit'

    merged = pd.concat([pre, resid, post], ignore_index=True)
    merged = merged.sort_values('time_btjd').reset_index(drop=True)
    merged = merged[['time_btjd', 'residual', 'segment']]

    out_path = os.path.join(transit_dir, 'merged_residuals.csv')
    merged.to_csv(out_path, index=False)
    return len(merged)


total_transits = 0
total_written  = 0

for planet in sorted(os.listdir(ROOT)):
    planet_dir = os.path.join(ROOT, planet)
    if not os.path.isdir(planet_dir):
        continue

    for sector in sorted(os.listdir(planet_dir)):
        sector_dir = os.path.join(planet_dir, sector)
        if not os.path.isdir(sector_dir):
            continue

        for transit in sorted(os.listdir(sector_dir)):
            transit_dir = os.path.join(sector_dir, transit)
            if not os.path.isdir(transit_dir):
                continue

            total_transits += 1
            label = f'{planet}/{sector}/{transit}'
            n = process_transit(transit_dir)
            if n is not None:
                total_written += 1
                print(f'  OK  {label}  ({n} rows)')
            else:
                print(f'  --  {label}')

print()
print(f'Done. {total_written}/{total_transits} transits processed.')

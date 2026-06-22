"""
Merge pre_baseline, in_transit_raw, and post_baseline CSVs
into one raw flux CSV per transit.

Output per transit:
  transit_csvs/Qatar-1/sector_024/transit_NN/merged_raw.csv
    columns: time_btjd, flux, segment
"""

import os
import glob
import pandas as pd

PLANET  = 'Qatar-1'
SECTOR  = 24

BASE_DIR     = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
TRANSIT_ROOT = os.path.join(BASE_DIR, 'transit_csvs', PLANET, f'sector_{SECTOR:03d}')

dirs = sorted(glob.glob(os.path.join(TRANSIT_ROOT, 'transit_*', '')))
print(f'Found {len(dirs)} transits for {PLANET} S{SECTOR}\n')

for tdir in dirs:
    label = os.path.basename(tdir.rstrip('/\\'))
    pre_path  = os.path.join(tdir, 'pre_baseline.csv')
    in_path   = os.path.join(tdir, 'in_transit_raw.csv')
    post_path = os.path.join(tdir, 'post_baseline.csv')

    if not all(os.path.exists(p) for p in [pre_path, in_path, post_path]):
        print(f'  {label}: missing files, skipping')
        continue

    pre  = pd.read_csv(pre_path)
    pre['segment'] = 'pre_baseline'

    intrans = pd.read_csv(in_path)
    intrans['segment'] = 'in_transit'

    post = pd.read_csv(post_path)
    post['segment'] = 'post_baseline'

    merged = pd.concat([pre, intrans, post], ignore_index=True)\
               .sort_values('time_btjd').reset_index(drop=True)

    out_path = os.path.join(tdir, 'merged_raw.csv')
    merged.to_csv(out_path, index=False)

    n_pre  = len(pre)
    n_in   = len(intrans)
    n_post = len(post)
    print(f'  {label}: pre={n_pre} in={n_in} post={n_post} total={len(merged)} -> merged_raw.csv')

print(f'\nDone. merged_raw.csv written in each transit folder.')

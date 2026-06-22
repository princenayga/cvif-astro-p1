"""
For every planet/sector/transit folder already extracted, merge
pre_baseline, in_transit_raw, and post_baseline into merged_raw.csv.

Also prints an alignment diagnostic:
  median_in   = median flux inside transit window
  median_out  = median flux of baselines
  depth_pct   = (median_out - median_in) / median_out * 100
  status      = OK if depth > 0.1%, CHECK if not (possible misalignment)
"""

import os
import glob
import pandas as pd
import numpy as np

BASE_DIR     = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
TRANSIT_ROOT = os.path.join(BASE_DIR, 'transit_csvs')

planet_dirs = sorted(glob.glob(os.path.join(TRANSIT_ROOT, '*', '')))

summary_rows = []
n_ok = 0
n_check = 0

for pdir in planet_dirs:
    planet = os.path.basename(pdir.rstrip('/\\'))
    sector_dirs = sorted(glob.glob(os.path.join(pdir, 'sector_*', '')))

    for sdir in sector_dirs:
        sector = os.path.basename(sdir.rstrip('/\\'))
        transit_dirs = sorted(glob.glob(os.path.join(sdir, 'transit_*', '')))

        for tdir in transit_dirs:
            label = os.path.basename(tdir.rstrip('/\\'))
            pre_path  = os.path.join(tdir, 'pre_baseline.csv')
            in_path   = os.path.join(tdir, 'in_transit_raw.csv')
            post_path = os.path.join(tdir, 'post_baseline.csv')

            if not all(os.path.exists(p) for p in [pre_path, in_path, post_path]):
                continue

            pre     = pd.read_csv(pre_path)
            intrans = pd.read_csv(in_path)
            post    = pd.read_csv(post_path)

            pre['segment']     = 'pre_baseline'
            intrans['segment'] = 'in_transit'
            post['segment']    = 'post_baseline'

            merged = pd.concat([pre, intrans, post], ignore_index=True)\
                       .sort_values('time_btjd').reset_index(drop=True)

            out_path = os.path.join(tdir, 'merged_raw.csv')
            merged.to_csv(out_path, index=False)

            # ── Alignment diagnostic ──────────────────────────────────────
            median_in  = np.median(intrans['flux'].values)
            median_out = np.median(np.concatenate([pre['flux'].values,
                                                   post['flux'].values]))
            depth_pct  = (median_out - median_in) / median_out * 100
            status     = 'OK' if depth_pct > 0.10 else 'CHECK'

            if status == 'OK':
                n_ok += 1
            else:
                n_check += 1

            summary_rows.append({
                'planet':     planet,
                'sector':     sector,
                'transit':    label,
                'n_pre':      len(pre),
                'n_in':       len(intrans),
                'n_post':     len(post),
                'depth_pct':  round(depth_pct, 3),
                'status':     status,
            })

# ── Print summary ─────────────────────────────────────────────────────────────
df = pd.DataFrame(summary_rows)
print(f"{'Planet':<14} {'Sector':<12} {'Transit':<12} "
      f"{'pre':>5} {'in':>5} {'post':>5} {'depth%':>8}  status")
print('-' * 72)
for _, r in df.iterrows():
    print(f"{r.planet:<14} {r.sector:<12} {r.transit:<12} "
          f"{r.n_pre:>5} {r.n_in:>5} {r.n_post:>5} {r.depth_pct:>8.3f}  {r.status}")

print(f"\nTotal: {len(df)} transits  |  OK: {n_ok}  |  CHECK: {n_check}")

# Save summary
out_csv = os.path.join(BASE_DIR, 'results', 'transit_alignment_check.csv')
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"Alignment summary saved to: results/transit_alignment_check.csv")

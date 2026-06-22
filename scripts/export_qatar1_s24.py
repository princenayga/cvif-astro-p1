"""
Export transit CSVs for Qatar-1 Sector 24.

Produces per-transit:
  transit_csvs/Qatar-1/sector_024/transit_NN/
    pre_baseline.csv       (time_btjd, flux)
    in_transit_raw.csv     (time_btjd, flux)
    in_transit_model.csv   (time_btjd, flux_model)
    in_transit_resid.csv   (time_btjd, residual)   = raw - model
    post_baseline.csv      (time_btjd, flux)
    merged_residuals.csv   (time_btjd, residual, segment)
                            pre/post: flux - 1.0
                            in_transit: raw - model

Qatar-1 parameters (TEPCat / Alsubai et al. 2011, updated 2017):
  R_b = 1.143 Rjup, R_A = 0.803 Rsun -> k = 0.1429
  a = 0.02332 AU, R_A = 0.803 Rsun   -> a/Rs = 6.245
  P = 1.42 d, ecc = 0, Teff = 4820 K -> u1=0.44, u2=0.25
  inc = 84.0 deg  (Alsubai et al. 2011)
  T14 ~ 2.0 hr
"""

import os, sys, time as time_mod, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightkurve as lk
import batman
from scipy.optimize import minimize_scalar

# ── Qatar-1 parameters ────────────────────────────────────────────────────────
PLANET   = 'Qatar-1'
HOST     = 'Qatar-1'
SECTOR   = 24
PERIOD   = 1.42002504       # days (Maciejewski et al. 2015 value)
RJUP_RSUN = 0.10045
R_B      = 1.143            # Rjup
R_A      = 0.803            # Rsun
K        = R_B * RJUP_RSUN / R_A   # Rp/Rs = 0.14291
A_RS     = 6.245            # a/Rs
INC      = 84.0             # degrees
ECC      = 0.0
U1, U2   = 0.44, 0.25      # quadratic LD for Teff~4820K
T14_HR   = 2.00             # hours (approximate)
T_DAYS   = T14_HR / 24.0   # transit duration in days

GAP_THRESHOLD_DAYS = 10 / 60 / 24   # 10-minute gap threshold

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
OUT_ROOT = os.path.join(BASE_DIR, 'transit_csvs')


# ── Download ──────────────────────────────────────────────────────────────────
def download_lc(host, sector, retries=4):
    for attempt in range(retries):
        try:
            sr = lk.search_lightcurve(host, mission='TESS', sector=sector,
                                       author='SPOC', exptime=120)
            if sr is None or len(sr) == 0:
                return None
            lc = sr[0].download(flux_column='pdcsap_flux')
            if lc is None:
                return None
            lc = lc.remove_nans()
            flux = np.array(lc.flux.value)
            med  = np.nanpercentile(flux, 75)
            return np.array(lc.time.value), flux / med
        except Exception as e:
            if attempt < retries - 1:
                print(f'  Retry {attempt+1}: {e}'); time_mod.sleep(15)
            else:
                print(f'  Download failed: {e}'); return None


# ── Batman model ──────────────────────────────────────────────────────────────
def batman_model(tarr, tc):
    p = batman.TransitParams()
    p.t0 = tc; p.per = PERIOD; p.rp = K
    p.a  = A_RS; p.inc = INC; p.ecc = ECC; p.w = 90.0
    p.u  = [U1, U2]; p.limb_dark = 'quadratic'
    try:
        return batman.TransitModel(p, tarr).light_curve(p)
    except Exception:
        return np.ones_like(tarr)


# ── t0 refinement: stacking ───────────────────────────────────────────────────
def refine_t0_stacking(tarr, farr, t0_init):
    half_dur = T_DAYS / 2
    phase = (tarr - t0_init) % PERIOD
    phase[phase > PERIOD / 2] -= PERIOD
    mask = np.abs(phase) <= 2 * half_dur
    if mask.sum() < 20:
        return t0_init
    p_in, f_in = phase[mask], farr[mask]
    bins = np.linspace(-2*half_dur, 2*half_dur, 41)
    bct  = 0.5*(bins[:-1]+bins[1:])
    bt, bf = [], []
    for i in range(40):
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
        a2, b2, _ = np.polyfit(bt[inner], bf[inner], 2)
        if a2 <= 0:
            return t0_init
        offset = -b2 / (2*a2)
        return t0_init + offset if abs(offset) <= half_dur else t0_init
    except Exception:
        return t0_init


# ── Per-transit tc refinement: batman chi2 ────────────────────────────────────
def refine_tc_batman(tarr, farr, tc_guess):
    half_search = 0.6 * T_DAYS
    mask = (tarr >= tc_guess - 1.5*T_DAYS) & (tarr <= tc_guess + 1.5*T_DAYS)
    if mask.sum() < 10:
        return tc_guess
    t_fit, f_fit = tarr[mask], farr[mask]

    def chi2(shift):
        model = batman_model(t_fit, tc_guess + shift)
        out = np.abs(t_fit - (tc_guess + shift)) > 0.6*T_DAYS
        if out.sum() > 3:
            scale = np.median(f_fit[out]) / np.median(model[out])
            model = model * scale
        return np.sum((f_fit - model)**2)

    try:
        res = minimize_scalar(chi2, bounds=(-half_search, half_search), method='bounded')
        return tc_guess + np.clip(res.x, -half_search, half_search)
    except Exception:
        return tc_guess


def has_gap(tarr, tc):
    mask = (tarr >= tc - 0.5*T_DAYS) & (tarr <= tc + 0.5*T_DAYS)
    t_in = np.sort(tarr[mask])
    if len(t_in) < 2:
        return True
    return bool(np.any(np.diff(t_in) > GAP_THRESHOLD_DAYS))


# ── Export CSVs ───────────────────────────────────────────────────────────────
def export_transit(tarr, farr, tc, transit_n):
    half_T = T_DAYS / 2.0

    # 5T window
    mask_win = (tarr >= tc - 2.5*T_DAYS) & (tarr <= tc + 2.5*T_DAYS)
    t_win = tarr[mask_win]
    f_win = farr[mask_win]

    # batman model — rescale to data baseline
    model = batman_model(t_win, tc)
    out_m = np.abs(t_win - tc) > 0.6 * T_DAYS
    if out_m.sum() > 3:
        scale = np.median(f_win[out_m]) / np.median(model[out_m])
        model = model * scale

    resid_all = f_win - model

    in_mask   = (t_win >= tc - half_T) & (t_win <= tc + half_T)
    pre_mask  = (t_win >= tc - 1.5*T_DAYS) & (t_win < tc - half_T)
    post_mask = (t_win > tc + half_T) & (t_win <= tc + 1.5*T_DAYS)

    out_dir = os.path.join(OUT_ROOT, PLANET, f'sector_{SECTOR:03d}', f'transit_{transit_n:02d}')
    os.makedirs(out_dir, exist_ok=True)

    # Raw files
    pd.DataFrame({'time_btjd': t_win[pre_mask],  'flux': f_win[pre_mask]})\
      .to_csv(os.path.join(out_dir, 'pre_baseline.csv'), index=False)
    pd.DataFrame({'time_btjd': t_win[post_mask], 'flux': f_win[post_mask]})\
      .to_csv(os.path.join(out_dir, 'post_baseline.csv'), index=False)
    pd.DataFrame({'time_btjd': t_win[in_mask],   'flux': f_win[in_mask]})\
      .to_csv(os.path.join(out_dir, 'in_transit_raw.csv'), index=False)
    pd.DataFrame({'time_btjd': t_win[in_mask],   'flux_model': model[in_mask]})\
      .to_csv(os.path.join(out_dir, 'in_transit_model.csv'), index=False)
    pd.DataFrame({'time_btjd': t_win[in_mask],   'residual': resid_all[in_mask]})\
      .to_csv(os.path.join(out_dir, 'in_transit_resid.csv'), index=False)

    # Merged residuals: baseline residual = flux - 1.0, transit residual = raw - model
    pre_df = pd.DataFrame({
        'time_btjd': t_win[pre_mask],
        'residual':  f_win[pre_mask] - 1.0,
        'segment':   'pre_baseline'
    })
    in_df = pd.DataFrame({
        'time_btjd': t_win[in_mask],
        'residual':  resid_all[in_mask],
        'segment':   'in_transit'
    })
    post_df = pd.DataFrame({
        'time_btjd': t_win[post_mask],
        'residual':  f_win[post_mask] - 1.0,
        'segment':   'post_baseline'
    })
    merged = pd.concat([pre_df, in_df, post_df], ignore_index=True)\
               .sort_values('time_btjd').reset_index(drop=True)
    merged.to_csv(os.path.join(out_dir, 'merged_residuals.csv'), index=False)

    n_pre  = int(pre_mask.sum())
    n_in   = int(in_mask.sum())
    n_post = int(post_mask.sum())
    print(f'  Transit {transit_n:02d}  tc={tc:.5f}  '
          f'pre={n_pre} in={n_in} post={n_post}  -> {out_dir}/')
    return len(merged)


# ── Main ──────────────────────────────────────────────────────────────────────
print(f'Downloading {PLANET} Sector {SECTOR}...')
result = download_lc(HOST, SECTOR)
if result is None:
    print('Download failed. Exiting.'); sys.exit(1)

tarr, farr = result
print(f'Downloaded {len(tarr)} points  ({tarr.min():.3f} – {tarr.max():.3f} BTJD)')

# ── Find longest gap-free segment (same logic as inspect notebooks) ───────────
diffs      = np.diff(tarr)
gap_idx    = np.where(diffs > GAP_THRESHOLD_DAYS)[0]
boundaries = np.concatenate([[-1], gap_idx, [len(tarr) - 1]])
best_len, best_start, best_end = 0, 0, 0
for i in range(len(boundaries) - 1):
    s = boundaries[i] + 1
    e = boundaries[i + 1]
    seg_len = tarr[e] - tarr[s]
    if seg_len > best_len:
        best_len  = seg_len
        best_start = s
        best_end   = e

seg_t0_time = tarr[best_start]
seg_t1_time = tarr[best_end]
seg_mask    = np.zeros(len(tarr), bool)
seg_mask[best_start: best_end + 1] = True

print(f'Longest gap-free segment: {seg_t0_time:.3f} – {seg_t1_time:.3f} BTJD '
      f'({best_len:.2f} d, {seg_mask.sum()} pts)')

# Restrict analysis to segment only
tarr_seg = tarr[seg_mask]
farr_seg = farr[seg_mask]

# Find t0 via full phase scan over one period (robust against noise)
best_t0, best_depth = tarr_seg[0], 0
for t0_try in np.linspace(tarr_seg[0], tarr_seg[0] + PERIOD, 500):
    phase = (tarr_seg - t0_try) % PERIOD
    phase[phase > PERIOD/2] -= PERIOD
    in_t = np.abs(phase) <= T_DAYS/2
    if in_t.sum() < 5:
        continue
    depth = 1.0 - np.median(farr_seg[in_t])
    if depth > best_depth:
        best_depth = depth
        best_t0 = t0_try

t0_sec = refine_t0_stacking(tarr_seg, farr_seg, best_t0)
print(f'Phase-scan t0 = {best_t0:.5f} BTJD  (depth {best_depth*100:.3f}%)')
print(f'Refined t0    = {t0_sec:.5f} BTJD')

# Generate all candidate transit centers within segment bounds
n_lo = int(np.ceil((tarr_seg.min() - t0_sec) / PERIOD))
n_hi = int(np.floor((tarr_seg.max() - t0_sec) / PERIOD))
all_tc = [t0_sec + n * PERIOD for n in range(n_lo, n_hi + 1)]

# Validate: require pre, in, post windows each with enough points, no in-transit gap
# Use full tarr for window extraction (so baseline can extend slightly outside segment)
valid_tcs = []
for tc_c in all_tc:
    # transit center must be within segment
    if tc_c < seg_t0_time or tc_c > seg_t1_time:
        continue
    pre  = ((tarr >= tc_c - 1.5*T_DAYS) & (tarr <  tc_c - T_DAYS/2)).sum()
    in_t = ((tarr >= tc_c - T_DAYS/2)   & (tarr <= tc_c + T_DAYS/2)).sum()
    post = ((tarr >  tc_c + T_DAYS/2)   & (tarr <= tc_c + 1.5*T_DAYS)).sum()
    if pre >= 8 and in_t >= 8 and post >= 8 and not has_gap(tarr, tc_c):
        valid_tcs.append(tc_c)

print(f'Valid transits found: {len(valid_tcs)}')

total_rows = 0
for i, tc_pred in enumerate(valid_tcs, start=1):
    tc_refined = refine_tc_batman(tarr, farr, tc_pred)
    n = export_transit(tarr, farr, tc_refined, i)  # full tarr/farr for window extraction
    total_rows += n

print(f'\nDone. {len(valid_tcs)} transits exported for {PLANET} S{SECTOR}.')

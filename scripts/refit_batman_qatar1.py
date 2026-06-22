"""
Refit batman model per transit for Qatar-1 S24.
Optimizes (tc, k) jointly via chi2 minimization against TESS data,
then rewrites in_transit_resid.csv and merged_residuals.csv with corrected residuals.
"""

import pandas as pd
import numpy as np
import batman
import glob, os, warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# ── Fixed parameters ──────────────────────────────────────────────────────────
PLANET  = 'Qatar-1'
SECTOR  = 24
PERIOD  = 1.42002504
A_RS    = 6.245
INC     = 84.0
ECC     = 0.0
U1, U2  = 0.44, 0.25
T_DAYS  = 2.00 / 24.0
K_INIT  = 1.143 * 0.10045 / 0.803   # starting guess from TEPCat

TRANSIT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'transit_csvs',
                 PLANET, f'sector_{SECTOR:03d}'))


def batman_flux(tarr, tc, k):
    p = batman.TransitParams()
    p.t0 = tc; p.per = PERIOD; p.rp = abs(k)
    p.a  = A_RS; p.inc = INC; p.ecc = ECC; p.w = 90.0
    p.u  = [U1, U2]; p.limb_dark = 'quadratic'
    try:
        return batman.TransitModel(p, tarr).light_curve(p)
    except Exception:
        return np.ones_like(tarr)


def fit_transit(t_win, f_win, tc_guess):
    """Fit tc and k jointly via chi2 on the full 3T window."""
    half_T  = T_DAYS / 2

    def chi2(params):
        tc_shift, k = params
        tc = tc_guess + tc_shift
        model = batman_flux(t_win, tc, k)
        # Rescale model to data out-of-transit baseline
        out = np.abs(t_win - tc) > 0.6 * T_DAYS
        if out.sum() > 3:
            scale = np.median(f_win[out]) / np.median(model[out])
            model = model * scale
        return np.sum((f_win - model) ** 2)

    res = minimize(chi2,
                   x0=[0.0, K_INIT],
                   method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 5000})
    tc_shift_fit, k_fit = res.x
    return tc_guess + tc_shift_fit, abs(k_fit)


dirs = sorted(glob.glob(os.path.join(TRANSIT_ROOT, 'transit_*', '')))
print(f'Refitting {len(dirs)} transits for {PLANET} S{SECTOR}')
print(f'Initial k={K_INIT:.5f} (TEPCat),  fitting tc + k per transit\n')
print(f"{'Transit':<12} {'tc_old':>12} {'tc_new':>12} {'shift_min':>10} "
      f"{'k_old':>8} {'k_new':>8} {'depth_new%':>10} {'mean_resid':>11} {'status'}")
print('-' * 95)

summary = []

for tdir in dirs:
    label = os.path.basename(tdir.rstrip('/'))
    try:
        raw  = pd.read_csv(os.path.join(tdir, 'in_transit_raw.csv'))
        pre  = pd.read_csv(os.path.join(tdir, 'pre_baseline.csv'))
        post = pd.read_csv(os.path.join(tdir, 'post_baseline.csv'))
        mdf  = pd.read_csv(os.path.join(tdir, 'in_transit_model.csv'))
    except Exception as e:
        print(f'{label:<12}  ERROR loading: {e}')
        continue

    # Assemble 3T window
    t_win = np.concatenate([pre['time_btjd'].values,
                             raw['time_btjd'].values,
                             post['time_btjd'].values])
    f_win = np.concatenate([pre['flux'].values,
                             raw['flux'].values,
                             post['flux'].values])
    sort_idx = np.argsort(t_win)
    t_win, f_win = t_win[sort_idx], f_win[sort_idx]

    tc_old = mdf['time_btjd'].values[np.argmin(mdf['flux_model'].values)]

    # Fit
    tc_new, k_new = fit_transit(t_win, f_win, tc_old)
    shift_min = (tc_new - tc_old) * 24 * 60

    # Recompute model and residuals with fitted params
    half_T   = T_DAYS / 2
    model_new = batman_flux(t_win, tc_new, k_new)
    out       = np.abs(t_win - tc_new) > 0.6 * T_DAYS
    if out.sum() > 3:
        scale     = np.median(f_win[out]) / np.median(model_new[out])
        model_new = model_new * scale

    in_mask   = (t_win >= tc_new - half_T) & (t_win <= tc_new + half_T)
    pre_mask  = (t_win >= tc_new - 1.5*T_DAYS) & (t_win < tc_new - half_T)
    post_mask = (t_win > tc_new + half_T) & (t_win <= tc_new + 1.5*T_DAYS)

    resid_all = f_win - model_new
    resid_in  = resid_all[in_mask]
    mean_resid = np.mean(resid_in)
    std_base   = np.std(np.concatenate([f_win[pre_mask] - 1,
                                        f_win[post_mask] - 1]))
    status = 'OK' if abs(mean_resid) < std_base else 'CHECK'

    print(f'{label:<12} {tc_old:>12.5f} {tc_new:>12.5f} {shift_min:>+10.2f}m '
          f'{K_INIT:>8.5f} {k_new:>8.5f} {k_new**2*100:>10.3f}% '
          f'{mean_resid:>11.6f}  {status}')

    # ── Overwrite in_transit_resid.csv ─────────────────────────────────────
    pd.DataFrame({
        'time_btjd': t_win[in_mask],
        'residual':  resid_in
    }).to_csv(os.path.join(tdir, 'in_transit_resid.csv'), index=False)

    # ── Overwrite in_transit_model.csv ─────────────────────────────────────
    pd.DataFrame({
        'time_btjd':  t_win[in_mask],
        'flux_model': model_new[in_mask]
    }).to_csv(os.path.join(tdir, 'in_transit_model.csv'), index=False)

    # ── Overwrite merged_residuals.csv ─────────────────────────────────────
    pre_df = pd.DataFrame({
        'time_btjd': t_win[pre_mask],
        'residual':  f_win[pre_mask] - 1.0,
        'segment':   'pre_baseline'
    })
    in_df = pd.DataFrame({
        'time_btjd': t_win[in_mask],
        'residual':  resid_in,
        'segment':   'in_transit'
    })
    post_df = pd.DataFrame({
        'time_btjd': t_win[post_mask],
        'residual':  f_win[post_mask] - 1.0,
        'segment':   'post_baseline'
    })
    merged = pd.concat([pre_df, in_df, post_df], ignore_index=True)\
               .sort_values('time_btjd').reset_index(drop=True)
    merged.to_csv(os.path.join(tdir, 'merged_residuals.csv'), index=False)

    summary.append({'transit': label, 'tc': tc_new, 'k_fit': k_new,
                    'depth_pct': k_new**2*100, 'mean_resid_in': mean_resid,
                    'std_baseline': std_base})

print()
sdf = pd.DataFrame(summary)
print(f'Mean fitted k  = {sdf.k_fit.mean():.5f} ± {sdf.k_fit.std():.5f}')
print(f'Mean depth     = {sdf.depth_pct.mean():.3f}% ± {sdf.depth_pct.std():.3f}%')
print(f'Mean resid in  = {sdf.mean_resid_in.mean():.6f}')
print(f'\nAll merged_residuals.csv and model files updated.')

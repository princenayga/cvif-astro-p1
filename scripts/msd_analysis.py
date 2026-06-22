"""
White noise MSD analysis on per-transit merged_raw.csv files.
Follows Calotes (2023) methodology exactly:

  Empirical MSD (Eq. 3.1):
    MSD(dt) = 1/(N-dt) * sum_i [x(i+dt) - x(i)]^2

  Theoretical MSD (Eq. 3.5):
    MSD = N_coeff * sqrt(pi)*Gamma(mu)*cos(vT/2)*J_{mu-1/2}(vT/2) / (T/v)^(0.5-mu)

  Fit upper limit: first local minimum of empirical MSD, else N/2.

Outputs:
  results/msd_plots/<planet>/<sector>/<transit>_msd.png   diagnostic image
  results/msd_results.csv                                  parameter table
  results/msd_results.xlsx                                 same as Excel
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gamma, jv
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

BASE_DIR    = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
TRANSIT_ROOT = os.path.join(BASE_DIR, 'transit_csvs')
PLOT_ROOT   = os.path.join(BASE_DIR, 'results', 'msd_plots')
CADENCE_MIN = 2   # TESS 2-min cadence

# ── Skip transits flagged as CHECK (empty files) ───────────────────────────────
SKIP = {
    ('TOI-157',  'sector_011', 'transit_06'),
    ('TOI-157',  'sector_011', 'transit_07'),
    ('WASP-019', 'sector_063', 'transit_17'),
}

# ── Empirical MSD ──────────────────────────────────────────────────────────────
def empirical_msd(flux, max_lag):
    N = len(flux)
    lags = np.arange(1, max_lag + 1)
    msd  = np.array([np.mean((flux[dt:] - flux[:-dt])**2) for dt in lags])
    return lags * CADENCE_MIN, msd   # lag time in minutes


def first_local_min(msd, min_idx=3):
    """Index of first local minimum after min_idx."""
    for i in range(min_idx, len(msd) - 1):
        if msd[i] < msd[i - 1] and msd[i] < msd[i + 1]:
            return i
    return len(msd) - 1


# ── Theoretical MSD (Calotes Eq. 3.5) ─────────────────────────────────────────
def theoretical_msd(T, N_coeff, mu, nu):
    T    = np.asarray(T, dtype=float)
    arg  = nu * T / 2.0
    Jval = jv(mu - 0.5, arg)
    num  = np.sqrt(np.pi) * gamma(mu) * np.cos(arg) * Jval
    den  = (T / nu) ** (0.5 - mu)
    with np.errstate(invalid='ignore', divide='ignore'):
        val = N_coeff * num / den
    return np.where(np.isfinite(val), val, 0.0)


def fit_msd(lag_min, msd_emp):
    """Fit theoretical MSD; returns (N_coeff, mu, nu, r2)."""
    # Initial guesses
    p0 = [np.max(msd_emp) * 1e-1, 1.0, 0.005]
    bounds = ([0, 0.1, 1e-6], [np.inf, 3.0, 1.0])
    try:
        popt, _ = curve_fit(theoretical_msd, lag_min, msd_emp,
                            p0=p0, bounds=bounds,
                            maxfev=20000, method='trf')
        msd_fit = theoretical_msd(lag_min, *popt)
        ss_res  = np.sum((msd_emp - msd_fit)**2)
        ss_tot  = np.sum((msd_emp - np.mean(msd_emp))**2)
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return popt[0], popt[1], popt[2], round(r2, 4)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


# ── Main loop ──────────────────────────────────────────────────────────────────
rows = []
planet_dirs = sorted(glob.glob(os.path.join(TRANSIT_ROOT, '*', '')))

total = sum(
    len(glob.glob(os.path.join(pdir, 'sector_*', 'transit_*', 'merged_raw.csv')))
    for pdir in planet_dirs
)
done = 0

for pdir in planet_dirs:
    planet = os.path.basename(pdir.rstrip('/\\'))
    for sdir in sorted(glob.glob(os.path.join(pdir, 'sector_*', ''))):
        sector = os.path.basename(sdir.rstrip('/\\'))
        for tdir in sorted(glob.glob(os.path.join(sdir, 'transit_*', ''))):
            transit = os.path.basename(tdir.rstrip('/\\'))
            done += 1

            # Skip bad transits
            if (planet, sector, transit) in SKIP:
                print(f'[{done}/{total}] SKIP  {planet} {sector} {transit}')
                continue

            csv_path = os.path.join(tdir, 'merged_raw.csv')
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            if len(df) < 20 or 'flux' not in df.columns:
                continue

            flux = df['flux'].values

            # Empirical MSD — upper limit = first local min or N/2
            max_lag  = len(flux) // 2
            lag_min, msd_emp = empirical_msd(flux, max_lag)
            ulim_idx = first_local_min(msd_emp)
            lag_fit  = lag_min[:ulim_idx + 1]
            msd_fit_range = msd_emp[:ulim_idx + 1]

            # Fit
            N_coeff, mu, nu, r2 = fit_msd(lag_fit, msd_fit_range)
            msd_theory = theoretical_msd(lag_fit, N_coeff, mu, nu) \
                         if np.isfinite(mu) else np.full_like(lag_fit, np.nan)

            # ── Diagnostic plot ───────────────────────────────────────────────
            plot_dir = os.path.join(PLOT_ROOT, planet, sector)
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f'{transit}_msd.png')

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Left: raw flux
            t_col = 'time_btjd' if 'time_btjd' in df.columns else df.columns[0]
            colors = {'pre_baseline': '#aaaaaa', 'in_transit': '#c0392b',
                      'post_baseline': '#aaaaaa'}
            for seg, grp in df.groupby('segment'):
                axes[0].plot(grp[t_col], grp['flux'], '.',
                             ms=2, alpha=0.7,
                             color=colors.get(seg, 'steelblue'),
                             label=seg)
            axes[0].set_xlabel('BTJD (days)', fontsize=9)
            axes[0].set_ylabel('Normalized flux', fontsize=9)
            axes[0].set_title(f'{planet} {sector} {transit} — raw flux', fontsize=9)
            axes[0].legend(fontsize=7, markerscale=3)
            axes[0].tick_params(labelsize=8)

            # Right: MSD
            axes[1].plot(lag_min, msd_emp, 'o', ms=3, alpha=0.6,
                         color='black', label='Empirical')
            if np.isfinite(mu):
                axes[1].plot(lag_fit, msd_theory, '-', color='red', lw=1.8,
                             label='Theoretical')
            axes[1].axvline(lag_fit[-1], color='blue', lw=0.8, ls='--',
                            alpha=0.6, label='fit limit')
            axes[1].set_xlabel('Lag time (minutes)', fontsize=9)
            axes[1].set_ylabel('MSD', fontsize=9)
            mu_str  = f'{mu:.4f}'  if np.isfinite(mu)  else 'N/A'
            nu_str  = f'{nu:.6f}'  if np.isfinite(nu)  else 'N/A'
            r2_str  = f'{r2:.4f}'  if np.isfinite(r2)  else 'N/A'
            axes[1].set_title(
                f'mu={mu_str}  nu={nu_str}  R2={r2_str}', fontsize=9)
            axes[1].legend(fontsize=7)
            axes[1].tick_params(labelsize=8)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close(fig)

            rows.append({
                'planet':   planet,
                'sector':   sector,
                'transit':  transit,
                'n_points': len(flux),
                'fit_lag_limit_min': round(lag_fit[-1], 1),
                'mu':       round(mu, 5) if np.isfinite(mu) else np.nan,
                'nu':       round(nu, 7) if np.isfinite(nu) else np.nan,
                'N_coeff':  f'{N_coeff:.4e}' if np.isfinite(N_coeff) else 'nan',
                'R2':       r2,
            })

            status = f'mu={mu_str}  nu={nu_str}  R2={r2_str}'
            print(f'[{done}/{total}] {planet:<14} {sector} {transit}  {status}')

# ── Save results ───────────────────────────────────────────────────────────────
out_dir = os.path.join(BASE_DIR, 'results')
os.makedirs(out_dir, exist_ok=True)

results_df = pd.DataFrame(rows)
csv_path   = os.path.join(out_dir, 'msd_results.csv')
xlsx_path  = os.path.join(out_dir, 'msd_results.xlsx')

results_df.to_csv(csv_path, index=False)

with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='All Transits', index=False)

    # Summary sheet: mean ± std of mu and nu per planet
    summary = results_df.groupby('planet').agg(
        n_transits=('transit', 'count'),
        mu_mean=('mu', 'mean'),
        mu_std=('mu', 'std'),
        nu_mean=('nu', 'mean'),
        nu_std=('nu', 'std'),
        R2_mean=('R2', 'mean'),
    ).reset_index()
    summary = summary.round(5)
    summary.to_excel(writer, sheet_name='Planet Summary', index=False)

print(f'\nDone. {len(rows)} transits analysed.')
print(f'  CSV:  results/msd_results.csv')
print(f'  XLSX: results/msd_results.xlsx')
print(f'  Plots: results/msd_plots/<planet>/<sector>/<transit>_msd.png')

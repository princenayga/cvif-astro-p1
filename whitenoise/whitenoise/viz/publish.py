"""
viz/publish.py — Publication-quality plots for whitenoise analysis results.

Applies strict rcParams styling via a context manager so global state is
never modified.  All functions return a matplotlib Figure.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd


# ── Style and palette constants ────────────────────────────────────────────────

STYLE: dict = {
    'figure.dpi':       150,
    'font.family':      'serif',
    'font.size':        11,
    'axes.linewidth':   0.8,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
    'legend.frameon':   False,
}

PALETTES: dict[str, dict[str, str]] = {
    'default': {
        'empirical': '#555555',
        'theory':    '#1B3A6B',
        'pdf':       '#C0392B',
    },
    'colorblind': {
        'empirical': '#999999',
        'theory':    '#0072B2',
        'pdf':       '#D55E00',
    },
}


@contextlib.contextmanager
def _style_ctx():
    """Apply STYLE rcParams within a context, restore afterwards."""
    with matplotlib.rc_context(STYLE):
        yield


def _resolve_palette(palette: str) -> dict[str, str]:
    if palette not in PALETTES:
        raise ValueError(
            f"\u2717 Unknown palette '{palette}'. "
            f"Available: {list(PALETTES.keys())}"
        )
    return PALETTES[palette]


def _save(fig: matplotlib.figure.Figure, save_path: str | None) -> None:
    if save_path is None:
        return
    dirn = os.path.dirname(save_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f'\u2713 Saved to {save_path}')


def _unit_label(name: str, unit: str) -> str:
    if not unit or unit.lower() == 'unitless':
        return name
    return f'{name} ({unit})'


def _fd_bins(data: np.ndarray) -> int:
    n = len(data)
    if n < 2:
        return 10
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return max(10, int(np.sqrt(n)))
    h = 2.0 * iqr * n ** (-1.0 / 3.0)
    data_range = np.ptp(data)
    if h == 0 or data_range == 0:
        return 10
    return max(5, int(np.ceil(data_range / h)))


# ── publish_msd ───────────────────────────────────────────────────────────────

def publish_msd(
    result,
    palette:   str = 'default',
    figsize:   tuple = (5, 4),
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Publication-quality MSD plot: empirical scatter + fitted curve.

    Parameters
    ----------
    result : AnalysisResult
    palette : str, default ``'default'``
        ``'default'`` or ``'colorblind'``.
    figsize : tuple, default ``(5, 4)``
    save_path : str, optional
        If provided, save as PDF.

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors = _resolve_palette(palette)
    meta      = result.metadata
    time_unit = meta.get('time_unit', '')
    obs_unit  = meta.get('value_unit', '')
    dname     = result.dataset_name

    xlabel = f'Lag ({time_unit})' if time_unit else 'Lag'
    ylabel = f'MSD ({obs_unit}\u00b2)' if obs_unit else 'MSD'

    with _style_ctx():
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            result.lags, result.msd_empirical,
            s=10, color=colors['empirical'], alpha=0.6, label='Empirical MSD', zorder=3,
        )

        if result.fit is not None:
            r2    = result.fit.r_squared
            label = f'Fitted {result.model} (R\u00b2={r2:.4f})'
            lags_used  = result.fit.lags_used
            msd_fitted = result.fit.msd_fitted
            finite = np.isfinite(msd_fitted)
            ax.plot(
                lags_used[finite], msd_fitted[finite],
                color=colors['theory'], linewidth=1.8, label=label, zorder=4,
            )
        else:
            ax.annotate(
                'Fit not available',
                xy=(0.5, 0.85), xycoords='axes fraction',
                ha='center', fontsize=9, color='#C0392B',
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{dname} \u2014 MSD')
        ax.legend(loc='lower right')
        fig.tight_layout()

    _save(fig, save_path)
    return fig


# ── publish_pdf ───────────────────────────────────────────────────────────────

def publish_pdf(
    result,
    lag_index: int | None = None,
    palette:   str = 'default',
    figsize:   tuple = (5, 4),
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Publication-quality displacement PDF plot.

    Parameters
    ----------
    result : AnalysisResult
    lag_index : int, optional
        Index into ``result.lags``.  Defaults to ``len(lags) // 4``.
    palette : str, default ``'default'``
    figsize : tuple, default ``(5, 4)``
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors   = _resolve_palette(palette)
    meta     = result.metadata
    obs_unit = meta.get('value_unit', '')
    dname    = result.dataset_name

    lags   = result.lags
    values = result.values

    if lag_index is None:
        lag_index = max(0, len(lags) // 4)
    lag_index = min(lag_index, len(lags) - 1)
    T       = float(lags[lag_index])
    lag_int = max(1, int(round(T)))

    xlabel = f'\u0394x ({obs_unit})' if obs_unit else '\u0394x'

    if lag_int < len(values):
        displacements = values[lag_int:] - values[:-lag_int]
        displacements = displacements[np.isfinite(displacements)]
    else:
        displacements = np.array([])

    with _style_ctx():
        fig, ax = plt.subplots(figsize=figsize)

        if len(displacements) > 1:
            bins = _fd_bins(displacements)
            ax.hist(
                displacements, bins=bins, density=True,
                color=colors['empirical'], alpha=0.5, label='Empirical displacements',
            )

            if result.fit is not None:
                from ..core.models import MODELS
                info   = MODELS.get(result.model, {})
                msd_fn = info.get('msd')
                params = result.fit.params

                if msd_fn is not None:
                    phys_names = info.get('params', [])
                    phys_vals  = [params[n] for n in phys_names if n in params]
                    try:
                        sigma2 = float(msd_fn(T, *phys_vals)) * params.get('N', 1.0)
                        if np.isfinite(sigma2) and sigma2 > 0:
                            dx_range = np.linspace(displacements.min(), displacements.max(), 500)
                            pdf_vals = (
                                np.exp(-dx_range ** 2 / (2.0 * sigma2))
                                / np.sqrt(2.0 * np.pi * sigma2)
                            )
                            ax.plot(
                                dx_range, pdf_vals,
                                color=colors['pdf'], linewidth=1.8,
                                label=f'PDF (T={T:.3f})',
                            )
                    except Exception:
                        pass

            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='#888888')

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability density')
        ax.set_title(f'{dname} \u2014 PDF at lag T={T:.3f}')
        fig.tight_layout()

    _save(fig, save_path)
    return fig


# ── publish_comparison ────────────────────────────────────────────────────────

def publish_comparison(
    cr,
    palette:   str = 'default',
    figsize:   tuple = (7, 4),
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Publication-quality μ comparison bar chart with 95 % CI error bars.

    Parameters
    ----------
    cr : ComparisonResult
    palette : str, default ``'default'``
    figsize : tuple, default ``(7, 4)``
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors = _resolve_palette(palette)
    df = cr.summary_df.dropna(subset=['mu'])

    labels  = list(df['dataset_name'])
    mu_vals = np.array(df['mu'], dtype=float)

    # Parse CI strings like "(1.100, 1.300)" → half-width error bars
    yerr = np.zeros(len(df))
    for i, ci_str in enumerate(df['mu_ci']):
        try:
            lo_str, hi_str = str(ci_str).strip('()').split(',')
            lo, hi = float(lo_str), float(hi_str)
            yerr[i] = (hi - lo) / 2.0
        except Exception:
            yerr[i] = 0.0

    with _style_ctx():
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(labels))
        ax.bar(x, mu_vals, color=colors['theory'], alpha=0.75, zorder=3)
        ax.errorbar(
            x, mu_vals, yerr=yerr,
            fmt='none', color='black', linewidth=1.2, capsize=4, zorder=4,
        )

        # Brownian reference line at μ = 1.0
        ax.axhline(1.0, color='#888888', linewidth=0.8, linestyle='--',
                   label='\u03bc = 1 (Brownian)')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Memory parameter \u03bc')
        ax.set_title('Memory Parameter Comparison')
        ax.legend()
        fig.tight_layout()

    _save(fig, save_path)
    return fig

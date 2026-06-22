"""
analysis/compare.py — Multi-dataset / multi-model comparison utilities.

compare()          Run the same model on several CSV files and collect results.
print_comparison() Print a formatted ASCII comparison table.
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import pandas as pd

from .pipeline import analyze, AnalysisResult


# ── DataFrame row helpers ──────────────────────────────────────────────────────

_DF_COLS = [
    'dataset_name', 'model',
    'mu', 'mu_ci',
    'nu_or_beta', 'nu_or_beta_ci',
    'N', 'r_squared', 'regime',
]

_NAN = float('nan')


def _nan_row(dataset_name: str, model: str) -> dict:
    """Return a row dict with NaN / 'N/A' for a failed dataset."""
    return {
        'dataset_name':    dataset_name,
        'model':           model,
        'mu':              _NAN,
        'mu_ci':           'N/A',
        'nu_or_beta':      _NAN,
        'nu_or_beta_ci':   'N/A',
        'N':               _NAN,
        'r_squared':       _NAN,
        'regime':          'N/A',
    }


def _fmt_ci(ci: tuple) -> str:
    return f'({ci[0]:.3f}, {ci[1]:.3f})'


def _result_row(ar: AnalysisResult) -> dict:
    """Build a summary-DataFrame row from a successful AnalysisResult."""
    row: dict = {
        'dataset_name': ar.dataset_name,
        'model':        ar.model,
        'regime':       ar.regime,
    }

    if ar.fit is None:
        row.update({
            'mu':            _NAN,
            'mu_ci':         'N/A',
            'nu_or_beta':    _NAN,
            'nu_or_beta_ci': 'N/A',
            'N':             _NAN,
            'r_squared':     _NAN,
        })
        return row

    params = ar.fit.params
    cis    = ar.fit.confidence_intervals

    # Primary parameter: first of mu / H / a
    for key in ('mu', 'H', 'a'):
        if key in params:
            row['mu']    = params[key]
            row['mu_ci'] = _fmt_ci(cis[key]) if key in cis else 'N/A'
            break
    else:
        row['mu']    = _NAN
        row['mu_ci'] = 'N/A'

    # Secondary parameter: first of nu / beta / b
    for key in ('nu', 'beta', 'b'):
        if key in params:
            row['nu_or_beta']    = params[key]
            row['nu_or_beta_ci'] = _fmt_ci(cis[key]) if key in cis else 'N/A'
            break
    else:
        row['nu_or_beta']    = _NAN
        row['nu_or_beta_ci'] = 'N/A'

    row['N']         = params.get('N', _NAN)
    row['r_squared'] = ar.fit.r_squared

    return row


# ── ComparisonResult dataclass ────────────────────────────────────────────────

class ComparisonResult:
    """
    Results from a multi-dataset or multi-model comparison.

    Attributes
    ----------
    results : list[AnalysisResult]
        Successful AnalysisResult objects only (failed paths are excluded).
    models_used : list[str]
        Distinct model names used across all runs.
    summary_df : pd.DataFrame
        One row per path/model combination (including failures as NaN rows).
        Columns: dataset_name, model, mu, mu_ci, nu_or_beta, nu_or_beta_ci,
        N, r_squared, regime.
    """

    def __init__(
        self,
        results:    list,
        models_used: list[str],
        summary_df: pd.DataFrame,
    ) -> None:
        self.results     = results
        self.models_used = models_used
        self.summary_df  = summary_df


# ── compare ───────────────────────────────────────────────────────────────────

def compare(
    paths: list[str],
    model: str = 'cosine',
    detrend_method: str | None = 'linear',
    normalize: bool = False,
    max_lag_fraction: float = 1.0,
    fit_kwargs: dict | None = None,
) -> ComparisonResult:
    """
    Run the same SWNA model on multiple CSV files and collect results.

    Parameters
    ----------
    paths : list[str]
        Paths to whitenoise-format CSV files.
    model : str, default ``'cosine'``
        SWNA model name passed to :func:`~whitenoise.analysis.pipeline.analyze`.
    detrend_method : str or None, default ``'linear'``
        Detrending method (see :func:`~whitenoise.utils.preprocess.detrend`).
    normalize : bool, default ``False``
        Whether to z-score normalize values before fitting.
    max_lag_fraction : float, default 1.0
        Fraction of lags used in fitting. The MSD is already computed over
        lags 1 to N/2, so 1.0 fits all displayed MSD points.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :func:`~whitenoise.core.fitting.fit_msd`.

    Returns
    -------
    ComparisonResult
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    n = len(paths)
    print(f'\U0001f504 Comparing {n} dataset{"s" if n != 1 else ""}...')

    results: list[AnalysisResult] = []
    rows: list[dict] = []

    for i, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ar = analyze(
                    path,
                    model=model,
                    detrend_method=detrend_method,
                    normalize=normalize,
                    max_lag_fraction=max_lag_fraction,
                    fit_kwargs=fit_kwargs,
                )
            results.append(ar)
            row = _result_row(ar)
            r2_str = f"{ar.fit.r_squared:.4f}" if ar.fit is not None else 'N/A'
            print(f'  [{i + 1}/{n}] {name} \u2192 R\u00b2={r2_str}, regime={ar.regime}')
        except Exception as exc:
            print(f'  [{i + 1}/{n}] {name} \u2192 ERROR: {exc}')
            row = _nan_row(name, model)

        rows.append(row)

    summary_df = pd.DataFrame(rows, columns=_DF_COLS)
    print('\u2705 Comparison complete.')

    return ComparisonResult(
        results=results,
        models_used=[model],
        summary_df=summary_df,
    )


# ── print_comparison ──────────────────────────────────────────────────────────

def _trunc(s: str, maxlen: int = 16) -> str:
    """Truncate string to maxlen chars; append '...' if truncated."""
    if len(s) <= maxlen:
        return s
    return s[:maxlen - 3] + '...'


def print_comparison(cr: ComparisonResult) -> None:
    """
    Print a formatted ASCII table from a :class:`ComparisonResult`.

    Dataset names longer than 16 characters are truncated with ``...``.

    Parameters
    ----------
    cr : ComparisonResult
    """
    # Column headers and widths
    cols = [
        ('Dataset',  16),
        ('Model',     9),
        ('mu/H/a',    8),
        ('nu/beta',   9),
        ('R\u00b2',           8),
        ('Regime',   14),
    ]

    sep   = '+' + '+'.join('-' * (w + 2) for _, w in cols) + '+'
    hdr   = '|' + '|'.join(f' {h:<{w}} ' for h, w in cols) + '|'

    print(sep)
    print(hdr)
    print(sep)

    df = cr.summary_df
    for _, row in df.iterrows():
        name     = _trunc(str(row['dataset_name']), 16)
        model    = str(row['model'])
        mu_val   = row['mu']
        nb_val   = row['nu_or_beta']
        r2_val   = row['r_squared']
        regime   = str(row['regime'])

        mu_str = f'{mu_val:.4f}' if pd.notna(mu_val) else 'N/A'
        nb_str = f'{nb_val:.4f}' if pd.notna(nb_val) else 'N/A'
        r2_str = f'{r2_val:.4f}' if pd.notna(r2_val) else 'N/A'

        cells = [
            (name,    16),
            (model,    9),
            (mu_str,   8),
            (nb_str,   9),
            (r2_str,   8),
            (regime,  14),
        ]
        line = '|' + '|'.join(f' {v:<{w}} ' for v, w in cells) + '|'
        print(line)

    print(sep)

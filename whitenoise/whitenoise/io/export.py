"""
io/export.py — Export analysis results to CSV files.

export_csv(result, path)    Save lags + empirical/fitted MSD for an AnalysisResult.
export_summary(cr, path)    Save ComparisonResult.summary_df to CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def export_csv(result, path: str) -> None:
    """
    Save lag, empirical MSD, and fitted MSD from an :class:`AnalysisResult`.

    The output is a 3-column CSV::

        lag,msd_empirical,msd_fitted
        1.0,0.12345,0.11980
        2.0,0.24100,0.23750
        ...

    The ``msd_fitted`` column contains the theoretical curve evaluated at each
    lag.  Values outside the fitting window (set by *max_lag_fraction*) and rows
    where the model returns non-finite values are written as ``NaN``.

    Parameters
    ----------
    result : AnalysisResult
        Output of :func:`~whitenoise.analysis.pipeline.analyze`.
    path : str
        Destination file path (e.g. ``'results/msd.csv'``).
    """
    lags        = np.asarray(result.lags, dtype=float)
    msd_emp     = np.asarray(result.msd_empirical, dtype=float)
    n_lags      = len(lags)

    # Build fitted-MSD column aligned to the full lags array
    msd_fitted_full = np.full(n_lags, np.nan, dtype=float)
    if result.fit is not None:
        lags_used  = np.asarray(result.fit.lags_used, dtype=float)
        msd_fitted = np.asarray(result.fit.msd_fitted, dtype=float)
        n_use = len(lags_used)
        # lags_used is always lags[:n_use] (a prefix), so align directly
        n_copy = min(n_use, n_lags)
        msd_fitted_full[:n_copy] = msd_fitted[:n_copy]

    df = pd.DataFrame({
        'lag':          lags,
        'msd_empirical': msd_emp,
        'msd_fitted':    msd_fitted_full,
    })
    df.to_csv(path, index=False)
    print(f'\U0001f4be Saved to {path}')


def export_summary(cr, path: str) -> None:
    """
    Save the comparison summary table from a :class:`ComparisonResult` to CSV.

    Writes ``cr.summary_df`` using :meth:`pandas.DataFrame.to_csv` (no index).

    Parameters
    ----------
    cr : ComparisonResult
        Output of :func:`~whitenoise.analysis.compare.compare` or
        :func:`~whitenoise.analysis.batch.batch_model_search`.
    path : str
        Destination file path.
    """
    cr.summary_df.to_csv(path, index=False)
    print(f'\U0001f4be Saved to {path}')

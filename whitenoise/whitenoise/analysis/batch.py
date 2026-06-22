"""
analysis/batch.py — Batch and model-search utilities.

batch_analyze()      Run one model on many files, optionally in parallel.
batch_model_search() Run many models on one file; report the best fit.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

from .pipeline import analyze, AnalysisResult
from .compare import ComparisonResult, _result_row, _nan_row, _DF_COLS

import pandas as pd

# Path to the package root so child processes can find the package.
_PKG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# ── Module-level worker (must be picklable for ProcessPoolExecutor) ────────────

def _analyze_job(args: tuple):
    """
    Worker function for parallel batch execution.

    Must be defined at module level so multiprocessing can pickle it.
    """
    path, model, detrend_method, normalize, max_lag_fraction, fit_kwargs = args
    # Ensure the package is importable in the child process
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return analyze(
                path,
                model=model,
                detrend_method=detrend_method,
                normalize=normalize,
                max_lag_fraction=max_lag_fraction,
                fit_kwargs=fit_kwargs or {},
            )
    except Exception:
        return None


# ── batch_analyze ──────────────────────────────────────────────────────────────

def batch_analyze(
    paths: list[str],
    model: str = 'cosine',
    detrend_method: str | None = 'linear',
    normalize: bool = False,
    max_lag_fraction: float = 1.0,
    fit_kwargs: dict | None = None,
    n_jobs: int = 1,
) -> list[AnalysisResult | None]:
    """
    Run the same SWNA model on many files, optionally in parallel.

    Parameters
    ----------
    paths : list[str]
        Paths to whitenoise-format CSV files.
    model : str, default ``'cosine'``
        SWNA model name.
    detrend_method : str or None, default ``'linear'``
        Detrending method.
    normalize : bool, default ``False``
        Whether to z-score normalize before fitting.
    max_lag_fraction : float, default 1.0
        Fraction of lags used in fitting. The MSD is already computed over
        lags 1 to N/2, so 1.0 fits all displayed MSD points.
    fit_kwargs : dict, optional
        Extra keyword arguments for :func:`~whitenoise.core.fitting.fit_msd`.
    n_jobs : int, default 1
        Number of parallel workers.  ``1`` = serial.  ``>1`` = parallel
        (uses :class:`concurrent.futures.ThreadPoolExecutor`).

    Returns
    -------
    list[AnalysisResult or None]
        One entry per path; ``None`` for failed analyses.
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    n = len(paths)
    print(f'\U0001f680 Batch analyzing {n} dataset{"s" if n != 1 else ""} '
          f'(n_jobs={n_jobs})...')

    job_args = [
        (path, model, detrend_method, normalize, max_lag_fraction, fit_kwargs)
        for path in paths
    ]

    results: list[AnalysisResult | None]

    if n_jobs == 1:
        results = [_analyze_job(args) for args in job_args]
    else:
        # ThreadPoolExecutor avoids Windows spawn/pickle issues with non-installed
        # packages while still providing genuine concurrency for I/O-bound work.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_analyze_job, job_args))

    succeeded = 0
    for path, ar in zip(paths, results):
        name = os.path.splitext(os.path.basename(path))[0]
        if ar is not None:
            r2_str = (
                f'{ar.fit.r_squared:.4f}' if ar.fit is not None else 'N/A'
            )
            print(f'  \u2713 {name}  (R\u00b2={r2_str})')
            succeeded += 1
        else:
            print(f'  \u2717 {name}  (failed)')

    print(f'\U0001f3c1 Batch complete. {succeeded}/{n} succeeded.')
    return results


# ── batch_model_search ────────────────────────────────────────────────────────

def batch_model_search(
    path: str,
    models: list[str] | None = None,
    detrend_method: str | None = 'linear',
    normalize: bool = False,
    max_lag_fraction: float = 1.0,
    fit_kwargs: dict | None = None,
) -> ComparisonResult:
    """
    Fit every available model to a single CSV and report the best.

    Stub models (``status != 'available'``) and models that raise
    :exc:`ValueError` or :exc:`NotImplementedError` are silently skipped.

    Parameters
    ----------
    path : str
        Path to a whitenoise-format CSV file.
    models : list[str] or None
        Models to try.  ``None`` → all available models from the MODELS registry.
    detrend_method : str or None, default ``'linear'``
        Detrending method.
    normalize : bool, default ``False``
        Whether to z-score normalize before fitting.
    max_lag_fraction : float, default 1.0
        Fraction of lags used in fitting. The MSD is already computed over
        lags 1 to N/2, so 1.0 fits all displayed MSD points.
    fit_kwargs : dict, optional
        Extra keyword arguments for :func:`~whitenoise.core.fitting.fit_msd`.

    Returns
    -------
    ComparisonResult
        One row per successfully tried model.
    """
    from ..core.models import MODELS

    if fit_kwargs is None:
        fit_kwargs = {}

    if models is None:
        models = [n for n, v in MODELS.items() if v['status'] == 'available']

    dataset_name = os.path.splitext(os.path.basename(path))[0]
    print(f'\U0001f50d Model search on \'{dataset_name}\' across {len(models)} model(s)...')

    results: list[AnalysisResult] = []
    rows: list[dict] = []

    for m in models:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ar = analyze(
                    path,
                    model=m,
                    detrend_method=detrend_method,
                    normalize=normalize,
                    max_lag_fraction=max_lag_fraction,
                    fit_kwargs=fit_kwargs,
                )
            r2 = ar.fit.r_squared if ar.fit is not None else float('nan')
            r2_str = f'{r2:.4f}' if ar.fit is not None else 'N/A'
            print(f'  {m} \u2192 R\u00b2={r2_str}')
            results.append(ar)
            rows.append(_result_row(ar))
        except (ValueError, NotImplementedError):
            # Silently skip stub / invalid models
            pass
        except Exception as exc:
            print(f'  {m} \u2192 ERROR: {exc}')
            rows.append(_nan_row(dataset_name, m))

    summary_df = pd.DataFrame(rows, columns=_DF_COLS)

    # Report best model
    valid = summary_df.dropna(subset=['r_squared'])
    if not valid.empty:
        best_idx  = valid['r_squared'].idxmax()
        best_name = valid.loc[best_idx, 'model']
        best_r2   = valid.loc[best_idx, 'r_squared']
        print(f'  Best model: {best_name}  (R\u00b2={best_r2:.4f})')

    return ComparisonResult(
        results=results,
        models_used=list(summary_df['model']) if not summary_df.empty else [],
        summary_df=summary_df,
    )

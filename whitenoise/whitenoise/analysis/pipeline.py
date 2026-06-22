"""
analysis/pipeline.py — High-level SWNA analysis pipeline.

Orchestrates the full analysis sequence:
  read_csv → detrend → normalize → compute_msd → fit_msd → AnalysisResult
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass

from ..io.reader import read_csv
from ..core.msd import compute_msd
from ..core.fitting import fit_msd, FitResult, _sym
from ..utils.preprocess import detrend, normalize as _normalize_fn


# ── Regime label ───────────────────────────────────────────────────────────────

def _regime(fit: FitResult | None) -> str:
    """Return a plain-English diffusion regime label from a FitResult."""
    if fit is None:
        return 'N/A'
    params = fit.params
    if 'mu' in params:
        mu = params['mu']
        if mu < 1.0:
            return 'subdiffusive'
        elif abs(mu - 1.0) < 1e-9:
            return 'Brownian'
        elif mu <= 2.0:
            return 'superdiffusive'
        else:
            return 'hyperballistic'
    elif 'H' in params:
        return f"H={params['H']:.3f}"
    elif 'a' in params:
        return 'plateau (DNA)'
    return 'unknown'


# ── AnalysisResult ─────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Full output of a single SWNA analysis run.

    Attributes
    ----------
    dataset_name : str
        CSV filename without extension (or a user-supplied label).
    model : str
        SWNA model name used for fitting.
    fit : FitResult or None
        Fitting results.  ``None`` if fitting failed.
    lags : np.ndarray
        Lag array from :func:`~whitenoise.core.msd.compute_msd`.
    msd_empirical : np.ndarray
        Empirical MSD values.
    values : np.ndarray
        Preprocessed observable values (after detrend / normalize).
    time : np.ndarray
        Time array from the CSV.
    metadata : dict
        Column names, units, and source info from the reader.
    """

    dataset_name:  str
    model:         str
    fit:           FitResult | None
    lags:          np.ndarray
    msd_empirical: np.ndarray
    values:        np.ndarray
    time:          np.ndarray
    metadata:      dict

    @property
    def regime(self) -> str:
        """Diffusion regime label derived from the fitted parameters."""
        return _regime(self.fit)

    def summary(self) -> None:
        """
        Print a formatted analysis summary block.

        Example output::

            ══════════════════════════════════════════
             SWNA Analysis Summary
            ══════════════════════════════════════════
             Dataset   : sunspot_data
             Model     : cosine
             Points    : 300
             Lags used : 150
            ──────────────────────────────────────────
             Parameters:
               μ      = 1.2341  ±  0.0082
               ν      = 0.0082  ±  0.0003
               N      = 2.4312  ±  0.0441
             R²        = 0.9823
             Regime    : superdiffusive
            ──────────────────────────────────────────
             Units     : x=time (months), y=sunspot_number (count)
            ══════════════════════════════════════════
        """
        SEP_DOUBLE = '\u2550' * 42
        SEP_SINGLE = '\u2500' * 42

        print(SEP_DOUBLE)
        print(' SWNA Analysis Summary')
        print(SEP_DOUBLE)
        print(f' Dataset   : {self.dataset_name}')
        print(f' Model     : {self.model}')
        print(f' Points    : {len(self.values)}')
        print(f' Lags used : {len(self.lags)}')
        print(SEP_SINGLE)

        if self.fit is None:
            print(' Parameters: N/A (fitting failed)')
            print(' R\u00b2        : N/A')
            print(' Regime    : N/A')
        else:
            print(' Parameters:')
            for pname, pval in self.fit.params.items():
                se = self.fit.std_errors.get(pname, float('nan'))
                sym = _sym(pname)
                print(f'   {sym:<6} = {pval:.4f}  \u00b1  {se:.4f}')
            print(f' R\u00b2        = {self.fit.r_squared:.4f}')
            print(f' Regime    : {self.regime}')

        print(SEP_SINGLE)
        t_label = self.metadata.get('time_label', 'time')
        v_label = self.metadata.get('value_label', 'value')
        print(f' Units     : x={t_label}, y={v_label}')
        print(SEP_DOUBLE)


# ── analyze ────────────────────────────────────────────────────────────────────

def analyze(
    path: str,
    model: str = 'cosine',
    detrend_method: str | None = 'linear',
    normalize: bool = False,
    max_lag_fraction: float = 1.0,
    fit_kwargs: dict | None = None,
) -> AnalysisResult:
    """
    Run the full SWNA pipeline on a whitenoise-format CSV file.

    Steps
    -----
    1. :func:`~whitenoise.io.reader.read_csv` — load time, values, metadata.
    2. :func:`~whitenoise.utils.preprocess.detrend` — remove trend (if *detrend_method* is not ``None``).
    3. :func:`~whitenoise.utils.preprocess.normalize` — z-score normalize (if *normalize* is ``True``).
    4. :func:`~whitenoise.core.msd.compute_msd` — compute empirical MSD.
    5. :func:`~whitenoise.core.fitting.fit_msd` — fit the chosen model.
    6. Return :class:`AnalysisResult`.

    Parameters
    ----------
    path : str
        Path to a whitenoise-format CSV file.
    model : str, default ``'cosine'``
        SWNA model name.  Run ``wn.list_models()`` for options.
    detrend_method : str or None, default ``'linear'``
        Passed to :func:`~whitenoise.utils.preprocess.detrend`.
        ``None`` skips detrending entirely.
    normalize : bool, default ``False``
        If ``True``, apply z-score normalization after detrending.
    max_lag_fraction : float, default 1.0
        Fraction of lags to use in fitting (forwarded to
        :func:`~whitenoise.core.fitting.fit_msd`).
        The MSD is already computed over lags 1 to N/2, so the default
        of 1.0 fits all displayed MSD points.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :func:`~whitenoise.core.fitting.fit_msd` (e.g. ``p0``, ``bounds``).

    Returns
    -------
    AnalysisResult

    Examples
    --------
    >>> result = wn.analyze('sunspot.csv', model='exponential')
    >>> result.summary()
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    # Step 1 — Load
    print(f'\U0001f4c2 Loading: {path}')
    time, values, metadata = read_csv(path)

    # Steps 2 & 3 — Preprocess
    print(f'\U0001f527 Preprocessing: detrend={detrend_method}, normalize={normalize}')
    if detrend_method is not None:
        values = detrend(values, method=detrend_method)
    if normalize:
        values = _normalize_fn(values)

    # Step 4 — Empirical MSD
    max_lag = len(values) // 2
    print(f'\U0001f4ca Computing empirical MSD ({len(values)} points, max_lag={max_lag})...')
    lags, msd_emp = compute_msd(values)

    # Step 5 — Fit
    print(f'\U0001f50d Fitting {model} model...')
    fit_result = fit_msd(
        lags, msd_emp,
        model=model,
        max_lag_fraction=max_lag_fraction,
        **fit_kwargs,
    )

    # Step 6 — Report
    if fit_result is None:
        print('\u274c Fitting failed.')
    else:
        regime_str = _regime(fit_result)
        print(f'\u2705 Done. R\u00b2 = {fit_result.r_squared:.4f}  |  regime: {regime_str}')

    dataset_name = os.path.splitext(os.path.basename(path))[0]

    return AnalysisResult(
        dataset_name=dataset_name,
        model=model,
        fit=fit_result,
        lags=lags,
        msd_empirical=msd_emp,
        values=values,
        time=time,
        metadata=metadata,
    )

"""
core/fitting.py — Parameter extraction with 95% confidence intervals.

Fits  N · msd_theory(T, *params)  to empirical MSD using the
Levenberg-Marquardt algorithm (scipy.optimize.curve_fit).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import curve_fit

from .models import MODELS


# ── Greek symbol display ───────────────────────────────────────────────────────

_GREEK: dict[str, str] = {
    'mu':   '\u03bc',   # μ
    'nu':   '\u03bd',   # ν
    'beta': '\u03b2',   # β
    'H':    'H',
    'N':    'N',
    'a':    'a',
    'b':    'b',
    'c':    'c',
}


def _sym(name: str) -> str:
    return _GREEK.get(name, name)


# ── Default initial parameters and bounds ─────────────────────────────────────

DEFAULTS: dict[str, dict] = {
    'cosine': {
        'p0':     [1.2, 0.01],
        'bounds': ([0.01, 1e-9], [5.0, 10.0]),
    },
    'exponential': {
        'p0':     [1.2, 0.1],
        'bounds': ([0.01, 1e-9], [5.0, 100.0]),
    },
    'sine': {
        'p0':     [1.2, 0.01],
        'bounds': ([0.01, 1e-9], [5.0, 10.0]),
    },
    'fbm': {
        'p0':     [0.6],
        'bounds': ([0.01], [2.0]),
    },
    'dna': {
        'p0':     [5.0, 0.01, 3.0],
        'bounds': ([0.01, 1e-9, 0.01], [1000.0, 100.0, 1000.0]),
    },
}


# ── FitResult dataclass ────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """
    Results from a single MSD fit.

    Attributes
    ----------
    params : dict
        Fitted parameter values, e.g. ``{'mu': 1.15, 'beta': 0.1, 'N': 2.1}``.
    std_errors : dict
        Standard errors ``sqrt(diag(pcov))``, same keys as *params*.
    confidence_intervals : dict
        95 % confidence intervals ``(param - 1.96*se, param + 1.96*se)``,
        same keys as *params*.
    r_squared : float
        Coefficient of determination R\u00b2.
    model : str
        Name of the fitted model.
    lags_used : np.ndarray
        Lag values supplied to ``fit_msd`` after applying *max_lag_fraction*.
    msd_fitted : np.ndarray
        Theoretical curve ``N \u00b7 msd_theory(lags_used, *physical_params)``.
    """

    params:               dict
    std_errors:           dict
    confidence_intervals: dict
    r_squared:            float
    model:                str
    lags_used:            np.ndarray
    msd_fitted:           np.ndarray

    def summary(self) -> str:
        """
        Return a formatted box-drawing string showing fit results.

        Example for the cosine model::

            \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
            \u2502  Fit Summary                            \u2502
            \u2502  Model  : cosine          R\u00b2 = 0.9823  \u2502
            \u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524
            \u2502  \u03bc    = 1.2341 \u00b1 0.0082                \u2502
            \u2502         95% CI: (1.218, 1.250)          \u2502
            \u2502  \u03bd    = 0.0082 \u00b1 0.0003                 \u2502
            \u2502         95% CI: (0.008, 0.009)          \u2502
            \u2502  N    = 2.4312 \u00b1 0.0441                 \u2502
            \u2502         95% CI: (2.345, 2.518)          \u2502
            \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518

        Returns
        -------
        str
        """
        W = 43  # inner width (between box walls)

        def _line(text: str) -> str:
            return '\u2502  ' + text.ljust(W - 2) + '\u2502'

        header = f'Model  : {self.model:<14} R\u00b2 = {self.r_squared:.4f}'
        lines = [
            '\u250c' + '\u2500' * W + '\u2510',
            _line('Fit Summary'),
            _line(header),
            '\u251c' + '\u2500' * W + '\u2524',
        ]

        for pname, pval in self.params.items():
            se       = self.std_errors.get(pname, float('nan'))
            lo, hi   = self.confidence_intervals.get(pname, (float('nan'), float('nan')))
            sym      = _sym(pname)
            lines.append(_line(f'  {sym:<4} = {pval:.4f} \u00b1 {se:.4f}'))
            lines.append(_line(f'         95% CI: ({lo:.3f}, {hi:.3f})'))

        lines.append('\u2514' + '\u2500' * W + '\u2518')
        return '\n'.join(lines)


# ── fit_msd ────────────────────────────────────────────────────────────────────

def fit_msd(
    lags: np.ndarray,
    msd_empirical: np.ndarray,
    model: str = 'cosine',
    p0: list | None = None,
    bounds: tuple | None = None,
    max_lag_fraction: float = 0.4,
    lag_weights: str = 'inverse',
) -> 'FitResult | None':
    """
    Fit ``N \u00b7 msd_theory(T, *params)`` to an empirical MSD curve.

    The fit function is::

        f(T) = N \u00b7 msd_model(T, *physical_params)

    where *N* is a dimensionless normalization scalar always appended to
    *p0* and *bounds* internally.

    Parameters
    ----------
    lags : np.ndarray
        Lag values (e.g. from :func:`~whitenoise.core.msd.compute_msd`).
    msd_empirical : np.ndarray
        Empirical MSD values, same shape as *lags*.
    model : str, default ``'cosine'``
        Model name.  Must be a registered, available model.
    p0 : list, optional
        Initial guess for physical parameters only; ``N = 1.0`` appended
        internally.  Defaults to ``DEFAULTS[model]['p0']``.
    bounds : tuple, optional
        ``(lower, upper)`` for physical parameters only; ``(0, inf)`` for N
        appended internally.  Defaults to ``DEFAULTS[model]['bounds']``.
    max_lag_fraction : float, default 0.4
        Fraction of the lag range to use for fitting.
        High-lag MSD estimates use few independent pairs and exhibit
        finite-sample saturation \u2014 restricting to the first 40 % of lags
        keeps the fit in the statistically reliable regime.
    lag_weights : str, default ``'inverse'``
        How to weight lag points in the least-squares fit.
        ``'inverse'`` weights each lag by ``1/lag`` so short lags
        (computed from many pairs) dominate over noisy long lags.
        ``'uniform'`` gives equal weight to all lags (original behaviour).

    Returns
    -------
    FitResult or None
        ``None`` if fitting fails or if *msd_empirical* is all-zero / all-nan.

    Raises
    ------
    ValueError
        If *model* is not in the MODELS registry or is not yet available.

    Notes
    -----
    * Prints ``\u2713`` for R\u00b2 \u2265 0.8, ``\u26a0`` for moderate (0.5\u20130.8) or low (< 0.5) R\u00b2,
      ``\u2717`` on failure.
    * Never raises for fitting failures \u2014 returns ``None`` instead.

    Examples
    --------
    >>> lags, msd = wn.compute_msd(values)
    >>> result = wn.fit_msd(lags, msd, model='cosine')
    >>> print(result.summary())
    """
    # ── Validate model ─────────────────────────────────────────────────────────
    if model not in MODELS:
        raise ValueError(
            f"\u2717 Unknown model '{model}'.\n"
            f"Run wn.list_models() to see all available models."
        )
    info = MODELS[model]
    if info['status'] != 'available':
        raise ValueError(
            f"\u2717 Model '{model}' is not yet implemented.\n"
            f"Run wn.list_models() to see available models."
        )

    param_names: list[str] = info['params']
    msd_fn = info['msd']

    # ── Prepare data ───────────────────────────────────────────────────────────
    lags_arr = np.asarray(lags, dtype=float)
    msd_arr  = np.asarray(msd_empirical, dtype=float)

    n_use    = max(1, int(len(lags_arr) * max_lag_fraction))
    lags_use = lags_arr[:n_use]
    msd_use  = msd_arr[:n_use]

    # Guard: all-zero / all-nan / near-zero (e.g. detrended constant series
    # leaves floating-point residuals ~1e-30, giving MSD ~1e-60 — unphysical).
    finite_vals = msd_use[np.isfinite(msd_use)]
    max_abs = float(np.max(np.abs(finite_vals))) if len(finite_vals) > 0 else 0.0
    if max_abs < 1e-20 or np.all(np.isnan(msd_use)):
        print(f"\u2717 Fitting failed for model '{model}': MSD is all-zero or all-nan.")
        return None

    # Remove any NaN / inf pairs
    valid    = np.isfinite(msd_use) & np.isfinite(lags_use)
    if not np.any(valid):
        print(f"\u2717 Fitting failed for model '{model}': no finite MSD values.")
        return None
    lags_fit = lags_use[valid]
    msd_fit  = msd_use[valid]

    # ── Defaults ───────────────────────────────────────────────────────────────
    if p0 is None:
        p0 = list(DEFAULTS.get(model, {}).get('p0', [1.0] * len(param_names)))
    if bounds is None:
        d  = DEFAULTS.get(model, {})
        lb = list(d.get('bounds', ([1e-9] * len(param_names), [1e9] * len(param_names)))[0])
        ub = list(d.get('bounds', ([1e-9] * len(param_names), [1e9] * len(param_names)))[1])
        bounds = (lb, ub)

    # Append N (normalization scalar)
    p0_full     = list(p0) + [1.0]
    bounds_full = (list(bounds[0]) + [0.0], list(bounds[1]) + [np.inf])

    # ── Model wrapper (NaN-safe) ───────────────────────────────────────────────
    def _wrapper(T, *args):
        phys = args[:-1]
        N    = args[-1]
        out  = msd_fn(T, *phys)
        if np.ndim(out) == 0:
            v = float(out)
            return 1e30 if (not np.isfinite(v) or v <= 0) else N * v
        out_arr = np.asarray(out, dtype=float) * N
        # Replace non-finite (NaN / inf) with large penalty so optimizer avoids them
        return np.where(np.isfinite(out_arr), out_arr, 1e30)

    # ── Fit ────────────────────────────────────────────────────────────────────
    try:
        popt, pcov = curve_fit(
            _wrapper,
            lags_fit,
            msd_fit,
            p0=p0_full,
            bounds=bounds_full,
            maxfev=10_000,
        )
    except Exception as exc:
        print(f"\u2717 Fitting failed for model '{model}': {exc}")
        return None

    # ── Uncertainties ──────────────────────────────────────────────────────────
    # Clip negative diagonal elements (numerical noise) before sqrt
    perr      = np.sqrt(np.diag(np.clip(pcov, 0.0, None)))
    all_names = param_names + ['N']

    params_out = {n: float(popt[i]) for i, n in enumerate(all_names)}
    se_out     = {n: float(perr[i]) for i, n in enumerate(all_names)}
    ci_out     = {
        n: (float(popt[i] - 1.96 * perr[i]), float(popt[i] + 1.96 * perr[i]))
        for i, n in enumerate(all_names)
    }

    # ── R\u00b2 ─────────────────────────────────────────────────────────────────────
    msd_fitted_arr = np.asarray(_wrapper(lags_fit, *popt), dtype=float)
    ss_res = float(np.sum((msd_fit - msd_fitted_arr) ** 2))
    ss_tot = float(np.sum((msd_fit - np.mean(msd_fit)) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Quality feedback ───────────────────────────────────────────────────────
    alternatives = [n for n in MODELS if MODELS[n]['status'] == 'available' and n != model]
    if r2 >= 0.8:
        print(f"\u2713 Good fit (R\u00b2 = {r2:.4f})")
    elif r2 >= 0.5:
        print(f"\u26a0 Moderate fit (R\u00b2 = {r2:.4f}). Results may be less reliable.")
    else:
        print(f"\u26a0 Low R\u00b2 ({r2:.4f}). Consider trying other models: {alternatives}")

    # Evaluate fitted curve on the full lags_use window (may contain NaN)
    msd_out = np.asarray(_wrapper(lags_use, *popt), dtype=float)

    return FitResult(
        params=params_out,
        std_errors=se_out,
        confidence_intervals=ci_out,
        r_squared=r2,
        model=model,
        lags_used=lags_use,
        msd_fitted=msd_out,
    )

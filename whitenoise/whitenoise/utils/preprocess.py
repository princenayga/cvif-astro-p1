"""
utils/preprocess.py — Optional preprocessing helpers for whitenoise.

These functions are called manually by the researcher BEFORE analyze().
The pipeline never invokes them automatically.

Typical workflow::

    time, values, meta = wn.read_csv('co2.csv')
    fluct = wn.detrend(values, method='polynomial', poly_order=2)
    result = wn.analyze(fluct, model='cosine', label='CO2 fluctuations')
"""

from __future__ import annotations

import numpy as np


# ── Internal helper ───────────────────────────────────────────────────────────

def _as_1d(values) -> np.ndarray:
    """Convert array-like to 1-D float ndarray."""
    return np.asarray(values, dtype=float).ravel()


# ── Public API ────────────────────────────────────────────────────────────────

def detrend(
    values,
    method: str = 'linear',
    poly_order: int = 1,
) -> np.ndarray:
    """
    Remove a trend from a time series to extract fluctuations.

    The trend is estimated by fitting a polynomial to the data (indexed
    ``0, 1, …, N-1``) and subtracting it.

    Parameters
    ----------
    values : array-like (1D)
        Input time series.
    method : str, default ``'linear'``
        Detrending method:

        * ``'linear'``      — subtract a degree-1 (straight-line) fit.
        * ``'polynomial'``  — subtract a degree-``poly_order`` polynomial fit.
        * ``'mean'``        — subtract the global mean only.

    poly_order : int, default 1
        Polynomial degree.  Only used when ``method='polynomial'``.

    Returns
    -------
    np.ndarray
        Detrended fluctuations, same length as ``values``.

    Raises
    ------
    ValueError
        ``'✗ Unknown detrend method …'`` for unrecognised ``method``.

    Examples
    --------
    >>> time, values, meta = wn.read_csv('co2.csv')
    >>> fluct = wn.detrend(values, method='polynomial', poly_order=2)
    >>> result = wn.analyze(fluct, model='cosine', label='CO2')
    """
    arr = _as_1d(values)
    idx = np.arange(len(arr), dtype=float)

    if method == 'linear':
        coeffs = np.polyfit(idx, arr, 1)
        return arr - np.polyval(coeffs, idx)
    elif method == 'polynomial':
        coeffs = np.polyfit(idx, arr, int(poly_order))
        return arr - np.polyval(coeffs, idx)
    elif method == 'mean':
        return arr - np.mean(arr)
    else:
        raise ValueError(
            f"✗ Unknown detrend method '{method}'. "
            f"Choose: 'linear', 'polynomial', 'mean'."
        )


def normalize(
    values,
    method: str = 'zscore',
) -> np.ndarray:
    """
    Normalize a time series.

    Parameters
    ----------
    values : array-like (1D)
        Input time series.
    method : str, default ``'zscore'``
        Normalization method:

        * ``'zscore'`` — subtract mean, divide by standard deviation.
        * ``'minmax'`` — scale to the interval ``[0, 1]``.
        * ``'mean'``   — divide by the mean only (preserves shape).

    Returns
    -------
    np.ndarray
        Normalized series, same length as ``values``.

    Raises
    ------
    ValueError
        ``'✗ Unknown normalize method …'`` for unrecognised ``method``.
    """
    arr = _as_1d(values)

    if method == 'zscore':
        return (arr - np.mean(arr)) / np.std(arr)
    elif method == 'minmax':
        lo, hi = np.min(arr), np.max(arr)
        return (arr - lo) / (hi - lo)
    elif method == 'mean':
        return arr / np.mean(arr)
    else:
        raise ValueError(
            f"✗ Unknown normalize method '{method}'. "
            f"Choose: 'zscore', 'minmax', 'mean'."
        )


def smooth(
    values,
    window: int = 5,
    method: str = 'moving_average',
) -> np.ndarray:
    """
    Smooth a time series.  Output is always the same length as the input.

    Parameters
    ----------
    values : array-like (1D)
        Input time series.
    window : int, default 5
        Number of points in the smoothing kernel.  Must be a positive odd
        integer.  If an even value is given it is incremented to the next
        odd integer and a warning is printed:
        ``"⚠ Window size must be odd. Using {window+1} instead."``
    method : str, default ``'moving_average'``
        Smoothing method:

        * ``'moving_average'`` — uniform (box) kernel via ``np.convolve``.
        * ``'gaussian'``       — Gaussian kernel with
          ``sigma = window / 4`` via ``scipy.ndimage.gaussian_filter1d``.

    Returns
    -------
    np.ndarray
        Smoothed series, same length as ``values``.

    Raises
    ------
    ValueError
        ``'✗ Unknown smooth method …'`` for unrecognised ``method``.
    """
    arr = _as_1d(values)

    window = int(window)
    if window % 2 == 0:
        print(f'⚠ Window size must be odd. Using {window + 1} instead.')
        window += 1

    if method == 'moving_average':
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='same')
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = window / 4.0
        return gaussian_filter1d(arr, sigma=sigma)
    else:
        raise ValueError(
            f"✗ Unknown smooth method '{method}'. "
            f"Choose: 'moving_average', 'gaussian'."
        )

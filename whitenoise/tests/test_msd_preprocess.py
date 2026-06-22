"""
tests/test_msd_preprocess.py — Self-contained tests for:
  whitenoise/core/msd.py
  whitenoise/utils/preprocess.py

Run with:
    python tests/test_msd_preprocess.py

Each test prints "PASS" or "FAIL: {reason}".
Final line: "X/10 tests passed."
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Force UTF-8 output (Windows cp1252 safety)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from whitenoise.core.msd import compute_msd
from whitenoise.utils.preprocess import detrend, normalize, smooth


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(name: str, fn) -> bool:
    try:
        fn()
        print(f'PASS  {name}')
        return True
    except AssertionError as exc:
        print(f'FAIL  {name}: {exc}')
        return False
    except Exception as exc:
        print(f'FAIL  {name}: unexpected {type(exc).__name__}: {exc}')
        return False


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_brownian_msd_linear():
    """Brownian MSD (cumsum of iid Gaussian) grows approximately linearly (R² > 0.85).

    We restrict to the first 10% of lags (max_lag=100) where each MSD estimate
    uses at least 900 samples — the statistically reliable linear regime.
    Large-lag MSD estimates are inherently noisy and not expected to be linear.
    """
    np.random.seed(42)
    x = np.cumsum(np.random.randn(1000))
    lags, msd = compute_msd(x, max_lag=100)

    coeffs = np.polyfit(lags, msd, 1)
    msd_pred = np.polyval(coeffs, lags)
    r2 = _r_squared(msd, msd_pred)

    assert r2 > 0.85, f'R² = {r2:.4f}, expected > 0.85'


def test_02_superdiffusive_msd():
    """Superdiffusive MSD grows faster than linear."""
    np.random.seed(42)
    increments = np.random.randn(500) * np.arange(1, 501) ** 0.4
    x = np.cumsum(increments)
    lags, msd = compute_msd(x)

    midpoint = len(msd) // 2
    ratio_data = msd[-1] / msd[midpoint]
    ratio_linear = lags[-1] / lags[midpoint]

    assert ratio_data > ratio_linear * 1.2, (
        f'ratio_data={ratio_data:.3f}, ratio_linear={ratio_linear:.3f} '
        f'(expected ratio_data > {ratio_linear * 1.2:.3f})'
    )


def test_03_normalize_makes_msd0_equal_one():
    """normalize=True → MSD[0] == 1.0 exactly."""
    np.random.seed(0)
    x = np.cumsum(np.random.randn(200))
    _, msd = compute_msd(x, normalize=True)

    assert abs(msd[0] - 1.0) < 1e-10, f'msd[0] = {msd[0]}'


def test_04_pandas_series_input():
    """pd.Series input returns correct shape and finite values."""
    import pandas as pd
    np.random.seed(1)
    s = pd.Series(np.random.randn(200))
    lags, msd = compute_msd(s)

    assert len(lags) == 100, f'len(lags) = {len(lags)}, expected 100'
    assert all(np.isfinite(msd)), f'Non-finite MSD values found'


def test_05_list_input():
    """list input produces correct lag count."""
    np.random.seed(2)
    x = list(np.random.randn(50))
    lags, msd = compute_msd(x)

    assert len(lags) == 25, f'len(lags) = {len(lags)}, expected 25'


def test_06_2d_array_raises():
    """2D array raises ValueError with ✗ and '1D' in message."""
    x_2d = np.random.randn(10, 5)
    try:
        compute_msd(x_2d)
        assert False, 'Should have raised ValueError'
    except ValueError as exc:
        msg = str(exc)
        assert '✗' in msg, f'No ✗ in message: {msg!r}'
        assert '1D' in msg, f"'1D' not in message: {msg!r}"


def test_07_too_few_points_raises():
    """Fewer than 10 points raises ValueError with ✗ and '10' in message."""
    try:
        compute_msd([1.0, 2.0, 3.0])
        assert False, 'Should have raised ValueError'
    except ValueError as exc:
        msg = str(exc)
        assert '✗' in msg, f'No ✗ in message: {msg!r}'
        assert '10' in msg, f"'10' not in message: {msg!r}"


def test_08_detrend_linear_removes_trend():
    """detrend(method='linear') removes slope; output is same length."""
    np.random.seed(3)
    x = np.linspace(0, 100, 300) + np.random.randn(300) * 0.5
    fluct = detrend(x, method='linear')

    slope = np.polyfit(np.arange(300), fluct, 1)[0]
    assert abs(slope) < 0.01, f'Residual slope = {slope:.6f}, expected < 0.01'
    assert len(fluct) == 300, f'len(fluct) = {len(fluct)}, expected 300'


def test_09_normalize_zscore():
    """normalize('zscore') → mean ≈ 0, std ≈ 1."""
    np.random.seed(4)
    x = np.random.randn(200) * 8 + 50
    n = normalize(x, method='zscore')

    assert abs(np.mean(n)) < 0.01, f'mean = {np.mean(n):.6f}'
    assert abs(np.std(n) - 1.0) < 0.01, f'std = {np.std(n):.6f}'


def test_10_smooth_same_length():
    """Both smooth methods return array of same length as input."""
    np.random.seed(5)
    x = np.random.randn(150)

    s_ma = smooth(x, window=7, method='moving_average')
    s_g  = smooth(x, window=7, method='gaussian')

    assert len(s_ma) == 150, f'moving_average len = {len(s_ma)}'
    assert len(s_g)  == 150, f'gaussian len = {len(s_g)}'


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ('01 Brownian MSD ~ linear (R² > 0.85)',    test_01_brownian_msd_linear),
    ('02 Superdiffusive MSD > linear growth',   test_02_superdiffusive_msd),
    ('03 normalize=True → MSD[0] == 1.0',       test_03_normalize_makes_msd0_equal_one),
    ('04 pd.Series input',                      test_04_pandas_series_input),
    ('05 list input',                           test_05_list_input),
    ('06 2D array raises ValueError',           test_06_2d_array_raises),
    ('07 < 10 points raises ValueError',        test_07_too_few_points_raises),
    ('08 detrend linear removes slope',         test_08_detrend_linear_removes_trend),
    ('09 normalize zscore mean≈0 std≈1',        test_09_normalize_zscore),
    ('10 smooth output same length',            test_10_smooth_same_length),
]

if __name__ == '__main__':
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print(f'\n{passed}/{total} tests passed.')
    sys.exit(0 if passed == total else 1)

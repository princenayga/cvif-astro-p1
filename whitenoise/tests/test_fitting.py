"""
tests/test_fitting.py — Self-contained tests for whitenoise/core/fitting.py

Run with:
    python tests/test_fitting.py

Each test prints "PASS" or "FAIL: {reason}".
Final line: "X/8 tests passed."
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

# Force UTF-8 output (Windows cp1252 safety)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from whitenoise.core.models import msd_cosine, msd_exponential, msd_dna
from whitenoise.core.fitting import fit_msd, FitResult


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


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_cosine_recovers_mu():
    """fit_msd recovers μ from noiseless cosine MSD (|error| < 0.1)."""
    T = np.linspace(1.0, 100.0, 200)
    true_mu, true_nu = 1.3, 0.008
    msd_true = msd_cosine(T, true_mu, true_nu)

    # Silence fit quality print during test
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_msd(T, msd_true, model='cosine')

    assert result is not None, 'fit_msd returned None'
    err = abs(result.params['mu'] - true_mu)
    assert err < 0.1, f'|fitted_mu - true_mu| = {err:.4f}, expected < 0.1'


def test_02_exponential_recovers_mu():
    """fit_msd recovers μ from noiseless exponential MSD (|error| < 0.1)."""
    T = np.linspace(1.0, 100.0, 200)
    true_mu, true_beta = 1.15, 0.1
    msd_true = msd_exponential(T, true_mu, true_beta)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_msd(T, msd_true, model='exponential')

    assert result is not None, 'fit_msd returned None'
    err = abs(result.params['mu'] - true_mu)
    assert err < 0.1, f'|fitted_mu - true_mu| = {err:.4f}, expected < 0.1'


def test_03_confidence_intervals_bracket_estimate():
    """95% CI for each parameter brackets the point estimate (lo <= pval <= hi)."""
    T = np.linspace(1.0, 100.0, 200)
    msd_true = msd_exponential(T, 1.15, 0.1)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_msd(T, msd_true, model='exponential')

    assert result is not None, 'fit_msd returned None'
    assert 'mu' in result.confidence_intervals, "'mu' not in confidence_intervals"
    for pname, pval in result.params.items():
        lo, hi = result.confidence_intervals[pname]
        # lo = pval - 1.96*se, hi = pval + 1.96*se  → always lo <= pval <= hi
        # (equality occurs when se≈0 on noiseless data, which is fine)
        assert lo <= pval <= hi, (
            f"CI for '{pname}': ({lo:.6f}, {hi:.6f}) does not bracket {pval:.6f}"
        )


def test_04_summary_returns_non_empty_string():
    """FitResult.summary() returns a non-empty string without error."""
    T = np.linspace(1.0, 100.0, 200)
    msd_true = msd_exponential(T, 1.15, 0.1)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_msd(T, msd_true, model='exponential')

    assert result is not None, 'fit_msd returned None'
    s = result.summary()
    assert isinstance(s, str), f'summary() returned {type(s)}, expected str'
    assert len(s) > 0,         'summary() returned empty string'
    assert 'exponential' in s, f"model name not in summary: {s!r}"
    assert 'R' in s,           f"R not in summary: {s!r}"


def test_05_fbm_recovers_H_from_brownian():
    """fit_msd recovers H ≈ 0.5 from MSD = T (ordinary Brownian motion)."""
    T = np.linspace(1.0, 100.0, 200)
    msd_linear = T * 1.0  # MSD = T → H = 0.5 exactly

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_msd(T, msd_linear, model='fbm')

    assert result is not None, 'fit_msd returned None'
    err = abs(result.params['H'] - 0.5)
    assert err < 0.1, f'|fitted_H - 0.5| = {err:.4f}, expected < 0.1'


def test_06_max_lag_fraction_halves_lags_used():
    """max_lag_fraction=0.5 → lags_used has length == len(T) // 2."""
    T = np.linspace(1.0, 100.0, 200)
    msd_true = msd_cosine(T, 1.3, 0.008)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_msd(T, msd_true, model='cosine', max_lag_fraction=0.5)

    assert result is not None, 'fit_msd returned None'
    expected = len(T) // 2
    actual   = len(result.lags_used)
    assert actual == expected, (
        f'len(lags_used) = {actual}, expected {expected}'
    )


def test_07_all_zero_msd_returns_none():
    """All-zero MSD returns None and prints ✗ message."""
    msd_zero = np.zeros(100)
    lags     = np.arange(1, 101, dtype=float)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_msd(lags, msd_zero, model='cosine')

    assert result is None, f'Expected None, got {result}'
    output = buf.getvalue()
    assert '\u2717' in output, f'No \u2717 in output: {output!r}'


def test_08_poor_fit_prints_warning():
    """Fitting a plateau MSD (DNA model) with a power-law (fbm) prints ⚠."""
    # DNA plateau: rises sharply then saturates at a=5.0 within a short L range.
    # fbm (N·L^2H, monotonically increasing) cannot replicate a plateau shape.
    # With b=2.0 the plateau is reached near L=2, so >80 % of points are flat
    # at ~5.0 while the first few points rise from ~3. fbm fitted to this gives
    # R² ≈ 0.06 (well below 0.5), guaranteeing the ⚠ warning.
    #
    # Note: the CLAUDE.md spec suggested "exponential → fbm", but in the range
    # T=[1,100] with beta=0.1 both models are effectively power-law-shaped
    # (R²>0.8). DNA plateau data is used instead as it reliably gives R²<<0.5.
    L = np.linspace(0.5, 5.0, 100)
    msd_plateau = msd_dna(L, a=5.0, b=2.0, c=4.99)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fit_msd(L, msd_plateau, model='fbm')

    output = buf.getvalue()
    assert '\u26a0' in output, f'No \u26a0 in output: {output!r}'
    assert 'R' in output,     f'No R in output: {output!r}'


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ('01 cosine: recover μ (|err| < 0.1)',           test_01_cosine_recovers_mu),
    ('02 exponential: recover μ (|err| < 0.1)',      test_02_exponential_recovers_mu),
    ('03 confidence intervals bracket estimate',      test_03_confidence_intervals_bracket_estimate),
    ('04 summary() returns non-empty string',         test_04_summary_returns_non_empty_string),
    ('05 fbm: recover H ≈ 0.5 from Brownian MSD',    test_05_fbm_recovers_H_from_brownian),
    ('06 max_lag_fraction=0.5 halves lags_used',      test_06_max_lag_fraction_halves_lags_used),
    ('07 all-zero MSD → None + ✗ message',           test_07_all_zero_msd_returns_none),
    ('08 plateau → power-law: ⚠ warning printed',    test_08_poor_fit_prints_warning),
]

if __name__ == '__main__':
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total  = len(TESTS)
    print(f'\n{passed}/{total} tests passed.')
    sys.exit(0 if passed == total else 1)

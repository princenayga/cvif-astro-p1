"""
tests/test_models.py — Self-contained tests for whitenoise/core/models.py

Run with:
    python tests/test_models.py

Each test prints "PASS" or "FAIL: {reason}".
Final line: "X/12 tests passed."
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

from whitenoise.core.models import (
    msd_cosine, msd_exponential, msd_fbm, msd_dna,
    pdf_cosine, pdf_exponential, pdf_dna,
    get_model, list_models, MODELS,
)


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

def test_01_cosine_scalar_positive():
    """msd_cosine scalar → positive finite float."""
    result = msd_cosine(10.0, 1.2, 0.05)
    assert np.isfinite(result), f'Not finite: {result}'
    assert result > 0,          f'Not positive: {result}'


def test_02_cosine_nan_when_cos_nonpositive():
    """msd_cosine returns nan (no exception) when cos(nu*T/2) <= 0."""
    # arg = nu*T/2 = 0.05 * T_large / 2; push arg well past pi/2
    T_large = np.pi / 0.05 + 1.0   # arg = pi/2 + 0.025 → cos < 0
    result = msd_cosine(T_large, 1.2, 0.05)
    assert np.isnan(result), f'Expected nan, got {result}'


def test_03_cosine_array_T():
    """msd_cosine accepts ndarray T and returns same shape."""
    T_arr = np.linspace(1.0, 20.0, 50)
    result = msd_cosine(T_arr, 1.2, 0.05)
    assert result.shape == (50,),          f'Shape {result.shape} != (50,)'
    assert np.any(np.isfinite(result)),    'All nan — expected some finite values'


def test_04_exponential_scalar_positive():
    """msd_exponential scalar → positive finite float."""
    result = msd_exponential(10.0, 1.15, 0.1)
    assert np.isfinite(result), f'Not finite: {result}'
    assert result > 0,          f'Not positive: {result}'


def test_05_exponential_array_T():
    """msd_exponential accepts ndarray T and all values are finite."""
    T_arr = np.linspace(1.0, 50.0, 100)
    result = msd_exponential(T_arr, 1.15, 0.1)
    assert result.shape == (100,),       f'Shape {result.shape} != (100,)'
    assert np.all(np.isfinite(result)),  f'Non-finite values found'


def test_06_fbm_brownian_equals_T():
    """msd_fbm(T, H=0.5) == T (ordinary Brownian motion)."""
    T_arr = np.array([1.0, 4.0, 9.0, 16.0])
    result = msd_fbm(T_arr, 0.5)
    assert np.allclose(result, T_arr, rtol=1e-10), (
        f'Expected {T_arr}, got {result}'
    )


def test_07_pdf_cosine_integrates_to_one():
    """pdf_cosine integrates to approximately 1 (Gaussian normalization)."""
    dx = np.linspace(-100.0, 100.0, 20_000)
    pdf = pdf_cosine(dx, T=10.0, mu=1.2, nu=0.05)
    finite = np.isfinite(pdf)
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else getattr(np, 'trapz')
    integral = _trapz(pdf[finite], dx[finite])
    assert abs(integral - 1.0) < 0.05, (
        f'Integral = {integral:.4f}, expected close to 1.0'
    )


def test_08_pdf_exponential_integrates_to_one():
    """pdf_exponential integrates to approximately 1."""
    dx = np.linspace(-200.0, 200.0, 20_000)
    pdf = pdf_exponential(dx, T=10.0, mu=1.15, beta=0.1)
    finite = np.isfinite(pdf)
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else getattr(np, 'trapz')
    integral = _trapz(pdf[finite], dx[finite])
    assert abs(integral - 1.0) < 0.05, (
        f'Integral = {integral:.4f}, expected close to 1.0'
    )


def test_09_pdf_returns_nan_when_sigma2_invalid():
    """pdf_cosine returns all-nan array (no exception) when sigma^2 is invalid."""
    # T=1000, nu=0.05: the Bessel function at large argument goes negative,
    # making the product negative → msd_cosine returns nan → pdf returns all-nan
    pdf = pdf_cosine(
        np.linspace(-10.0, 10.0, 100),
        T=1000.0, mu=1.2, nu=0.05,
    )
    assert isinstance(pdf, np.ndarray),  'Expected ndarray'
    assert np.all(np.isnan(pdf)),        f'Expected all-nan; got {pdf[:5]}'


def test_10_get_model_cosine():
    """get_model('cosine') returns correct registry entry."""
    m = get_model('cosine')
    assert 'msd' in m,                    "'msd' key missing"
    assert 'pdf' in m,                    "'pdf' key missing"
    assert 'params' in m,                 "'params' key missing"
    assert m['status'] == 'available',    f"status={m['status']!r}"
    assert m['row'] == 10,                f"row={m['row']}"


def test_11_get_model_stub_raises_not_implemented():
    """get_model on a stub model raises NotImplementedError with expected text."""
    try:
        get_model('exp_whittaker')
        assert False, 'Should have raised NotImplementedError'
    except NotImplementedError as exc:
        msg = str(exc)
        assert 'not yet implemented' in msg, f"'not yet implemented' missing: {msg!r}"
        assert 'cosine' in msg,              f"'cosine' missing from: {msg!r}"


def test_12_get_model_unknown_raises_and_list_models_runs():
    """Unknown model raises ValueError with ✗; list_models() prints all 17 names."""
    # Part A: unknown name
    try:
        get_model('magic_model')
        assert False, 'Should have raised ValueError'
    except ValueError as exc:
        assert '✗' in str(exc), f"No ✗ in error: {exc}"

    # Part B: list_models() runs without error and contains all model names
    expected_names = list(MODELS.keys())
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        list_models()
    output = buf.getvalue()
    for name in expected_names:
        assert name in output, f"'{name}' not found in list_models() output"


def test_13_dna_msd_plateau_behavior():
    """msd_dna rises with L and plateaus near 'a' at large L."""
    L = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    msd = msd_dna(L, a=5.21, b=0.0024, c=3.81)

    assert msd[-1] > msd[0],               f'MSD should rise: {msd}'
    assert abs(msd[-1] - 5.21) < 0.01,     f'Expected plateau near 5.21, got {msd[-1]:.4f}'


def test_14_dna_msd_large_L_approaches_a():
    """msd_dna(L→∞) approaches 'a' to within 1e-6."""
    result = msd_dna(1e6, a=5.0, b=0.01, c=3.0)
    assert abs(result - 5.0) < 1e-6, f'Expected ~5.0, got {result}'


def test_15_pdf_dna_integrates_to_one():
    """pdf_dna integrates to approximately 1.0 (Gaussian normalization)."""
    dx = np.linspace(-50.0, 50.0, 10_000)
    pdf = pdf_dna(dx, L=100.0, a=5.21, b=0.0024, c=3.81)
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else getattr(np, 'trapz')
    integral = _trapz(pdf, dx)
    assert abs(integral - 1.0) < 0.05, f'Integral = {integral:.4f}, expected close to 1.0'


def test_16_get_model_dna():
    """get_model('dna') returns correct registry entry."""
    m = get_model('dna')
    assert m['status'] == 'available',     f"status={m['status']!r}"
    assert m['params'] == ['a', 'b', 'c'], f"params={m['params']}"
    assert m['n_params'] == 3,             f"n_params={m['n_params']}"
    assert m['row'] is None,               f"row={m['row']!r}"


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ('01 msd_cosine scalar positive',                test_01_cosine_scalar_positive),
    ('02 msd_cosine nan when cos <= 0',              test_02_cosine_nan_when_cos_nonpositive),
    ('03 msd_cosine array T returns shape (50,)',    test_03_cosine_array_T),
    ('04 msd_exponential scalar positive',           test_04_exponential_scalar_positive),
    ('05 msd_exponential array T all finite',        test_05_exponential_array_T),
    ('06 msd_fbm H=0.5 == T (Brownian)',             test_06_fbm_brownian_equals_T),
    ('07 pdf_cosine integrates to 1.0',              test_07_pdf_cosine_integrates_to_one),
    ('08 pdf_exponential integrates to 1.0',         test_08_pdf_exponential_integrates_to_one),
    ('09 pdf returns nan when sigma2 invalid',       test_09_pdf_returns_nan_when_sigma2_invalid),
    ('10 get_model cosine returns correct dict',     test_10_get_model_cosine),
    ('11 get_model stub raises NotImplementedError', test_11_get_model_stub_raises_not_implemented),
    ('12 unknown raises ValueError; list_models',    test_12_get_model_unknown_raises_and_list_models_runs),
    ('13 msd_dna plateau behavior',                  test_13_dna_msd_plateau_behavior),
    ('14 msd_dna L→∞ approaches a',                 test_14_dna_msd_large_L_approaches_a),
    ('15 pdf_dna integrates to 1.0',                 test_15_pdf_dna_integrates_to_one),
    ('16 get_model dna returns correct dict',        test_16_get_model_dna),
]

if __name__ == '__main__':
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print(f'\n{passed}/{total} tests passed.')
    sys.exit(0 if passed == total else 1)

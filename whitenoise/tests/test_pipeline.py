"""
tests/test_pipeline.py — pytest tests for whitenoise/analysis/pipeline.py

Run with:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import pytest

# Force UTF-8 output (Windows cp1252 safety)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from whitenoise.analysis.pipeline import analyze, AnalysisResult


# ── Shared helper ─────────────────────────────────────────────────────────────

def _write_csv(path: str, n: int = 300, seed: int = 42, constant: bool = False) -> None:
    """Write a minimal whitenoise-format CSV with synthetic data."""
    rng = np.random.default_rng(seed)
    time = np.arange(1, n + 1, dtype=float)
    if constant:
        values = np.full(n, 5.0)
    else:
        # Brownian motion — gives reasonable (non-zero) MSD
        values = np.cumsum(rng.standard_normal(n))

    with open(path, 'w') as f:
        f.write('time [index], value [units]\n')
        for t, v in zip(time, values):
            f.write(f'{t},{v:.6f}\n')


def _silent(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), suppressing all stdout output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_analyze_cosine_returns_result(tmp_path):
    """analyze() with cosine model returns a valid AnalysisResult."""
    csv = str(tmp_path / 'data.csv')
    _write_csv(csv)

    result = _silent(analyze, csv, model='cosine')

    assert isinstance(result, AnalysisResult), \
        f'Expected AnalysisResult, got {type(result)}'
    assert result.model == 'cosine'
    assert result.lags is not None and len(result.lags) > 0
    assert result.msd_empirical is not None and len(result.msd_empirical) > 0
    assert len(result.values) == 300


def test_02_analyze_exponential_returns_result(tmp_path):
    """analyze() with exponential model returns a valid AnalysisResult."""
    csv = str(tmp_path / 'data.csv')
    _write_csv(csv)

    result = _silent(analyze, csv, model='exponential')

    assert isinstance(result, AnalysisResult)
    assert result.model == 'exponential'
    assert result.time is not None and len(result.time) == 300


def test_03_detrend_none_skips_detrending(tmp_path):
    """detrend_method=None skips detrending; result is still returned."""
    csv = str(tmp_path / 'data.csv')
    _write_csv(csv)

    result = _silent(analyze, csv, model='cosine', detrend_method=None)

    assert isinstance(result, AnalysisResult), \
        f'Expected AnalysisResult, got {type(result)}'
    # Values should equal raw CSV values (no detrending applied)
    assert len(result.values) == 300


def test_04_normalize_true_gives_unit_std(tmp_path):
    """normalize=True z-scores the values: std of stored values ≈ 1.0."""
    csv = str(tmp_path / 'data.csv')
    _write_csv(csv)

    result = _silent(analyze, csv, model='cosine', normalize=True)

    assert isinstance(result, AnalysisResult)
    std = float(np.std(result.values))
    assert abs(std - 1.0) < 0.01, \
        f'std={std:.4f} after normalize=True, expected ≈ 1.0'


def test_05_fit_none_handled_gracefully(tmp_path):
    """Constant values produce all-zero MSD → fit=None; no exception raised."""
    csv = str(tmp_path / 'const.csv')
    _write_csv(csv, constant=True)

    result = _silent(analyze, csv, model='cosine')

    assert isinstance(result, AnalysisResult), \
        'Should return AnalysisResult even when fitting fails'
    assert result.fit is None, \
        f'Expected fit=None for constant (all-zero MSD) data, got {result.fit}'


def test_06_summary_prints_when_fit_valid(tmp_path):
    """summary() prints a non-empty SWNA block when fit succeeded."""
    csv = str(tmp_path / 'data.csv')
    _write_csv(csv)

    result = _silent(analyze, csv, model='cosine')

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result.summary()

    output = buf.getvalue()
    assert len(output) > 0, 'summary() produced no output'
    assert 'SWNA' in output, f"'SWNA' not found in summary output"
    assert result.dataset_name in output, \
        f"dataset_name {result.dataset_name!r} not in summary"


def test_07_summary_prints_na_when_fit_none(tmp_path):
    """summary() shows 'N/A' cleanly when fit is None (no exception)."""
    csv = str(tmp_path / 'const.csv')
    _write_csv(csv, constant=True)

    result = _silent(analyze, csv, model='cosine')
    assert result.fit is None, 'Expected fit=None for constant data'

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result.summary()   # must not raise

    output = buf.getvalue()
    assert 'N/A' in output, f"'N/A' not in summary output: {output!r}"


def test_08_dataset_name_from_filename(tmp_path):
    """dataset_name equals the CSV basename without the .csv extension."""
    csv = str(tmp_path / 'sunspot_data.csv')
    _write_csv(csv)

    result = _silent(analyze, csv, model='cosine')

    assert result.dataset_name == 'sunspot_data', \
        f"dataset_name={result.dataset_name!r}, expected 'sunspot_data'"

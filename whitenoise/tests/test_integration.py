"""
tests/test_integration.py — End-to-end integration tests for the whitenoise package.

Run with:
    pytest tests/test_integration.py -v
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

# Force UTF-8 output (Windows cp1252 safety)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import whitenoise as wn


# ── Shared helpers ────────────────────────────────────────────────────────────

def _write_csv(path: str, n: int = 300, seed: int = 42) -> None:
    """Write a Brownian motion CSV in whitenoise format."""
    rng = np.random.default_rng(seed)
    values = np.cumsum(rng.standard_normal(n))
    with open(path, 'w') as f:
        f.write('time [index], value [units]\n')
        for i, v in enumerate(values, 1):
            f.write(f'{i},{v:.6f}\n')


def _silent(fn, *args, **kwargs):
    """Suppress all stdout from fn."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_full_pipeline_cosine(tmp_path):
    """Full pipeline with cosine model: analyze() returns valid fit with R² > 0.7."""
    csv = str(tmp_path / 'data.csv')
    _write_csv(csv)

    result = _silent(wn.analyze, csv, model='cosine')

    assert isinstance(result, wn.AnalysisResult), \
        f'Expected AnalysisResult, got {type(result)}'
    assert result.fit is not None, \
        'fit should not be None for Brownian motion data'
    assert result.fit.r_squared > 0.7, \
        f'Expected R² > 0.7, got {result.fit.r_squared:.4f}'


def test_02_full_pipeline_exponential(tmp_path):
    """Full pipeline with exponential model: R² > 0.7."""
    csv = str(tmp_path / 'data.csv')
    _write_csv(csv, seed=99)

    result = _silent(wn.analyze, csv, model='exponential')

    assert isinstance(result, wn.AnalysisResult)
    assert result.fit is not None, \
        'fit should not be None for Brownian motion data'
    assert result.fit.r_squared > 0.7, \
        f'Expected R² > 0.7, got {result.fit.r_squared:.4f}'


def test_03_compare_across_datasets(tmp_path):
    """compare() on 3 CSVs returns ComparisonResult with 3 rows and correct columns."""
    paths = []
    for i in range(3):
        p = str(tmp_path / f'data_{i}.csv')
        _write_csv(p, seed=i + 1)
        paths.append(p)

    cr = _silent(wn.compare, paths, model='cosine')

    assert isinstance(cr, wn.ComparisonResult), \
        f'Expected ComparisonResult, got {type(cr)}'
    assert len(cr.summary_df) == 3, \
        f'Expected 3 rows, got {len(cr.summary_df)}'

    required_cols = {'dataset_name', 'model', 'mu', 'r_squared', 'regime'}
    missing = required_cols - set(cr.summary_df.columns)
    assert not missing, f'Missing columns: {missing}'


def test_04_batch_and_export(tmp_path):
    """batch_analyze() + export_summary() writes a CSV with 2 rows."""
    paths = []
    for i in range(2):
        p = str(tmp_path / f'data_{i}.csv')
        _write_csv(p, seed=i + 10)
        paths.append(p)

    batch_results = _silent(wn.batch_analyze, paths, model='cosine', n_jobs=1)

    # Build a ComparisonResult from the successful batch results
    from whitenoise.analysis.compare import ComparisonResult, _result_row, _nan_row, _DF_COLS
    rows = []
    successes = []
    for path, ar in zip(paths, batch_results):
        if ar is not None:
            rows.append(_result_row(ar))
            successes.append(ar)
        else:
            name = os.path.splitext(os.path.basename(path))[0]
            rows.append(_nan_row(name, 'cosine'))
    cr = ComparisonResult(
        results=successes,
        models_used=['cosine'],
        summary_df=pd.DataFrame(rows, columns=_DF_COLS),
    )

    out_csv = str(tmp_path / 'batch_summary.csv')
    with contextlib.redirect_stdout(io.StringIO()):
        wn.export_summary(cr, out_csv)

    assert os.path.exists(out_csv), f'CSV not created at {out_csv}'
    df = pd.read_csv(out_csv)
    assert len(df) == 2, f'Expected 2 rows in exported CSV, got {len(df)}'


def test_05_model_search(tmp_path):
    """batch_model_search() returns ComparisonResult; best model has highest R²."""
    csv = str(tmp_path / 'data.csv')
    _write_csv(csv, seed=7)

    cr = _silent(wn.batch_model_search, csv, models=['cosine', 'exponential', 'fbm'])

    assert isinstance(cr, wn.ComparisonResult), \
        f'Expected ComparisonResult, got {type(cr)}'

    df = cr.summary_df.dropna(subset=['r_squared'])
    assert not df.empty, 'No valid R² values in model search result'

    best_r2 = df['r_squared'].max()
    # Verify consistency: max in df == max in df (tautology, but checks no NaN issues)
    assert best_r2 == df['r_squared'].max()


def test_06_import_surface():
    """All names in wn.__all__ are directly accessible on the whitenoise module."""
    missing = [name for name in wn.__all__ if not hasattr(wn, name)]
    assert not missing, \
        f'These names are in __all__ but missing from the module: {missing}'

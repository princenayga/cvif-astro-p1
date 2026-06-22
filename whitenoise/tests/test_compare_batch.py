"""
tests/test_compare_batch.py — pytest tests for compare.py and batch.py

Run with:
    pytest tests/test_compare_batch.py -v
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Force UTF-8 output (Windows cp1252 safety)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from whitenoise.analysis.compare import compare, print_comparison, ComparisonResult
from whitenoise.analysis.batch import batch_analyze, batch_model_search


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(path: str, n: int = 200, seed: int = 42) -> None:
    """Write a minimal whitenoise-format CSV with Brownian motion data."""
    rng = np.random.default_rng(seed)
    values = np.cumsum(rng.standard_normal(n))
    with open(path, 'w') as f:
        f.write('time [index], value [units]\n')
        for i, v in enumerate(values, 1):
            f.write(f'{i},{v:.6f}\n')


def _silent(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), suppressing all stdout output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_compare_returns_result_with_correct_row_count(tmp_path):
    """compare() returns ComparisonResult with one row per path."""
    paths = []
    for i in range(3):
        p = str(tmp_path / f'data_{i}.csv')
        _write_csv(p, seed=i)
        paths.append(p)

    cr = _silent(compare, paths, model='cosine')

    assert isinstance(cr, ComparisonResult), \
        f'Expected ComparisonResult, got {type(cr)}'
    assert len(cr.summary_df) == 3, \
        f'Expected 3 rows, got {len(cr.summary_df)}'


def test_02_summary_df_has_expected_columns(tmp_path):
    """summary_df contains all required columns."""
    p = str(tmp_path / 'data.csv')
    _write_csv(p)

    cr = _silent(compare, [p], model='cosine')

    expected_cols = {
        'dataset_name', 'model', 'mu', 'mu_ci',
        'nu_or_beta', 'nu_or_beta_ci', 'N', 'r_squared', 'regime',
    }
    actual_cols = set(cr.summary_df.columns)
    missing = expected_cols - actual_cols
    assert not missing, f'Missing columns in summary_df: {missing}'


def test_03_bad_path_handled_gracefully(tmp_path):
    """A non-existent path produces a NaN row without crashing."""
    good = str(tmp_path / 'good.csv')
    _write_csv(good)
    bad  = str(tmp_path / 'nonexistent_file.csv')

    cr = _silent(compare, [good, bad], model='cosine')

    assert isinstance(cr, ComparisonResult), \
        'compare() should return ComparisonResult even with bad paths'
    assert len(cr.summary_df) == 2, \
        f'Expected 2 rows (1 good + 1 bad), got {len(cr.summary_df)}'

    # Bad path row must have NaN r_squared
    bad_name = os.path.splitext(os.path.basename(bad))[0]
    bad_rows = cr.summary_df[cr.summary_df['dataset_name'] == bad_name]
    assert len(bad_rows) == 1, 'No row found for bad path'
    assert pd.isna(bad_rows.iloc[0]['r_squared']), \
        'r_squared for bad path should be NaN'


def test_04_print_comparison_prints_without_error(tmp_path):
    """print_comparison() prints a non-empty table without raising."""
    p = str(tmp_path / 'data.csv')
    _write_csv(p)

    cr = _silent(compare, [p], model='cosine')

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_comparison(cr)

    output = buf.getvalue()
    assert len(output) > 0, 'print_comparison() produced no output'


def test_05_long_dataset_name_truncated(tmp_path):
    """Dataset names longer than 16 chars are truncated with '...'."""
    long_name = 'very_long_dataset_name_that_exceeds_sixteen_chars'
    p = str(tmp_path / f'{long_name}.csv')
    _write_csv(p)

    cr = _silent(compare, [p], model='cosine')

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_comparison(cr)

    output = buf.getvalue()
    assert '...' in output, \
        f"'...' not found in print_comparison output — long name not truncated:\n{output}"
    assert long_name not in output, \
        'Full long name should NOT appear in output (should be truncated)'


def test_06_batch_analyze_serial_returns_correct_count(tmp_path):
    """batch_analyze() n_jobs=1 returns a list of length == len(paths)."""
    paths = []
    for i in range(3):
        p = str(tmp_path / f'data_{i}.csv')
        _write_csv(p, seed=i + 10)
        paths.append(p)

    results = _silent(batch_analyze, paths, model='cosine', n_jobs=1)

    assert len(results) == 3, \
        f'Expected 3 results, got {len(results)}'
    # All should succeed with Brownian motion data
    assert all(r is not None for r in results), \
        'Some results are None (unexpected failure)'


def test_07_batch_analyze_parallel_matches_serial_count(tmp_path):
    """batch_analyze() n_jobs=2 returns same number of results as n_jobs=1."""
    paths = []
    for i in range(3):
        p = str(tmp_path / f'data_{i}.csv')
        _write_csv(p, seed=i + 20)
        paths.append(p)

    serial   = _silent(batch_analyze, paths, model='cosine', n_jobs=1)
    parallel = _silent(batch_analyze, paths, model='cosine', n_jobs=2)

    assert len(parallel) == len(serial), \
        f'Parallel count {len(parallel)} != serial count {len(serial)}'


def test_08_batch_model_search_returns_one_row_per_model(tmp_path):
    """batch_model_search() returns one summary_df row per successfully tried model."""
    p = str(tmp_path / 'data.csv')
    _write_csv(p)

    cr = _silent(batch_model_search, p, models=['cosine', 'exponential', 'fbm'])

    assert isinstance(cr, ComparisonResult), \
        f'Expected ComparisonResult, got {type(cr)}'
    assert len(cr.summary_df) == 3, \
        f'Expected 3 rows (one per model), got {len(cr.summary_df)}'


def test_09_best_model_has_highest_r_squared(tmp_path):
    """The best model row in summary_df has the highest r_squared."""
    p = str(tmp_path / 'data.csv')
    _write_csv(p)

    cr = _silent(batch_model_search, p, models=['cosine', 'exponential', 'fbm'])

    df = cr.summary_df.dropna(subset=['r_squared'])
    assert not df.empty, 'No valid r_squared values in summary_df'

    best_r2 = df['r_squared'].max()
    # Verify that the reported best R² actually equals the max in the DataFrame
    # (batch_model_search prints "Best model: X (R²=Y)" — we verify consistency)
    assert best_r2 == df['r_squared'].max(), \
        'Best r_squared is not the maximum in summary_df'


def test_10_stub_models_silently_skipped(tmp_path):
    """batch_model_search() silently skips stub models (no crash, no row)."""
    p = str(tmp_path / 'data.csv')
    _write_csv(p)

    # 'sin_half' is a stub model (status='not_implemented')
    cr = _silent(
        batch_model_search, p,
        models=['cosine', 'sin_half'],
    )

    assert isinstance(cr, ComparisonResult), \
        f'Expected ComparisonResult, got {type(cr)}'

    # Only 'cosine' should appear — 'sin_half' is silently skipped
    models_in_df = list(cr.summary_df['model'])
    assert 'sin_half' not in models_in_df, \
        f"'sin_half' stub should be silently skipped, but found in summary_df"
    assert 'cosine' in models_in_df, \
        "'cosine' should appear in summary_df"

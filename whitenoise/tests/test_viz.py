"""
tests/test_viz.py — pytest tests for viz/explore.py and viz/publish.py

Run with:
    pytest tests/test_viz.py -v
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')  # must be set before any other matplotlib imports

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pytest

# Force UTF-8 output (Windows cp1252 safety)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from whitenoise.analysis.pipeline import analyze, AnalysisResult
from whitenoise.analysis.compare import compare, ComparisonResult
from whitenoise.viz.explore import plot_msd, plot_pdf, plot_timeseries, plot_diagnostics
from whitenoise.viz.publish import publish_msd, publish_pdf, publish_comparison


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _write_csv(path: str, n: int = 200, seed: int = 42) -> None:
    """Write a minimal whitenoise-format CSV with Brownian motion data."""
    rng = np.random.default_rng(seed)
    values = np.cumsum(rng.standard_normal(n))
    with open(path, 'w') as f:
        f.write('time [index], value [units]\n')
        for i, v in enumerate(values, 1):
            f.write(f'{i},{v:.6f}\n')


def _silent(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    """Shared AnalysisResult built from a synthetic CSV."""
    d = tmp_path_factory.mktemp('viz')
    csv = str(d / 'data.csv')
    _write_csv(csv)
    return _silent(analyze, csv, model='cosine')


@pytest.fixture(scope='module')
def result_no_fit(tmp_path_factory):
    """AnalysisResult where fit is None (constant data → all-zero MSD)."""
    d = tmp_path_factory.mktemp('viz_nofit')
    csv = str(d / 'const.csv')
    # Constant series → MSD ~0 → fit fails → fit=None
    with open(csv, 'w') as f:
        f.write('time [index], value [units]\n')
        for i in range(1, 201):
            f.write(f'{i},5.000000\n')
    return _silent(analyze, csv, model='cosine')


@pytest.fixture(scope='module')
def comparison_result(tmp_path_factory):
    """ComparisonResult from two synthetic CSVs."""
    d = tmp_path_factory.mktemp('viz_comp')
    paths = []
    for i in range(2):
        p = str(d / f'data_{i}.csv')
        _write_csv(p, seed=i + 5)
        paths.append(p)
    return _silent(compare, paths, model='cosine')


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_plot_msd_returns_figure_when_fit_valid(result):
    """plot_msd() returns a Figure when fit succeeded."""
    fig = plot_msd(result)
    assert isinstance(fig, matplotlib.figure.Figure), \
        f'Expected Figure, got {type(fig)}'
    plt.close(fig)


def test_02_plot_msd_returns_figure_when_fit_none(result_no_fit):
    """plot_msd() returns a Figure without raising when fit is None."""
    fig = plot_msd(result_no_fit)
    assert isinstance(fig, matplotlib.figure.Figure), \
        f'Expected Figure, got {type(fig)}'
    plt.close(fig)


def test_03_plot_pdf_returns_figure_with_default_lag(result):
    """plot_pdf() returns a Figure using the default lag_index (len//4)."""
    fig = plot_pdf(result)
    assert isinstance(fig, matplotlib.figure.Figure), \
        f'Expected Figure, got {type(fig)}'
    # Check that title contains the expected lag (lags[len//4])
    expected_T = float(result.lags[max(0, len(result.lags) // 4)])
    title_text = fig.axes[0].get_title()
    assert f'T={expected_T:.3f}' in title_text, \
        f'Title {title_text!r} does not contain expected lag T={expected_T:.3f}'
    plt.close(fig)


def test_04_plot_timeseries_returns_figure(result):
    """plot_timeseries() returns a Figure with the correct number of lines."""
    fig = plot_timeseries(result)
    assert isinstance(fig, matplotlib.figure.Figure), \
        f'Expected Figure, got {type(fig)}'
    ax = fig.axes[0]
    assert len(ax.lines) >= 1, 'Expected at least one line in time series plot'
    plt.close(fig)


def test_05_plot_diagnostics_returns_figure_with_4_axes(result):
    """plot_diagnostics() returns a Figure with exactly 4 axes."""
    fig = plot_diagnostics(result)
    assert isinstance(fig, matplotlib.figure.Figure), \
        f'Expected Figure, got {type(fig)}'
    assert len(fig.axes) == 4, \
        f'Expected 4 axes in diagnostics figure, got {len(fig.axes)}'
    plt.close(fig)


def test_06_publish_msd_returns_figure(result):
    """publish_msd() returns a Figure with publication styling."""
    fig = publish_msd(result)
    assert isinstance(fig, matplotlib.figure.Figure), \
        f'Expected Figure, got {type(fig)}'
    # DPI should be 150 per STYLE
    assert fig.get_dpi() == 150, \
        f'Expected dpi=150, got {fig.get_dpi()}'
    plt.close(fig)


def test_07_publish_comparison_returns_figure_with_error_bars(comparison_result):
    """publish_comparison() returns a Figure containing error bar containers."""
    fig = publish_comparison(comparison_result)
    assert isinstance(fig, matplotlib.figure.Figure), \
        f'Expected Figure, got {type(fig)}'
    ax = fig.axes[0]
    # Error bars are stored as ErrorbarContainer objects in ax.containers
    # or as line collections; check that there is at least one bar
    assert len(ax.patches) >= 1 or len(ax.containers) >= 1, \
        'publish_comparison() figure has no bars or containers'
    plt.close(fig)


def test_08_publish_msd_saves_pdf(result, tmp_path):
    """publish_msd() saves a PDF file when save_path is provided."""
    out = str(tmp_path / 'msd_output.pdf')
    with contextlib.redirect_stdout(io.StringIO()):
        fig = publish_msd(result, save_path=out)
    assert os.path.exists(out), \
        f'PDF not created at {out}'
    assert os.path.getsize(out) > 0, \
        f'PDF at {out} is empty'
    plt.close(fig)

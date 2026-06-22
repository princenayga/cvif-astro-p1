# whitenoise — Stochastic White Noise Analysis (SWNA)
# Hida-Bernido framework — University of San Carlos

from .io.reader import read_csv, read_csv_multi
from .io.export import export_csv, export_summary
from .core.msd import compute_msd
from .core.models import list_models, get_model
from .core.fitting import fit_msd, FitResult
from .utils.preprocess import detrend, normalize, smooth
from .analysis.pipeline import analyze, AnalysisResult
from .analysis.compare import compare, print_comparison, ComparisonResult
from .analysis.batch import batch_analyze, batch_model_search
from .viz.explore import plot_msd, plot_pdf, plot_timeseries, plot_diagnostics
from .viz.publish import publish_msd, publish_pdf, publish_comparison

__version__ = '0.1.0'
__all__ = [
    'read_csv', 'read_csv_multi',
    'export_csv', 'export_summary',
    'compute_msd',
    'list_models', 'get_model',
    'fit_msd', 'FitResult',
    'detrend', 'normalize', 'smooth',
    'analyze', 'AnalysisResult',
    'compare', 'print_comparison', 'ComparisonResult',
    'batch_analyze', 'batch_model_search',
    'plot_msd', 'plot_pdf', 'plot_timeseries', 'plot_diagnostics',
    'publish_msd', 'publish_pdf', 'publish_comparison',
]

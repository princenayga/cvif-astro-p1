# CLAUDE.md — whitenoise Package

## What This Package Is
A standalone Python research package implementing Stochastic White Noise
Analysis (SWNA) based on the Hida white noise calculus framework developed
by Bernido & Carpio-Bernido (University of San Carlos / CVIF, Philippines).

This is a RESEARCH TOOL. No built-in datasets. No workshop code.
No Colab assumptions. The researcher always provides their own CSV.

---

## Primary Input Format — Exact Specification

Standard input is a 2-column CSV file:

    time [months], sunspot_number [count]
    1, 12.4
    2, 15.1
    3, 14.8

Rules (enforced by io/reader.py):
  - Column 1 is ALWAYS time or index. Column 2 is ALWAYS the observable.
  - First row is always the header. No exceptions.
  - Header format per column:  name [unit]
      "time [months]"        → name='time',   unit='months'
      "distance [km]"        → name='distance', unit='km'
      "flux []"              → name='flux',   unit=''   (unitless)
      "value [unitless]"     → name='value',  unit='unitless'
  - Space before bracket is optional — parser handles both:
      "time [months]"   and   "time[months]"  both parse correctly
  - Column names are case-insensitive for internal use but preserved
    as-written for display.
  - Multi-column CSVs are supported for batch mode:
      time [yr], co2 [ppm], temperature [°C], sst [°C]
    Column 1 = time, all others = separate observables to analyze.

Units are used for:
  1. Axis labels on all plots (automatic, no manual labeling needed)
  2. Data validation (see validation rules below)
  Units are NOT stored beyond the session — no persistence needed.

---

## CSV Validation Rules (io/reader.py enforces these)

On read, the package checks and WARNS (never crashes) for:

  V1. Non-monotonic time:
      If time values are not strictly increasing, warn:
      "⚠ Time column is not strictly increasing. Check your data order."

  V2. Implausible values given unit:
      If unit contains 'count', 'counts', 'number', 'freq' and
      any value is negative, warn:
      "⚠ Negative values found in column '{name}' with unit '{unit}'.
         Expected non-negative counts."

  V3. Possible column swap:
      If the time column has values with std > 1000× the mean, warn:
      "⚠ Time column values look unusually large. Are columns swapped?"
      (Catches cases like accidentally putting the observable in column 1)

  V4. Unrecognized time unit:
      KNOWN_TIME_UNITS = {
          'day', 'days', 'month', 'months', 'year', 'years', 'yr',
          'hour', 'hours', 'hr', 'min', 'minute', 'minutes',
          'second', 'seconds', 's', 'ms', 'index', 'step', ''
      }
      If time unit not in KNOWN_TIME_UNITS (case-insensitive), warn:
      "⚠ Unrecognized time unit '{unit}'. Expected one of: {list}.
         Proceeding anyway."

All warnings use the ⚠ prefix. All are printed to stdout.
None of them stop execution.

---

## Scientific Framework

Core stochastic integral:
    x(T) = x₀ + ∫₀ᵀ f(T-t) · h(t) · ω(t) dt

Full pipeline:
    CSV  →  read_csv()  →  (time, values, metadata)
         →  [optional: wn.detrend(), wn.normalize()]
         →  analyze()
         →      compute_msd()
         →      fit_msd()   →  FitResult (params + CIs)
         →  AnalysisResult
         →  .summary() / .plot_all() / .to_csv()

---

## The 16 Models (Table 3.1, Bernido & Carpio-Bernido 2015)

Selected by name string. After fitting, package prints R² and suggests
alternatives if R² < 0.5. Package does NOT auto-select — researcher chooses.

Priority models (fully implemented):
  'cosine'       row 10   params: μ, ν
  'exponential'  row 4    params: μ, β
  'sine'         row 9    params: μ, ν
  'fbm'          row 1    params: H

Extended models (stubbed — NotImplementedError, implement later):
  'exp_whittaker'  row 5    params: μ, β, ν
  'bessel_K'       row 6    params: μ, β
  'hypergeom_F1'   row 7    params: μ, β, ν
  'bessel_I'       row 8    params: μ, β
  'sin_half'       row 2    params: none
  'cos_half'       row 3    params: none
  'hypergeom_3F2'  row 11   params: μ, β, λ
  'csc_power'      row 12   params: ν, c
  'cot_power'      row 13   params: ν, c
  'inc_gamma'      row 14   params: ν, μ
  'bessel_pair'    row 15   params: ν
  'bessel_pair2'   row 16   params: ν, μ

---

## Parameters
  μ (mu):   memory parameter
              μ < 1   subdiffusive
              μ = 1   Brownian (no memory)
              μ > 1   superdiffusive
              μ > 2   hyperballistic
  ν (nu):   characteristic frequency (cosine, sine models)
  β (beta): exponential decay rate (exponential and related models)
  H:        Hurst exponent (fBm, H=0.5 → Brownian)
  N:        normalization scalar — always fitted alongside physical
            params, scales MSD amplitude without changing shape

FitResult always includes 95% confidence intervals for every parameter.

---

## Package Structure

whitenoise/
├── __init__.py
├── io/
│   ├── __init__.py
│   ├── reader.py       read_csv(), read_csv_multi(), validation
│   └── export.py       to_csv(), to_excel(), summary_table()
├── core/
│   ├── __init__.py
│   ├── msd.py          compute_msd()
│   ├── models.py       16 models: msd_*(), pdf_*(), MODELS registry
│   ├── fitting.py      fit_msd() → FitResult with CIs
│   └── pdf.py          displacement_histogram()
├── analysis/
│   ├── __init__.py
│   ├── pipeline.py     analyze() → AnalysisResult
│   ├── compare.py      compare() → ComparisonResult
│   └── batch.py        batch_analyze() → ComparisonResult
├── viz/
│   ├── __init__.py
│   ├── explore.py      quick inspection plots (annotated)
│   └── publish.py      publication-quality plots (clean)
└── utils/
    ├── __init__.py
    └── preprocess.py   detrend(), normalize(), smooth()

---

## Axis Label Construction (from units)

_make_axis_label(name, unit):
  unit == '' or unit == 'unitless'  →  return name
  otherwise                          →  return f"{name} ({unit})"

Examples:
  'sunspot_number', 'count'    →  'sunspot_number (count)'
  'flux', ''                   →  'flux'
  'time', 'months'             →  'time (months)'
  'distance', 'km'             →  'distance (km)'

---

## Batch Processing — Three Input Modes

batch_analyze() handles all three:
  Mode A: folder path string  →  reads all *.csv in folder
  Mode B: list of file paths  →  reads each CSV
  Mode C: multi-column CSV    →  each non-time column = one system

All three return ComparisonResult.

---

## Key Design Rules

1. PRIMARY INPUT: CSV with format "name [unit]" headers, col1=time, col2=value.
2. Parser handles space or no space before bracket.
3. Units → axis labels automatically. No manual label needed ever.
4. Validation: 4 warning types (V1–V4), all non-fatal, all print ⚠ prefix.
5. Model selection: manual by researcher. Package suggests after low R².
6. FitResult: params, std_errors, confidence_intervals (95%), r_squared,
   model, lags_used, msd_fitted. Always complete.
7. AnalysisResult: FitResult + lags + msd_empirical + metadata + methods.
8. Two viz modes: explore (annotated, fast) and publish (clean, paper-ready).
9. All viz functions: ax=None, show=True. Return matplotlib Axes or Figure.
10. All errors: human-readable. Never let raw scipy/numpy tracebacks reach user.
11. Type hints + docstrings with examples on all public functions.
12. Warnings use ⚠ prefix. Errors use ✗ prefix. Success uses ✓ prefix.

---

## Validated Benchmarks (for testing)
  Earthquake (exponential):    μ ≈ 1.00–1.19  (Roque et al. 2024)
  Solar sunspot (exponential): μ ≈ 1.15        (Toledo et al. 2024)
  GBR coral (cosine):          μ ≈ 4.64        (Elnar et al. 2021)
  CO₂ Keeling (cosine):        μ ≈ 0.91–0.97  (Elnar et al. 2024)
  X-ray binaries (cosine):     μ ∈ [0.50,1.39] (Calotes thesis 2024)

---

## Workshop Usage (entirely separate from this package)

The workshop Colab notebook (NOT in this repo) uses this package:
    import whitenoise as wn
    result = wn.analyze('earthquake_ph.csv', model='exponential')
    result.summary()
    result.plot_all()

CSV files are distributed separately (Google Drive, USB, etc.).
This package has zero knowledge that a workshop exists.
```

---

## Build Status

All 8 prompts complete. Final test suite result:

    pytest tests/ -v --tb=short
    78 passed in 2.42s

Breakdown by file:
  tests/test_reader.py           12/12
  tests/test_msd_preprocess.py   10/10
  tests/test_models.py           16/16
  tests/test_fitting.py           8/8
  tests/test_pipeline.py          8/8
  tests/test_compare_batch.py    10/10
  tests/test_viz.py               8/8
  tests/test_integration.py       6/6
  ─────────────────────────────────
  TOTAL                          78/78

---

## Quickstart

    import whitenoise as wn
    result = wn.analyze("mydata.csv", model="cosine")
    result.summary()
    wn.plot_diagnostics(result).show()

---

## Public API

All names below are importable directly from `whitenoise` (i.e. `wn.<name>`).

### I/O
  read_csv(path)               Read a 2-column whitenoise-format CSV → (time, values, metadata).
  read_csv_multi(path)         Read a multi-column CSV → list of (time, values, metadata) tuples.
  export_csv(result, path)     Save lags + empirical/fitted MSD for an AnalysisResult to CSV.
  export_summary(cr, path)     Save ComparisonResult.summary_df to CSV.

### Core
  compute_msd(x)               Compute empirical MSD → (lags, msd) arrays.
  list_models()                Print formatted table of all 16 SWNA models and their status.
  get_model(name)              Return the MODELS registry entry for a named model.
  fit_msd(lags, msd, model)    Fit N·msd_model to empirical MSD → FitResult or None.
  FitResult                    Dataclass: params, std_errors, confidence_intervals, r_squared,
                               model, lags_used, msd_fitted. Has .summary() → str.

### Preprocessing (optional, applied before analyze())
  detrend(values, method)      Remove trend ('linear', 'polynomial', 'mean') → residuals.
  normalize(values, method)    Normalize time series ('zscore', 'minmax', 'mean').
  smooth(values, window)       Smooth time series ('moving_average', 'gaussian').

### Analysis
  analyze(path, model)         Full SWNA pipeline on a CSV → AnalysisResult.
  AnalysisResult               Dataclass: dataset_name, model, fit, lags, msd_empirical,
                               values, time, metadata. Has .summary() (prints), .regime property.
  compare(paths, model)        Run one model across multiple CSVs → ComparisonResult.
  print_comparison(cr)         Print ASCII table from a ComparisonResult.
  ComparisonResult             Container: results list, models_used, summary_df DataFrame.
  batch_analyze(paths, model)  Run one model on many files (serial or parallel) → list of results.
  batch_model_search(path)     Try all available models on one file → ComparisonResult.

### Exploratory Plots (viz/explore.py)
  plot_msd(result)             Empirical MSD scatter + fitted curve (interactive).
  plot_pdf(result)             Displacement histogram + theoretical Gaussian PDF.
  plot_timeseries(result)      Raw preprocessed time series.
  plot_diagnostics(result)     2×2 figure: time series, MSD, PDF, parameter text box.

### Publication Plots (viz/publish.py)
  publish_msd(result)          MSD plot with serif font, inward ticks, dpi=150.
  publish_pdf(result)          PDF plot with publication styling.
  publish_comparison(cr)       μ bar chart with 95% CI error bars across datasets.

---

## Known Limitations

1. **12 stub models**: The following models in `core/models.py` raise
   `NotImplementedError` when called — they are registered in the MODELS
   registry but their MSD formulas are not yet implemented:
     exp_whittaker, bessel_K, hypergeom_F1, bessel_I, sin_half, cos_half,
     hypergeom_3F2, csc_power, cot_power, inc_gamma, bessel_pair, bessel_pair2.
   Use `wn.list_models()` to see current status.

2. **batch_analyze parallel mode**: When `n_jobs > 1`, worker threads call
   `analyze()` which reads from disk. All CSV paths must be accessible from
   the calling process's file system. In-memory data (numpy arrays) cannot
   be passed to parallel workers — use `n_jobs=1` in that case.

3. **matplotlib not auto-imported**: `import whitenoise as wn` does not
   import matplotlib at package level. The viz functions (`wn.plot_*`,
   `wn.publish_*`) import matplotlib lazily on first call. If your
   environment has no display (e.g. headless server), set the backend
   before calling any viz function:
     import matplotlib; matplotlib.use("Agg")

---

## Prompt 1 ✓ — Scaffold + io/reader.py
```
Read CLAUDE.md fully before writing any code.

Task: Create the full package scaffold and implement io/reader.py.
This is the foundation everything else depends on — get it right first.

─── 1. Create directory structure ───────────────────────

whitenoise/
├── __init__.py          (empty)
├── io/
│   ├── __init__.py      (empty)
│   └── reader.py
├── core/
│   └── __init__.py      (empty)
├── analysis/
│   └── __init__.py      (empty)
├── viz/
│   └── __init__.py      (empty)
└── utils/
    └── __init__.py      (empty)

─── 2. Implement io/reader.py ───────────────────────────

Public functions:

def read_csv(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read a 2-column whitenoise-format CSV file.

    Header format: name [unit]  (space before bracket is optional)
    Column 1 = time/index, Column 2 = observable. Always.
    Unitless columns use empty brackets: flux []

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    time : np.ndarray (1D float)
    values : np.ndarray (1D float)
    metadata : dict
        Keys:
          'time_name'    str   e.g. 'time'
          'time_unit'    str   e.g. 'months'  ('' if empty brackets)
          'value_name'   str   e.g. 'sunspot_number'
          'value_unit'   str   e.g. 'count'   ('' if empty brackets)
          'time_label'   str   axis-ready: 'time (months)' or 'time'
          'value_label'  str   axis-ready: 'sunspot_number (count)'
          'source_file'  str   the path argument
          'n_points'     int   number of data rows

    Raises
    ------
    FileNotFoundError  with message: "✗ File not found: {path}"
    ValueError         with message: "✗ {reason}" for bad format

    Prints ⚠ warnings for validation issues (never raises for these).

    Example
    -------
    >>> time, values, meta = wn.read_csv('sunspot.csv')
    >>> print(meta['value_label'])   # 'sunspot_number (count)'
    >>> print(meta['time_label'])    # 'time (months)'
    """

def read_csv_multi(path: str) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """
    Read a multi-column CSV. Returns one (time, values, metadata) tuple
    per non-time column.
    Column 1 is always time. All other columns are observables.
    Used internally by batch_analyze().

    Example
    -------
    >>> results = wn.read_csv_multi('climate.csv')
    >>> len(results)   # 3 if CSV has time + 3 observable columns
    """

─── 3. Internal helpers (not public) ────────────────────

def _parse_header_column(header: str) -> tuple[str, str]:
    """
    Parse header string → (name, unit).

    Examples:
    'time [months]'      → ('time', 'months')
    'time[months]'       → ('time', 'months')   ← no space, still works
    'sunspot_number [count]' → ('sunspot_number', 'count')
    'flux []'            → ('flux', '')
    'flux[  ]'           → ('flux', '')          ← whitespace inside brackets
    'value [unitless]'   → ('value', 'unitless')
    'distance'           → ('distance', '')      ← no brackets at all
    """
    # Implementation hint: use regex r'([^\[]+)\s*(?:\[([^\]]*)\])?'
    # Strip whitespace from both name and unit

def _make_axis_label(name: str, unit: str) -> str:
    """
    Build axis label string.
    '' or 'unitless' → return name only (no parentheses)
    otherwise        → return f"{name} ({unit})"
    """

def _validate_data(time, values, metadata):
    """
    Run all 4 validation checks. Print ⚠ warnings. Never raise.

    V1. Non-monotonic time:
        if not all(time[i] < time[i+1]):
        "⚠ Time column is not strictly increasing. Check your data order."

    V2. Implausible values given unit:
        if metadata['value_unit'].lower() in
           {'count','counts','number','numbers','freq','frequency'}
        and any(values < 0):
        "⚠ Negative values found in '{value_name}' with unit '{unit}'.
           Expected non-negative counts."

    V3. Possible column swap:
        if len(time) > 1 and np.std(time) > 1000 * abs(np.mean(time) + 1e-10):
        "⚠ Time column values look unusually large. Are columns swapped?"

    V4. Unrecognized time unit:
        KNOWN_TIME_UNITS = {
            'day','days','month','months','year','years','yr','yrs',
            'hour','hours','hr','hrs','min','mins','minute','minutes',
            'second','seconds','s','ms','index','step','steps',
            'sample','samples','observation','observations',''
        }
        if metadata['time_unit'].lower() not in KNOWN_TIME_UNITS:
        "⚠ Unrecognized time unit '{unit}'. Known units: {sorted list}.
           Proceeding anyway."
    """

─── 4. Tests ────────────────────────────────────────────

Write a self-contained test script. Create all temp files inside it.
Print PASS or FAIL for each test. At the end print total pass/fail count.

TEST 1 — Standard 2-column CSV with units:
  Content: "time [months], sunspot_number [count]\n1,12.4\n2,15.1\n3,14.8"
  Assert: time == [1,2,3]
  Assert: values == [12.4, 15.1, 14.8]
  Assert: meta['time_unit'] == 'months'
  Assert: meta['value_unit'] == 'count'
  Assert: meta['time_label'] == 'time (months)'
  Assert: meta['value_label'] == 'sunspot_number (count)'
  Assert: meta['n_points'] == 3

TEST 2 — Unitless observable (empty brackets):
  Content: "time [days], normalized_flux []\n0,0.9987\n1,1.0023"
  Assert: meta['value_unit'] == ''
  Assert: meta['value_label'] == 'normalized_flux'   ← no parentheses
  Assert: meta['time_label'] == 'time (days)'

TEST 3 — No space before bracket:
  Content: "time[yr], co2[ppm]\n1958,315.2\n1959,315.9"
  Assert: meta['time_unit'] == 'yr'
  Assert: meta['value_unit'] == 'ppm'
  Assert: meta['value_label'] == 'co2 (ppm)'

TEST 4 — Multi-column CSV via read_csv_multi:
  Content: "time [yr], co2 [ppm], temperature [°C]\n1958,315.2,14.1\n1959,315.9,14.0"
  Assert: len(result) == 2
  Assert: result[0][2]['value_name'] == 'co2'
  Assert: result[1][2]['value_name'] == 'temperature'
  Assert: result[0][2]['value_unit'] == 'ppm'

TEST 5 — FileNotFoundError:
  Call read_csv('nonexistent_file.csv')
  Assert raises FileNotFoundError
  Assert error message contains '✗ File not found'

TEST 6 — Single column CSV raises ValueError:
  Content: "time [months]\n1\n2\n3"
  Assert raises ValueError
  Assert message contains '✗'

TEST 7 — Non-numeric data raises ValueError:
  Content: "time [days], value []\nabc,1.0\n2,3.0"
  Assert raises ValueError with clear message

TEST 8 — V1 warning: non-monotonic time:
  Content: "time [days], flux []\n3,1.0\n1,2.0\n2,3.0"
  Capture stdout. Assert '⚠' in output and 'not strictly increasing' in output.

TEST 9 — V2 warning: negative counts:
  Content: "time [days], count [count]\n1,-5.0\n2,3.0"
  Capture stdout. Assert '⚠' in output and 'Negative values' in output.

TEST 10 — V4 warning: unrecognized time unit:
  Content: "time [parsecs], value []\n1,1.0\n2,2.0"
  Capture stdout. Assert '⚠' in output and 'Unrecognized time unit' in output.
  Assert data still loaded correctly (non-fatal).

Run all 10 tests. Report: "X/10 tests passed."
```

---

## Prompt 2 ✓ — core/msd.py + utils/preprocess.py
```
Read CLAUDE.md before starting.

Task: Implement core/msd.py and utils/preprocess.py.

─── core/msd.py ─────────────────────────────────────────

def compute_msd(
    x: np.ndarray,
    max_lag: int = None,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical Mean Square Displacement of a 1D time series.

    MSD(Δ) = (1 / (N-Δ)) · Σᵢ [x(i+Δ) - x(i)]²

    Parameters
    ----------
    x : array-like (1D)
        Fluctuating observable. Accepts np.ndarray, pd.Series, list.
        Converted internally. Must be 1D with at least 10 points.
    max_lag : int, optional
        Maximum lag. Defaults to len(x) // 2.
    normalize : bool
        If True, divide all MSD values by MSD[0].

    Returns
    -------
    lags : np.ndarray[int]    shape (max_lag,)
    msd  : np.ndarray[float]  shape (max_lag,)

    Example
    -------
    >>> time, values, meta = wn.read_csv('sunspot.csv')
    >>> lags, msd = wn.compute_msd(values)
    """

Internal only — not public:

def _to_1d_array(x) -> np.ndarray:
    """
    Convert any array-like to 1D float np.ndarray.
    Handles: np.ndarray, pd.Series, list, tuple.
    Replaces isolated NaN with linear interpolation (np.interp).
    Raises clear ValueError if:
      - result is not 1D: "✗ Input must be 1D. Got shape {shape}."
      - fewer than 10 points: "✗ Need at least 10 data points. Got {n}."
      - more than 50% NaN: "✗ Too many missing values ({pct:.0f}% NaN)."
    """

─── utils/preprocess.py ─────────────────────────────────

These are OPTIONAL helpers. The researcher calls them manually before
analyze() if needed. The pipeline never calls them automatically.

def detrend(
    values: np.ndarray,
    method: str = 'linear',
    poly_order: int = 1,
) -> np.ndarray:
    """
    Remove trend to extract fluctuations.

    Parameters
    ----------
    values : 1D array-like
    method : str
        'linear'     subtract degree-1 polynomial fit
        'polynomial' subtract degree poly_order polynomial fit
        'mean'       subtract global mean only
    poly_order : int
        Only used when method='polynomial'. Ignored otherwise.

    Returns
    -------
    fluctuations : np.ndarray (same length as values)

    Example
    -------
    >>> time, values, meta = wn.read_csv('co2.csv')
    >>> fluct = wn.detrend(values, method='polynomial', poly_order=2)
    >>> result = wn.analyze(fluct, model='cosine', label='CO2 fluctuations')
    """

def normalize(
    values: np.ndarray,
    method: str = 'zscore',
) -> np.ndarray:
    """
    Normalize a time series.
    method: 'zscore'  subtract mean, divide std
            'minmax'  scale to [0,1]
            'mean'    divide by mean only
    """

def smooth(
    values: np.ndarray,
    window: int = 5,
    method: str = 'moving_average',
) -> np.ndarray:
    """
    Smooth a time series. Output same length as input.
    method: 'moving_average'  uniform kernel
            'gaussian'        Gaussian kernel (sigma = window/4)
    """

─── Tests ───────────────────────────────────────────────

Print PASS/FAIL for each. Report X/10 at end.

TEST 1 — Brownian MSD grows linearly:
  x = np.cumsum(np.random.randn(500))
  lags, msd = compute_msd(x)
  Fit line to msd vs lags. Assert R² of linear fit > 0.9.

TEST 2 — Superdiffusive MSD grows faster than linear:
  Generate data where increments scale as t^0.4 (μ≈1.5 ish)
  lags, msd = compute_msd(x)
  Assert msd[-1] / msd[len(msd)//2] > 2 * (max_lag / (max_lag//2))

TEST 3 — normalize=True makes MSD[0] == 1.0:
  _, msd = compute_msd(x, normalize=True)
  Assert abs(msd[0] - 1.0) < 1e-10

TEST 4 — pd.Series input works:
  import pandas as pd
  s = pd.Series(np.random.randn(100))
  lags, msd = compute_msd(s)
  Assert len(lags) == 50

TEST 5 — list input works:
  lags, msd = compute_msd([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                            7.0, 8.0, 9.0, 10.0, 11.0])
  Assert len(lags) > 0

TEST 6 — 2D array raises clear error:
  Assert raises ValueError with '✗' and 'must be 1D'

TEST 7 — < 10 points raises clear error:
  Assert raises ValueError with '✗' and '10'

TEST 8 — detrend linear removes trend:
  x = np.linspace(0, 100, 200) + np.random.randn(200) * 0.1
  fluct = detrend(x, method='linear')
  Assert abs(np.polyfit(np.arange(200), fluct, 1)[0]) < 0.01

TEST 9 — normalize zscore gives mean≈0, std≈1:
  x = np.random.randn(100) * 5 + 20
  n = normalize(x, method='zscore')
  Assert abs(np.mean(n)) < 0.01
  Assert abs(np.std(n) - 1.0) < 0.01

TEST 10 — smooth output same length as input:
  x = np.random.randn(100)
  s = smooth(x, window=7)
  Assert len(s) == 100
```

---

## Prompt 3 ✓ — core/models.py
```
Read CLAUDE.md before starting.

Task: Implement core/models.py — all 16 model formulas.

─── Priority models (fully implement) ───────────────────

For each model implement both MSD and PDF functions.

PDF formula (same for all models):
    P(dx; T) = 1/√(2π·σ²) · exp(-dx²/(2σ²))
    where σ² = msd_<name>(T, *params)
If σ² ≤ 0 or is nan: return array of np.nan (never raise).

1. msd_cosine(T, mu, nu) → float | np.ndarray
   Formula: √π · Γ(μ) · cos(νT/2) · J_{μ-1/2}(νT/2) / (T/ν)^(1/2-μ)
   scipy: gamma(mu), jv(mu-0.5, nu*T/2)
   Return np.nan (not raise) when cos(νT/2) ≤ 0.

2. msd_exponential(T, mu, beta) → float | np.ndarray
   Formula: Γ(μ) · β^(-μ) · T^(μ-1) · e^(-β/T)

3. msd_sine(T, mu, nu) → float | np.ndarray
   Table 3.1 row 9.
   Formula: same structure as cosine but replace cos(νT/2) with sin(νT/2).
   Return np.nan when sin(νT/2) ≤ 0.

4. msd_fbm(T, H) → float | np.ndarray
   Formula: T^(2H)
   H = 0.5 → Brownian (MSD = T)

For each: pdf_cosine, pdf_exponential, pdf_sine, pdf_fbm
    dx can be a scalar or np.ndarray.

─── Extended models (stub only) ─────────────────────────

For rows 2,3,5,6,7,8,11,12,13,14,15,16 create stub functions:

def msd_<name>(T, *params):
    """
    [Table 3.1 row N] — <description from table>
    Formula: <exact formula>
    Parameters: <param list>
    Status: Not yet implemented.
    """
    raise NotImplementedError(
        "Model '<name>' is not yet implemented.\n"
        "Available models: cosine, exponential, sine, fbm\n"
        "Run wn.list_models() to see all models and their status."
    )

─── MODELS registry ─────────────────────────────────────

MODELS = {
    'cosine': {
        'msd':         msd_cosine,
        'pdf':         pdf_cosine,
        'params':      ['mu', 'nu'],
        'n_params':    2,
        'row':         10,
        'status':      'available',
        'description': 'Power-law memory with cosine modulation',
        'reference':   'Table 3.1 row 10',
    },
    'exponential': {
        'msd':         msd_exponential,
        'pdf':         pdf_exponential,
        'params':      ['mu', 'beta'],
        'n_params':    2,
        'row':         4,
        'status':      'available',
        'description': 'Power-law memory with exponential modulation',
        'reference':   'Table 3.1 row 4',
    },
    'sine': { ... row 9 ... },
    'fbm':  { ... row 1 ... },
    # All 12 extended stubs with status='not_implemented'
    # Include correct row numbers, param names, descriptions
}

def get_model(name: str) -> dict:
    """
    Return model dict.
    If status == 'not_implemented': raise NotImplementedError with
      message listing all available models.
    If name not in MODELS: raise ValueError listing all 16 names.
    """

def list_models() -> None:
    """
    Print formatted table:
    ┌──────────────────┬─────┬──────────────┬───────────────┬──────────────────────────────────────┐
    │ Name             │ Row │ Params       │ Status        │ Description                          │
    ├──────────────────┼─────┼──────────────┼───────────────┼──────────────────────────────────────┤
    │ cosine           │  10 │ μ, ν         │ ✓ available   │ Power-law with cosine modulation     │
    │ exponential      │   4 │ μ, β         │ ✓ available   │ Power-law with exponential mod.      │
    │ exp_whittaker    │   5 │ μ, β, ν      │ ✗ stub        │ Power-law with Whittaker function    │
    ...
    """

─── Tests ───────────────────────────────────────────────

Print PASS/FAIL. Report X/10 at end.

TEST 1: msd_cosine(10.0, 1.2, 0.05) → positive finite float
TEST 2: msd_cosine where νT/2 > π/2 → np.nan, no exception
TEST 3: msd_exponential(10.0, 1.15, 0.1) → positive finite float
TEST 4: msd_fbm(4.0, 0.5) → value close to 4.0 (Brownian: MSD = T)
TEST 5: msd_fbm(T_array, 0.5) → array all close to T_array
TEST 6: pdf_cosine integrates to ≈ 1.0
        dx = np.linspace(-50, 50, 10000)
        pdf = pdf_cosine(dx, T=10, mu=1.2, nu=0.05)
        Assert abs(np.trapz(pdf[np.isfinite(pdf)],
                            dx[np.isfinite(pdf)]) - 1.0) < 0.05
TEST 7: pdf_exponential integrates to ≈ 1.0 (same method)
TEST 8: get_model('cosine') returns dict with 'msd', 'pdf', 'params' keys
TEST 9: get_model('exp_whittaker') raises NotImplementedError
TEST 10: get_model('nonexistent') raises ValueError
         list_models() runs without error and prints 16 rows
```

---

## Prompt 4 ✓ — core/fitting.py
```
Read CLAUDE.md before starting.

Task: Implement core/fitting.py — parameter extraction with 95% CIs.

─── FitResult dataclass ─────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class FitResult:
    params:               dict   # {'mu': 1.15, 'nu': 0.003, 'N': 2.1}
    std_errors:           dict   # {'mu': 0.02, 'nu': 0.0004, 'N': 0.1}
    confidence_intervals: dict   # {'mu': (1.111, 1.189), ...}
    r_squared:            float
    model:                str
    lags_used:            np.ndarray
    msd_fitted:           np.ndarray  # N·msd_theory at lags_used

    def summary(self) -> str:
        """
        Return formatted string. Example for cosine model:

        ┌─────────────────────────────────────────┐
        │  Fit Summary                            │
        │  Model  : cosine          R² = 0.9823  │
        ├─────────────────────────────────────────┤
        │  μ    = 1.2341 ± 0.0082                │
        │         95% CI: (1.218, 1.250)          │
        │  ν    = 0.0082 ± 0.0003                 │
        │         95% CI: (0.008, 0.009)          │
        │  N    = 2.4312 ± 0.0441                 │
        │         95% CI: (2.345, 2.518)          │
        └─────────────────────────────────────────┘
        """

─── fit_msd ─────────────────────────────────────────────

def fit_msd(
    lags: np.ndarray,
    msd_empirical: np.ndarray,
    model: str = 'cosine',
    p0: list = None,
    bounds: tuple = None,
    max_lag_fraction: float = 1.0,
) -> 'FitResult | None':
    """
    Fit N · msd_theory(T, *params) to empirical MSD.
    N is the normalization scalar — always included.

    Default p0 (physical params only, N=1.0 appended internally):
      cosine:      [1.2, 0.01]
      exponential: [1.2, 0.1]
      sine:        [1.2, 0.01]
      fbm:         [0.6]

    Default bounds (N bounds (0, inf) appended internally):
      cosine:      ([0.01, 1e-9],  [5.0, 10.0])
      exponential: ([0.01, 1e-9],  [5.0, 100.0])
      sine:        ([0.01, 1e-9],  [5.0, 10.0])
      fbm:         ([0.01],        [2.0])

    Confidence intervals:
      std_error = sqrt(diag(pcov))
      95% CI = (param - 1.96*se, param + 1.96*se)

    R² = 1 - SS_res / SS_tot

    On R² < 0.5 after successful fit, prints:
      "⚠ Low R² ({r2:.4f}). Consider trying other models: {alternatives}"

    On fitting failure, prints:
      "✗ Fitting failed for model '{model}': {reason}"
      Returns None. Never raises.

    Parameters listed in FitResult.params in this order:
      cosine/sine:   mu, nu, N
      exponential:   mu, beta, N
      fbm:           H, N
    """

─── Tests ───────────────────────────────────────────────

Print PASS/FAIL. Report X/8 at end.

TEST 1: Recover μ from known cosine MSD.
  T = np.linspace(1, 100, 200)
  true_mu, true_nu = 1.3, 0.008
  msd_true = msd_cosine(T, true_mu, true_nu)
  result = fit_msd(T, msd_true, model='cosine')
  Assert result is not None
  Assert abs(result.params['mu'] - true_mu) < 0.1

TEST 2: Recover μ from known exponential MSD.
  true_mu, true_beta = 1.15, 0.1
  msd_true = msd_exponential(T, true_mu, true_beta)
  result = fit_msd(T, msd_true, model='exponential')
  Assert abs(result.params['mu'] - true_mu) < 0.1

TEST 3: Confidence intervals are correct structure.
  Assert 'mu' in result.confidence_intervals
  lo, hi = result.confidence_intervals['mu']
  Assert lo < result.params['mu'] < hi

TEST 4: FitResult.summary() runs and returns non-empty string.

TEST 5: Recover H ≈ 0.5 from Brownian MSD.
  msd_linear = T * 1.0
  result = fit_msd(T, msd_linear, model='fbm')
  Assert abs(result.params['H'] - 0.5) < 0.1

TEST 6: max_lag_fraction=0.5 uses only first half.
  result = fit_msd(T, msd_true, model='cosine', max_lag_fraction=0.5)
  Assert len(result.lags_used) == len(T) // 2

TEST 7: All-zero MSD returns None with printed ✗ message.
  msd_zero = np.zeros(100)
  result = fit_msd(np.arange(1,101), msd_zero, model='cosine')
  Assert result is None

TEST 8: Wrong model prints ⚠ warning when R² < 0.5.
  Use exponential MSD data but fit with fbm model.
  Capture stdout. Assert '⚠' in output and 'R²' in output.
```

---

## Prompt 5 ✓ — analysis/pipeline.py
```
Read CLAUDE.md before starting.

Task: Implement analysis/pipeline.py.

─── AnalysisResult ──────────────────────────────────────

@dataclass
class AnalysisResult:
    label:        str
    model:        str
    time:         np.ndarray
    values:       np.ndarray
    lags:         np.ndarray
    msd_empirical: np.ndarray
    fit:          FitResult
    metadata:     dict

    @property
    def mu(self) -> float:
        """μ for all models. H for fbm."""
        return self.fit.params.get('mu', self.fit.params.get('H'))

    @property
    def regime(self) -> str:
        """
        For all non-fbm models, based on μ:
          μ < 0.95:   'subdiffusive'
          0.95-1.05:  'near-Brownian'
          1.05-2.0:   'superdiffusive'
          μ > 2.0:    'hyperballistic'
        For fbm, based on H:
          H < 0.475:  'subdiffusive'
          0.475-0.525:'near-Brownian'
          H > 0.525:  'superdiffusive'
        """

    def summary(self):
        """
        Print:
        ══════════════════════════════════════════════════
          SWNA Result: {label}
        ══════════════════════════════════════════════════
          Source  : {source_file or 'array input'}
          Model   : {model}
          N points: {n_points}
          R²      : {r_squared:.4f}
        ──────────────────────────────────────────────────
          {param1} = {value} ± {se}
                  95% CI: ({lo}, {hi})
          {param2} = {value} ± {se}
                  95% CI: ({lo}, {hi})
          N       = {value} ± {se}
        ──────────────────────────────────────────────────
          Regime  : {REGIME in CAPS}
          {one-line plain-language description of regime}
        ══════════════════════════════════════════════════

        Regime descriptions:
          subdiffusive:   'Past fluctuations resist change — slow spreading.'
          near-Brownian:  'Weak or no memory — near-ordinary random walk.'
          superdiffusive: 'Past fluctuations reinforce future ones — fast spreading.'
          hyperballistic: 'Extreme memory — strongly persistent dynamics.'
        """

    def plot_msd(self, **kwargs):
        from ..viz.publish import plot_msd_fit
        return plot_msd_fit(self, **kwargs)

    def plot_pdf(self, T=None, **kwargs):
        from ..viz.publish import plot_pdf
        return plot_pdf(self, T=T, **kwargs)

    def plot_all(self, **kwargs):
        from ..viz.publish import plot_all
        return plot_all(self, **kwargs)

    def to_dict(self) -> dict:
        """
        Return flat dict with ALL of these keys (no nesting):
        label, source_file, model, n_points, r_squared, regime,
        mu (or H for fbm),
        param2_name (nu/beta/—), param2_value, param2_se,
        mu_se, mu_ci_low, mu_ci_high,
        param2_ci_low, param2_ci_high,
        N_value, N_se
        """

    def to_csv(self, path: str):
        """Write to_dict() as single-row CSV with header."""

    def to_excel(self, path: str):
        """Write to_dict() as Excel. Print notice if openpyxl missing."""

─── analyze ─────────────────────────────────────────────

def analyze(
    source: 'str | np.ndarray | list',
    model: str = 'cosine',
    time: np.ndarray = None,
    label: str = '',
    max_lag: int = None,
    max_lag_fraction: float = 1.0,
    p0: list = None,
    bounds: tuple = None,
    verbose: bool = True,
) -> AnalysisResult:
    """
    Run the full SWNA pipeline.

    Parameters
    ----------
    source : str or array-like
        str → path to whitenoise-format CSV. Units/labels auto-read.
        array-like → 1D data. Provide label manually.
    model : str
        See wn.list_models() for all options.
    time : array-like, optional
        Time axis. Only used when source is array-like.
        Ignored (overridden by CSV) when source is str.
    label : str
        Auto-set from CSV value_name when source is CSV.
    max_lag : int, optional
        Defaults to len(x)//2.
    max_lag_fraction : float
        Fraction of lags used in fitting. Default 1.0.
    verbose : bool
        If True: print ✓ progress lines and final summary.

    Returns
    -------
    AnalysisResult

    Raises
    ------
    ValueError  if model is not in MODELS registry (with list of valid names)
    ValueError  if source array-like has no label provided

    Examples
    --------
    >>> # From CSV (recommended for research)
    >>> result = wn.analyze('sunspot.csv', model='exponential')
    >>> result.summary()
    >>> result.plot_all()

    >>> # From array
    >>> data = np.loadtxt('raw.txt')
    >>> fluct = wn.detrend(data)
    >>> result = wn.analyze(fluct, model='cosine', label='My System')
    """

    # When source is str (CSV path):
    #   call read_csv() → get time, values, metadata
    #   label = metadata['value_name'] if label==''
    # When source is array-like:
    #   metadata = {'source_file': 'array input',
    #               'time_label': 'index', 'value_label': label,
    #               'n_points': len(x)}
    #   Raise ValueError if label == '': "✗ Please provide label= when
    #   passing an array. Example: wn.analyze(data, label='My System')"

    # Progress prints (only if verbose=True):
    #   "✓ Loaded {n_points} points from {source_file}"
    #   "✓ Computing MSD..."
    #   "✓ Fitting {model} model..."
    #   "✓ Done.  R² = {r2:.4f}  μ = {mu:.4f}  regime: {regime}"

─── Tests ───────────────────────────────────────────────

Print PASS/FAIL. Report X/8 at end.

TEST 1: analyze() from CSV path.
  Create temp CSV. Call analyze(csv_path, model='exponential').
  Assert metadata['source_file'] == csv_path
  Assert isinstance(result.fit, FitResult)

TEST 2: analyze() from numpy array.
  data = np.random.randn(300)
  result = wn.analyze(data, model='cosine', label='Test')
  Assert result.label == 'Test'
  Assert result.metadata['source_file'] == 'array input'

TEST 3: analyze() from CSV → label auto-set from value column name.
  result = wn.analyze(csv_path, model='cosine')
  Assert result.label == meta['value_name']

TEST 4: Superdiffusive synthetic data → correct regime.
  Generate data with clear superdiffusive scaling.
  result = wn.analyze(data, model='cosine', label='Test')
  Assert result.regime in ['superdiffusive', 'hyperballistic']

TEST 5: to_dict() has required keys.
  d = result.to_dict()
  for key in ['label','model','mu','r_squared','regime','n_points']:
      Assert key in d

TEST 6: to_csv() creates readable file.
  result.to_csv('/tmp/test_result.csv')
  import pandas as pd
  df = pd.read_csv('/tmp/test_result.csv')
  Assert 'mu' in df.columns

TEST 7: verbose=False → no printed output.
  import io, contextlib
  f = io.StringIO()
  with contextlib.redirect_stdout(f):
      result = wn.analyze(data, model='cosine', label='T', verbose=False)
  Assert f.getvalue() == ''

TEST 8: Invalid model name → clear ValueError.
  try: wn.analyze(data, model='magic', label='T')
  except ValueError as e:
      Assert '✗' in str(e)
      Assert 'cosine' in str(e)   # lists valid names
```

---

## Prompt 6 ✓ — compare.py + batch.py
```
Read CLAUDE.md before starting.

Task: Implement analysis/compare.py and analysis/batch.py.

─── compare.py ──────────────────────────────────────────

@dataclass
class ComparisonResult:
    results: list   # list of AnalysisResult

    def summary_table(self) -> pd.DataFrame:
        """
        DataFrame with columns (sorted by mu ascending):
        label | source_file | model | mu | mu_ci_low | mu_ci_high |
        param2_name | param2 | r_squared | regime | n_points
        param2 = nu for cosine/sine, beta for exponential, H for fbm
        """

    def plot_comparison(self, **kwargs):
        from ..viz.publish import plot_comparison
        return plot_comparison(self, **kwargs)

    def to_csv(self, path: str):
        self.summary_table().to_csv(path, index=False)
        print(f"✓ Saved comparison table to {path}")

    def to_excel(self, path: str):
        try:
            self.summary_table().to_excel(path, index=False)
            print(f"✓ Saved comparison table to {path}")
        except ImportError:
            print("⚠ openpyxl not installed. Run: pip install openpyxl")


def compare(
    datasets: 'dict | list',
    model: 'str | dict' = 'cosine',
    **analyze_kwargs,
) -> ComparisonResult:
    """
    Analyze multiple datasets and compare memory parameters.

    Parameters
    ----------
    datasets : dict or list
        dict: keys=labels (str), values=CSV paths or array-like
              {'System A': 'a.csv', 'System B': array_b}
        list: list of CSV paths — labels from filenames (no extension)

    model : str or dict
        str: same model for all
        dict: per-label model
              {'System A': 'cosine', 'System B': 'exponential'}

    **analyze_kwargs
        Passed to analyze() for each system.

    Returns
    -------
    ComparisonResult

    Example
    -------
    >>> results = wn.compare({
    ...     'Earthquakes': 'earthquake_ph.csv',
    ...     'Sunspots':    'sunspot.csv',
    ... }, model='exponential')
    >>> print(results.summary_table())
    >>> results.plot_comparison()
    """

─── batch.py ────────────────────────────────────────────

def batch_analyze(
    source: 'str | list',
    model: str = 'cosine',
    pattern: str = '*.csv',
    **analyze_kwargs,
) -> ComparisonResult:
    """
    Batch SWNA on multiple CSV files.

    Parameters
    ----------
    source : str or list
        str (folder path): reads all files matching pattern in folder
        str (CSV path):    multi-column CSV — each non-time column
                           analyzed as separate system
        list:              explicit list of CSV file paths

    model : str
        Applied to all systems.

    pattern : str
        Glob pattern for folder mode. Default '*.csv'.

    Returns
    -------
    ComparisonResult

    Examples
    --------
    >>> # Folder of CSVs
    >>> results = wn.batch_analyze('data/xray_binaries/', model='cosine')
    >>> results.summary_table()

    >>> # Explicit list
    >>> results = wn.batch_analyze(
    ...     ['chile.csv', 'japan.csv', 'ph.csv'],
    ...     model='exponential'
    ... )

    >>> # Multi-column CSV
    >>> results = wn.batch_analyze('climate.csv', model='cosine')
    """

    # Detection logic:
    # 1. source is list → Mode B
    # 2. source is str ending in .csv → check if multi-column via read_csv_multi
    # 3. source is str not ending in .csv or is directory → Mode A (folder)

─── Tests ───────────────────────────────────────────────

Print PASS/FAIL. Report X/9 at end.

TEST 1: compare() with dict of 3 arrays.
  synthetic = {f'System {i}': np.random.randn(200) for i in range(3)}
  result = wn.compare(synthetic, model='cosine')
  Assert len(result.results) == 3

TEST 2: compare() summary_table has correct columns.
  df = result.summary_table()
  for col in ['label','model','mu','r_squared','regime']:
      Assert col in df.columns

TEST 3: summary_table sorted by mu ascending.
  mus = result.summary_table()['mu'].values
  Assert all(mus[i] <= mus[i+1] for i in range(len(mus)-1))

TEST 4: compare() with dict of 2 CSV file paths.
  Create 2 temp CSVs. Pass as dict.
  Assert len(result.results) == 2
  Assert result.results[0].metadata['source_file'] is a real path

TEST 5: compare() with per-dataset model dict.
  result = wn.compare({'A': arr_a, 'B': arr_b},
                      model={'A': 'cosine', 'B': 'exponential'})
  Assert result.results[0].model == 'cosine'
  Assert result.results[1].model == 'exponential'

TEST 6: batch_analyze() with temp folder of 3 CSVs.
  Create temp dir, write 3 CSVs, call batch_analyze(temp_dir).
  Assert len(result.results) == 3

TEST 7: batch_analyze() with explicit list of 2 CSV paths.
  Assert len(result.results) == 2

TEST 8: batch_analyze() with multi-column CSV.
  Write CSV: "time [yr], co2 [ppm], temp [°C]\n..."
  result = wn.batch_analyze(multi_csv_path, model='cosine')
  Assert len(result.results) == 2  (co2 + temp, not time)

TEST 9: to_csv() writes file with correct row count.
  result.to_csv('/tmp/comparison.csv')
  df = pd.read_csv('/tmp/comparison.csv')
  Assert len(df) == len(result.results)
```

---

## Prompt 7 ✓ — viz/explore.py + viz/publish.py
```
Read CLAUDE.md before starting.

Task: Implement viz/explore.py and viz/publish.py.

─── Shared palette (define at top of BOTH files) ────────

EMPIRICAL    = '#2C3E50'
THEORETICAL  = '#E74C3C'
HISTOGRAM    = '#5DADE2'
ACCENT       = '#F39C12'
SUBTLE       = '#BDC3C7'
REGIME_COLORS = {
    'subdiffusive':   '#3498DB',
    'near-Brownian':  '#2ECC71',
    'superdiffusive': '#F39C12',
    'hyperballistic': '#E74C3C',
}
FONT = {'title': 13, 'label': 11, 'tick': 9, 'caption': 9}

─── viz/explore.py ──────────────────────────────────────

All functions: ax=None, show=True, return Axes.
figsize applies when ax is None (standalone).

1. plot_series(time, values, title='', xlabel='Time', ylabel='Value',
               ax=None, show=True)
   Line: EMPIRICAL, lw=1.0, alpha=0.85
   Dashed zero line: SUBTLE, lw=0.8, linestyle='--'
   Grid: alpha=0.25, linestyle='--'
   figsize=(10, 3.5)

2. plot_msd_raw(lags, msd, title='', xlabel='Lag', ylabel='MSD',
                ax=None, show=True)
   Scatter: EMPIRICAL, s=15, alpha=0.7, label='Empirical MSD'
   Dashed reference: linear fit to first 10% of lags, SUBTLE, lw=1.2
   Label reference: 'Linear reference (Brownian)'
   figsize=(7, 4)

3. plot_histogram(values, bins=40, title='', xlabel='Value',
                  ax=None, show=True)
   Bars: HISTOGRAM, alpha=0.7, edgecolor='white'
   Vertical dashed line at np.mean(values): ACCENT, lw=1.5
   Text box upper-right: f'mean = {mean:.3f}\nstd = {std:.3f}'
   ylabel='Count'
   figsize=(6, 4)

4. plot_acf(values, max_lag=50, title='', xlabel='Lag',
            ylabel='Autocorrelation', ax=None, show=True)
   ACF computation:
     mean_x = np.mean(values)
     var_x = np.var(values)
     acf = [np.mean((values[:-k]-mean_x)*(values[k:]-mean_x))/var_x
            for k in range(1, max_lag+1)]
     acf = [1.0] + acf   (ACF[0] = 1 by definition)
   Bar chart: EMPIRICAL, alpha=0.7, width=0.8
   Confidence bounds ±1.96/√N: ACCENT, linestyle='--', lw=1.2
   Zero line: 'black', lw=0.8
   Annotation in upper-right text box:
     acf[1] > 0.3:  '↑ Positive memory detected'
     acf[1] < -0.3: '↓ Negative memory detected'
     else:          '~ Weak or no memory'
   figsize=(8, 3.5)

5. plot_overview(time, values, metadata=None, show=True) → Figure
   2×2 subplot grid. figsize=(14, 8).
   [0,0] plot_series    — uses metadata['time_label'], metadata['value_label']
                           if metadata provided, else 'Time', 'Value'
   [0,1] plot_histogram — uses metadata['value_label'] as xlabel
   [1,0] plot_msd_raw   — compute MSD internally: lags,msd=compute_msd(values)
   [1,1] plot_acf
   Supertitle: metadata['value_name'] if metadata else 'Data Overview'
   Returns Figure object.

─── viz/publish.py ──────────────────────────────────────

Publication-quality. No top/right spines on all plots.
ax.spines[['top','right']].set_visible(False)
All functions: ax=None, show=True, return Axes (or Figure for plot_all).

1. plot_msd_fit(result: AnalysisResult, ax=None, show=True)
   Scatter (empirical): EMPIRICAL, s=12, alpha=0.6, label='Empirical MSD'
   Line (theoretical):  THEORETICAL, lw=2.5, label='Theoretical fit'
   Parameter annotation box lower-right:
     'μ = {mu:.4f}\n{param2_name} = {val:.2e}\nR² = {r2:.4f}'
     bbox: facecolor='white', edgecolor=SUBTLE, alpha=0.9
   xlabel: result.metadata.get('time_label', 'Lag')
   ylabel: 'MSD'
   title: bold, f'{result.label} — MSD Fit'
   figsize=(7, 4.5)

2. plot_pdf(result: AnalysisResult, T=None, lag=None,
            ax=None, show=True)
   Auto T if None:
     cosine/sine:   T = π/(2·ν)  capped at len(values)//4
     exponential:   T = len(values) // 10
     fbm:           T = len(values) // 4
   lag = max(1, int(T))
   Compute displacement histogram:
     displacements = values[lag:] - values[:-lag]
     Use np.histogram with density=True
   Bars (empirical): HISTOGRAM, alpha=0.55, label='Empirical displacements'
   Line (theoretical PDF): THEORETICAL, lw=2.5, label=f'PDF (T={T:.1f})'
   xlabel: f'Displacement Δx  ({result.metadata.get("value_unit","")})'
            (omit unit parentheses if unit is '')
   ylabel: 'Probability Density'
   figsize=(7, 4.5)

3. plot_comparison(comparison: ComparisonResult, ax=None, show=True)
   Horizontal bar chart of μ values.
   One bar per system, color from REGIME_COLORS[regime].
   Error bars: ± (ci_high - ci_low)/2
   Reference vertical dashed line at μ=1.0: SUBTLE, lw=1.2
   Label each bar: f'  {label}  (μ={mu:.4f})'  (text inside bar or next to it)
   xlabel: 'Memory Parameter μ'
   ylabel: ''  (system names are y tick labels)
   figsize: (8, 0.7*n_systems + 2)

4. plot_all(result: AnalysisResult, show=True) → Figure
   Three-panel: [time series | MSD fit | PDF]
   figsize=(15, 4.5)
   Supertitle:
     f'{result.label}  ·  {result.model} model  ·  μ = {mu:.4f}  ·  {regime.upper()}'
   Left panel:
     wn.plot_series(result.time, result.values,
                    xlabel=result.metadata.get('time_label','index'),
                    ylabel=result.metadata.get('value_label','value'))
   Middle panel: plot_msd_fit(result, ax=ax2)
   Right panel: plot_pdf(result, ax=ax3)
   Returns Figure.

─── Tests ───────────────────────────────────────────────

Create synthetic AnalysisResult for testing. Print PASS/FAIL.
Report X/10 at end.

TEST 1:  plot_series saves PNG without error
TEST 2:  plot_msd_raw saves PNG without error
TEST 3:  plot_histogram saves PNG without error
TEST 4:  plot_acf saves PNG without error
TEST 5:  plot_overview returns Figure, saves PNG
TEST 6:  plot_overview with metadata → supertitle == metadata['value_name']
TEST 7:  plot_msd_fit saves PNG, has annotation text box
TEST 8:  plot_pdf saves PNG without error
TEST 9:  plot_comparison with 4 systems → 4 horizontal bars
TEST 10: plot_all returns Figure; publish plots have no top/right spines
         Check: ax.spines['top'].get_visible() == False
```

---

## Prompt 8 ✓ — __init__.py + io/export.py + setup + integration test
```
Read CLAUDE.md before starting.

Task: Wire everything into __init__.py, implement io/export.py,
write setup files, and run the full integration test.

─── io/export.py ────────────────────────────────────────

def to_csv(result, path: str):
    """
    Save AnalysisResult or ComparisonResult to CSV.
    AnalysisResult  → single-row CSV from result.to_dict()
    ComparisonResult → multi-row CSV from result.summary_table()
    Print: "✓ Saved to {path}"
    """

def to_excel(result, path: str):
    """Same as to_csv but .xlsx format.
    Print: "⚠ openpyxl not installed. Run: pip install openpyxl"
    if missing."""

def summary_table(results: list) -> pd.DataFrame:
    """
    Build summary DataFrame from list of AnalysisResult.
    Equivalent to ComparisonResult(results).summary_table()
    Useful for ad-hoc aggregation.
    """

─── __init__.py ─────────────────────────────────────────

# I/O
from .io.reader  import read_csv, read_csv_multi
from .io.export  import to_csv, to_excel, summary_table

# Preprocessing (optional)
from .utils.preprocess import detrend, normalize, smooth

# Core (advanced users)
from .core.msd     import compute_msd
from .core.models  import MODELS, get_model, list_models
from .core.fitting import fit_msd, FitResult

# Analysis
from .analysis.pipeline import analyze, AnalysisResult
from .analysis.compare  import compare, ComparisonResult
from .analysis.batch    import batch_analyze

# Exploratory plots
from .viz.explore import (
    plot_series, plot_msd_raw, plot_histogram, plot_acf, plot_overview
)

# Publication plots
from .viz.publish import (
    plot_msd_fit, plot_pdf, plot_comparison, plot_all
)

__version__ = '0.2.0'
__author__  = 'Bernido Group, University of San Carlos'

─── setup.py ────────────────────────────────────────────

name             = 'whitenoise-swna'
version          = '0.2.0'
description      = 'Stochastic White Noise Analysis with memory (Hida-Bernido framework)'
install_requires = ['numpy>=1.20', 'scipy>=1.7',
                    'matplotlib>=3.4', 'pandas>=1.3']
extras_require   = {'excel': ['openpyxl>=3.0']}
python_requires  = '>=3.9'
author           = 'Bernido Group, University of San Carlos'
classifiers: Python 3, Scientific/Engineering :: Physics,
             Intended Audience :: Science/Research

─── pyproject.toml ──────────────────────────────────────

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.backends.legacy:build"

─── README.md ───────────────────────────────────────────

Sections:
1. One-paragraph description (SWNA, Hida framework, Bernido group)
2. Installation: pip install whitenoise-swna
3. Quick start — exactly this code block:

   import whitenoise as wn

   # Analyze from CSV
   result = wn.analyze('sunspot.csv', model='exponential')
   result.summary()
   result.plot_all()

   # Compare multiple systems
   comparison = wn.compare({
       'Sunspots':    'sunspot.csv',
       'Earthquakes': 'earthquake_ph.csv',
   }, model='exponential')
   comparison.summary_table()
   comparison.plot_comparison()

   # Batch process a folder
   batch = wn.batch_analyze('data/xray_binaries/', model='cosine')
   batch.to_csv('results.csv')

4. CSV format specification:
   time [months], sunspot_number [count]
   1, 12.4
   2, 15.1
   Unitless: flux []

5. Model table (all 16, ✓ available or ✗ stub)
6. Key functions quick reference
7. Citation block (Bernido & Carpio-Bernido 2015)
8. Contact / contributing

─── Integration test: tests/test_integration.py ─────────

Write and run this full workflow. Print PASS/FAIL for each step.
Report X/10 at end.

import whitenoise as wn, numpy as np, tempfile, os, pandas as pd

# Generate synthetic superdiffusive data for testing
np.random.seed(42)
N = 300
increments = np.random.randn(N) * np.arange(1, N+1)**0.25
synthetic = np.cumsum(increments)

STEP 1 — Create valid CSV and read it:
  Write: "time [months], sunspot_number [count]\n"
         + "\n".join(f"{i},{v}" for i,v in enumerate(synthetic))
  time, values, meta = wn.read_csv(csv_path)
  Assert meta['value_unit'] == 'count'
  Assert meta['time_label'] == 'time (months)'
  Assert meta['value_label'] == 'sunspot_number (count)'
  Assert len(values) == N

STEP 2 — Exploratory plots:
  wn.plot_overview(time, values, metadata=meta, show=False)
  Assert figure was created without error.

STEP 3 — Analyze from CSV:
  result = wn.analyze(csv_path, model='cosine', verbose=True)
  Assert isinstance(result, wn.AnalysisResult)
  Assert 0 < result.mu < 6
  Assert result.regime in ['subdiffusive','near-Brownian',
                            'superdiffusive','hyperballistic']

STEP 4 — Analyze from array:
  result2 = wn.analyze(synthetic, model='cosine', label='Synthetic')
  Assert result2.label == 'Synthetic'

STEP 5 — Summary and publication plots:
  result.summary()
  fig = result.plot_all(show=False)
  Assert fig is not None

STEP 6 — Export single result:
  result.to_csv('/tmp/wn_result.csv')
  df = pd.read_csv('/tmp/wn_result.csv')
  Assert 'mu' in df.columns and 'regime' in df.columns

STEP 7 — Compare:
  comp = wn.compare(
      {'A': synthetic, 'B': synthetic[:200]},
      model='cosine'
  )
  df = comp.summary_table()
  Assert len(df) == 2
  Assert list(df.columns) contains 'mu' and 'regime'

STEP 8 — Batch from folder:
  Create temp folder, write 2 CSVs to it.
  batch = wn.batch_analyze(temp_folder, model='cosine')
  Assert len(batch.results) == 2

STEP 9 — Batch from multi-column CSV:
  Write: "time [yr], co2 [ppm], temp [°C]\n" + data
  batch2 = wn.batch_analyze(multi_csv, model='cosine')
  Assert len(batch2.results) == 2

STEP 10 — Package meta:
  comp.to_csv('/tmp/wn_comparison.csv')
  wn.list_models()
  Assert wn.__version__ == '0.2.0'

Run: pip install -e .
     python tests/test_integration.py
Report final: "X/10 integration steps passed."
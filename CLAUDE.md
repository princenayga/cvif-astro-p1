# CVIF Astro P1 — CLAUDE.md

## Project Overview

DOST-GIA 7 research project at CVIF Jagna Bohol applying the **MMWN/SWNA (Memory Modulated White Noise / Stochastic White Noise Analysis)** framework to TESS PDC-SAP exoplanet transit light curves. The goal is to characterize stochastic noise regimes in transit flux data using Mean Square Displacement (MSD) modelling.

**6 target planets:** CoRoT-01, HATS-13, TrEs-5 (Hot Jupiters) | TOI-4773, WASP-072, WASP-078 (Ultra Hot Jupiters)

---

## Environment

- Python 3.12 via `.venv/` (activate: `.venv/Scripts/activate`)
- Key packages: `lightkurve`, `numpy`, `pandas`, `matplotlib`, `scipy`
- Local whitenoise package: `whitenoise/` (v0.1.0) — add to path with `sys.path.insert(0, 'whitenoise')`
- Run scripts from project root: `python scripts/<script>.py`

---

## Directory Structure

```
cvif-astro-p1/
├── notebooks/                         # Jupyter notebooks
│   ├── selected6_interactive_extraction.ipynb  # Main transit extraction pipeline
│   ├── mu_vs_beta_scatter.ipynb                # SWNA scatter plot analysis
│   └── workshop_notebook.ipynb                 # whitenoise package usage reference
├── scripts/
│   └── run_swna_pipeline.py           # Batch SWNA analysis on all 6 planets
├── data/
│   └── selected_transits/
│       ├── <Planet>/                  # Full extracted CSVs with metadata
│       └── pipeline/
│           └── <Planet>/
│               └── <Planet>_T##.csv  # 2-column clean CSVs: time_min_from_mid, flux
├── results/
│   └── swna/
│       ├── cosine/<Planet>/           # Cosine model diagnostic PNGs
│       ├── exponential/<Planet>/      # Exponential model diagnostic PNGs
│       ├── exponential swna.csv       # Curated exponential fit results (hand-reviewed)
│       ├── swna_summary.csv           # Auto-generated full summary (both models)
│       ├── swna_summary.xlsx          # Excel version of summary
│       ├── mu_vs_beta_scatter.png     # Base scatter plot
│       ├── mu_vs_beta_clusters.png    # Scatter with cluster ellipses highlighted
│       └── mu_vs_beta_HJ_linearfit.png  # HJ zoom with linear fit + R²
└── whitenoise/                        # Local whitenoise v0.1.0 package
```

---

## Key Technical Concepts

### MMWN/SWNA Framework
- **μ (mu):** memory parameter — characterizes noise regime
- **β (beta):** exponential decay rate of MSD — decorrelation timescale
- **ν (nu):** frequency parameter in cosine model
- **N:** normalization scalar

### Transit Extraction (`selected6_interactive_extraction.ipynb`)
- Source: TESS SPOC PDC-SAP 2-min cadence via `lightkurve`
- **3T window:** `[t_mid − 1.5×T14, t_mid + 1.5×T14]`
- **T14:** `(P/π) × arcsin((R_star + R_p) / a)` at b=0
- **NN gap filling (`fill_gaps_nn`):** inserts points at 2-min cadence into real gaps (≤60 min) using nearest-neighbour flux; keeps all original observed points as-is (`is_filled=False`)
- **Empirical midpoint (`find_empirical_midpoint`):** 3-step hybrid — sliding scan (500 candidates) → parabolic refinement → T14 centroid blend
- Output CSVs include `interpolated` column (0=observed, 1=NN-filled)

### whitenoise v0.1.0 API
```python
import sys
sys.path.insert(0, 'whitenoise')
import whitenoise as wn

# Full pipeline from file path (no detrending)
result = wn.analyze('path/to/file.csv', model='cosine', detrend_method=None)
result.summary()
fig = wn.plot_diagnostics(result)

# Low-level API (array input)
from whitenoise.core.msd import compute_msd
from whitenoise.core.fitting import fit_msd
from whitenoise.analysis.pipeline import AnalysisResult

lags, msd_emp = compute_msd(flux)
fit_result = fit_msd(lags, msd_emp, model='exponential', max_lag_fraction=1.0)
result = AnalysisResult(dataset_name='label', model='exponential',
                        fit=fit_result, lags=lags, msd_empirical=msd_emp,
                        values=flux, time=time_min, metadata={})
fig = wn.plot_diagnostics(result)
```

**Important:** `analyze()` accepts only a file path (str), not arrays. On Windows, wrap stdout for UTF-8: `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`.

### Pipeline CSV format
2 columns, no units in header:
```
time_min_from_mid,flux
-45.2,0.9998
...
```

---

## Batch SWNA Script

```bash
python scripts/run_swna_pipeline.py
```

- Reads all `data/selected_transits/pipeline/<Planet>/<Planet>_T##.csv`
- Runs both `cosine` and `exponential` models with `detrend_method=None`
- Saves diagnostic PNGs to `results/swna/<model>/<Planet>/`
- Saves `results/swna/swna_summary.csv` and `swna_summary.xlsx`

---

## Analysis Results (`mu_vs_beta_scatter.ipynb`)

Planet classification:
- **HJ (circle marker):** CoRoT-01, HATS-13, TrEs-5
- **UHJ (triangle marker):** TOI-4773, WASP-072, WASP-078

Key findings from exponential model:
- UHJ planets cluster near **β ≈ 99** — highly persistent, slowly decorrelating flux noise
- WASP-072 is tightest UHJ cluster (μ ≈ 0.75–0.81); WASP-078 has lowest μ (0.41–0.68)
- HJ planets show **inverse μ–β trend** — fitted with linear equation shown in `mu_vs_beta_HJ_linearfit.png`
- TrEs-5 forms a distinct HJ sub-cluster (μ ≈ 0.62–0.74, β ≈ 25–30)

Notebook cells (in order):
1. Imports
2. Load `exponential swna.csv`
3. Base scatter plot → `mu_vs_beta_scatter.png`
4. Cluster-highlighted plot with per-planet ellipses → `mu_vs_beta_clusters.png`
5. HJ zoom + linear fit + R² → `mu_vs_beta_HJ_linearfit.png`

---

## Notes

- The cosine model produces very negative R² values for these transits — structural limitation (cannot follow MSD saturation at high lags). Use exponential model results for analysis.
- `max_lag_fraction=1.0` is used (full N/2 lags), no restriction to 40%.
- No detrending applied to flux before MSD computation (`detrend_method=None`).
- HATS-13 T08 was discarded (excessive baseline gaps); T09 was renamed T08.
- CoRoT-01 has 15 transits (T01–T15); duplicate T16–T30 were deleted.

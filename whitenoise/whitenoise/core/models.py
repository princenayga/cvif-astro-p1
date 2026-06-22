"""
core/models.py — Theoretical MSD and PDF formulas for all SWNA models.

Priority models (fully implemented): cosine, exponential, sine, fbm, dna
Extended models (stubs — NotImplementedError): 12 remaining Table 3.1 rows

References:
  Bernido & Carpio-Bernido (2015), Table 3.1
  Violanda et al. (2019), Phys. Scr. 94, 125006  [dna model]
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, jv

# ── Shared helpers ────────────────────────────────────────────────────────────

def _to_array(T) -> tuple[np.ndarray, bool]:
    """Return (T as 1-D float array, was_scalar)."""
    scalar = np.ndim(T) == 0
    return np.atleast_1d(np.asarray(T, dtype=float)), scalar


def _finalize_msd(val: np.ndarray, scalar: bool):
    """
    Replace non-positive or non-finite entries with nan.
    Return float if scalar input, else ndarray.
    """
    result = np.where(np.isfinite(val) & (val > 0), val, np.nan)
    if scalar:
        return float(result.flat[0])
    return result


def _pdf_from_sigma2(dx, sigma2: float) -> np.ndarray:
    """
    Gaussian PDF evaluated at dx with variance sigma2.
    Always returns ndarray. Returns all-nan if sigma2 is invalid.
    """
    dx_arr = np.atleast_1d(np.asarray(dx, dtype=float))
    if not (np.isfinite(sigma2) and sigma2 > 0):
        return np.full_like(dx_arr, np.nan, dtype=float)
    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma2)
    return norm * np.exp(-dx_arr ** 2 / (2.0 * sigma2))


_NOT_IMPL_MSG = (
    "Model '{name}' is not yet implemented.\n"
    "Available models: cosine, exponential, sine, fbm, dna\n"
    "Run wn.list_models() to see all models and their status."
)


# ── Priority Model: cosine (Table 3.1 row 10) ─────────────────────────────────

def msd_cosine(T, mu: float, nu: float):
    """
    Theoretical MSD for the cosine model (Table 3.1 row 10).

    Formula::

        MSD(T) = sqrt(pi) * Gamma(mu) * cos(nu*T/2)
                 * J_{mu-1/2}(nu*T/2) * (T/nu)^(mu - 1/2)

    Parameters
    ----------
    T : float or np.ndarray
        Time lag(s). Positive values expected.
    mu : float
        Memory parameter.  mu < 1 subdiffusive, mu = 1 Brownian,
        mu > 1 superdiffusive.
    nu : float
        Characteristic frequency (rad per time unit).

    Returns
    -------
    float or np.ndarray
        MSD value(s).  Returns nan where cos(nu*T/2) <= 0 or where
        the result is non-positive / non-finite.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 3.1 row 10.
    """
    T_arr, scalar = _to_array(T)
    arg = nu * T_arr / 2.0
    cos_val = np.cos(arg)

    result = np.full_like(T_arr, np.nan, dtype=float)
    mask = cos_val > 0

    if np.any(mask):
        T_m = T_arr[mask]
        bv = jv(mu - 0.5, arg[mask])
        val = (
            np.sqrt(np.pi) * gamma(mu)
            * cos_val[mask] * bv
            * (T_m / nu) ** (mu - 0.5)
        )
        result[mask] = np.where(np.isfinite(val) & (val > 0), val, np.nan)

    return float(result.flat[0]) if scalar else result


def pdf_cosine(dx, T: float, mu: float, nu: float) -> np.ndarray:
    """
    Theoretical PDF for the cosine model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = MSD_cosine(T, mu, nu))

    Parameters
    ----------
    dx : float or np.ndarray
        Displacement value(s).
    T : float
        Evaluation time (scalar).
    mu, nu : float
        Model parameters.

    Returns
    -------
    np.ndarray
        PDF values.  All-nan if sigma^2 is invalid.
    """
    sigma2 = msd_cosine(float(T), mu, nu)
    return _pdf_from_sigma2(dx, sigma2)


# ── Priority Model: exponential (Table 3.1 row 4) ────────────────────────────

def msd_exponential(T, mu: float, beta: float):
    """
    Theoretical MSD for the exponential model (Table 3.1 row 4).

    Formula::

        MSD(T) = Gamma(mu) * beta^(-mu) * T^(mu - 1) * exp(-beta / T)

    Parameters
    ----------
    T : float or np.ndarray
        Time lag(s). Positive values expected.
    mu : float
        Memory parameter.
    beta : float
        Exponential decay rate (inverse time unit).

    Returns
    -------
    float or np.ndarray
        MSD value(s).  Returns nan for non-positive or non-finite results.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 3.1 row 4.
    """
    T_arr, scalar = _to_array(T)
    val = gamma(mu) * beta ** (-mu) * T_arr ** (mu - 1.0) * np.exp(-beta / T_arr)
    return _finalize_msd(val, scalar)


def pdf_exponential(dx, T: float, mu: float, beta: float) -> np.ndarray:
    """
    Theoretical PDF for the exponential model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = MSD_exponential(T, mu, beta))

    Parameters
    ----------
    dx : float or np.ndarray
    T : float
    mu, beta : float

    Returns
    -------
    np.ndarray
    """
    sigma2 = msd_exponential(float(T), mu, beta)
    return _pdf_from_sigma2(dx, sigma2)


# ── Priority Model: sine (Table 3.1 row 9) ───────────────────────────────────

def msd_sine(T, mu: float, nu: float):
    """
    Theoretical MSD for the sine model (Table 3.1 row 9).

    Formula::

        MSD(T) = sqrt(pi) * Gamma(mu) * sin(nu*T/2)
                 * J_{mu-1/2}(nu*T/2) * (T/nu)^(mu - 1/2)

    Parameters
    ----------
    T : float or np.ndarray
    mu : float
        Memory parameter.
    nu : float
        Characteristic frequency.

    Returns
    -------
    float or np.ndarray
        Returns nan where sin(nu*T/2) <= 0 or result is invalid.

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 3.1 row 9.
    """
    T_arr, scalar = _to_array(T)
    arg = nu * T_arr / 2.0
    sin_val = np.sin(arg)

    result = np.full_like(T_arr, np.nan, dtype=float)
    mask = sin_val > 0

    if np.any(mask):
        T_m = T_arr[mask]
        bv = jv(mu - 0.5, arg[mask])
        val = (
            np.sqrt(np.pi) * gamma(mu)
            * sin_val[mask] * bv
            * (T_m / nu) ** (mu - 0.5)
        )
        result[mask] = np.where(np.isfinite(val) & (val > 0), val, np.nan)

    return float(result.flat[0]) if scalar else result


def pdf_sine(dx, T: float, mu: float, nu: float) -> np.ndarray:
    """
    Theoretical PDF for the sine model.

    P(dx; T) = Gaussian(mean=0, sigma^2 = MSD_sine(T, mu, nu))

    Parameters
    ----------
    dx : float or np.ndarray
    T : float
    mu, nu : float

    Returns
    -------
    np.ndarray
    """
    sigma2 = msd_sine(float(T), mu, nu)
    return _pdf_from_sigma2(dx, sigma2)


# ── Priority Model: fbm (Table 3.1 row 1) ────────────────────────────────────

def msd_fbm(T, H: float):
    """
    Theoretical MSD for fractional Brownian motion (Table 3.1 row 1).

    Formula::

        MSD(T) = T^(2H)

    Parameters
    ----------
    T : float or np.ndarray
    H : float
        Hurst exponent.
        H = 0.5  -> ordinary Brownian motion (MSD = T).
        H > 0.5  -> superdiffusive (persistent).
        H < 0.5  -> subdiffusive (anti-persistent).

    Returns
    -------
    float or np.ndarray

    References
    ----------
    Bernido & Carpio-Bernido (2015), Table 3.1 row 1.
    """
    T_arr, scalar = _to_array(T)
    val = T_arr ** (2.0 * H)
    return _finalize_msd(val, scalar)


def pdf_fbm(dx, T: float, H: float) -> np.ndarray:
    """
    Theoretical PDF for fractional Brownian motion.

    P(dx; T) = Gaussian(mean=0, sigma^2 = T^(2H))

    Parameters
    ----------
    dx : float or np.ndarray
    T : float
    H : float

    Returns
    -------
    np.ndarray
    """
    sigma2 = msd_fbm(float(T), H)
    return _pdf_from_sigma2(dx, sigma2)


# ── Published DNA model (Violanda et al. 2019) ───────────────────────────────

def msd_dna(L, a: float, b: float, c: float):
    """
    MSD model for DNA nucleotide separation distances.

    Derived from the stochastic integral with exponentially decaying memory::

        x(L) = x₀ + ∫₀ᴸ exp(-b(L-s)/2) · ω(s) ds

    The resulting MSD takes the shifted exponential (plateau) form::

        MSD(L) = a - c · exp(-b · L)

    The curve rises from ``(a - c)`` at ``L = 0`` and asymptotes to the
    plateau ``a`` as ``L → ∞``, characteristic of restricted diffusion.

    Parameters
    ----------
    L : float or np.ndarray
        Occurrence number (analogous to time) — the number of intervening
        bases between successive occurrences of a nucleotide.
    a : float
        Plateau height.  MSD approaches ``a`` as L → ∞.  Must satisfy
        ``a > c > 0``.
    b : float
        Exponential decay rate (memory decay parameter, analogous to β
        in other models).
    c : float
        Amplitude of the exponential term.  Controls the rate of rise to
        the plateau.  Must satisfy ``c < a``.

    Returns
    -------
    float or np.ndarray
        MSD values.  Returns nan for non-positive or non-finite results.

    Notes
    -----
    Physical interpretation:

    * Small L:  MSD rises from ``(a - c)`` toward plateau ``a``.
    * Large L:  MSD ≈ ``a``  (restricted diffusion plateau).
    * ``b`` controls the rate of rise; larger ``b`` → faster saturation.

    Validated parameters from Violanda et al. (2019),
    nucleotide A in *Synechococcus elongatus* PCC 7942:

        a ≈ 5.21,  b ≈ 0.0024,  c ≈ 3.81

    (Values vary by nucleotide identity and genome species.)

    References
    ----------
    Violanda, Bernido & Carpio-Bernido (2019),
    "White noise functional integral for exponentially decaying memory:
    nucleotide distribution in bacterial genomes",
    Physica Scripta 94, 125006.

    Examples
    --------
    >>> L = np.arange(1, 500)
    >>> msd = msd_dna(L, a=5.21, b=0.0024, c=3.81)
    """
    L_arr, scalar = _to_array(L)
    val = a - c * np.exp(-b * L_arr)
    return _finalize_msd(val, scalar)


def pdf_dna(dx, L: float, a: float, b: float, c: float) -> np.ndarray:
    """
    PDF for DNA nucleotide separation distances.

    Gaussian with variance equal to the plateau MSD::

        P(dx; L) = 1/√(2π·σ²) · exp(-dx²/(2σ²))
        where  σ² = a - c · exp(-b · L)

    Parameters
    ----------
    dx : float or np.ndarray
        Displacement values.
    L : float
        Occurrence number at which the PDF is evaluated (scalar).
    a, b, c : float
        Same parameters as :func:`msd_dna`.

    Returns
    -------
    np.ndarray
        PDF values.  All-nan if σ² is non-positive or non-finite.

    References
    ----------
    Violanda et al. (2019), Phys. Scr. 94, 125006.
    """
    sigma2 = msd_dna(float(L), a, b, c)
    return _pdf_from_sigma2(dx, sigma2)


# ── Extended model stubs ──────────────────────────────────────────────────────

def _stub(name: str):
    raise NotImplementedError(_NOT_IMPL_MSG.format(name=name))


def msd_sin_half(T):
    """
    MSD for the sin^(1/2) memory kernel model (Table 3.1 row 2).
    No free parameters.
    Status: not yet implemented.
    """
    _stub('sin_half')


def msd_cos_half(T):
    """
    MSD for the cos^(1/2) memory kernel model (Table 3.1 row 3).
    No free parameters.
    Status: not yet implemented.
    """
    _stub('cos_half')


def msd_exp_whittaker(T, mu: float, beta: float, nu: float):
    """
    MSD for the exponential-Whittaker model (Table 3.1 row 5).
    Involves the Whittaker W function.
    Parameters: mu, beta, nu.
    Status: not yet implemented.
    """
    _stub('exp_whittaker')


def msd_bessel_K(T, mu: float, beta: float):
    """
    MSD for the modified Bessel K model (Table 3.1 row 6).
    Parameters: mu, beta.
    Status: not yet implemented.
    """
    _stub('bessel_K')


def msd_hypergeom_F1(T, mu: float, beta: float, nu: float):
    """
    MSD for the Appell hypergeometric F1 model (Table 3.1 row 7).
    Parameters: mu, beta, nu.
    Status: not yet implemented.
    """
    _stub('hypergeom_F1')


def msd_bessel_I(T, mu: float, beta: float):
    """
    MSD for the modified Bessel I model (Table 3.1 row 8).
    Parameters: mu, beta.
    Status: not yet implemented.
    """
    _stub('bessel_I')


def msd_hypergeom_3F2(T, mu: float, beta: float, lam: float):
    """
    MSD for the generalized hypergeometric 3F2 model (Table 3.1 row 11).
    Parameters: mu, beta, lam (lambda).
    Status: not yet implemented.
    """
    _stub('hypergeom_3F2')


def msd_csc_power(T, nu: float, c: float):
    """
    MSD for the cosecant power-law model (Table 3.1 row 12).
    Parameters: nu, c.
    Status: not yet implemented.
    """
    _stub('csc_power')


def msd_cot_power(T, nu: float, c: float):
    """
    MSD for the cotangent power-law model (Table 3.1 row 13).
    Parameters: nu, c.
    Status: not yet implemented.
    """
    _stub('cot_power')


def msd_inc_gamma(T, nu: float, mu: float):
    """
    MSD for the incomplete gamma model (Table 3.1 row 14).
    Parameters: nu, mu.
    Status: not yet implemented.
    """
    _stub('inc_gamma')


def msd_bessel_pair(T, nu: float):
    """
    MSD for the Bessel function pair model (Table 3.1 row 15).
    Parameter: nu.
    Status: not yet implemented.
    """
    _stub('bessel_pair')


def msd_bessel_pair2(T, nu: float, mu: float):
    """
    MSD for the Bessel function pair with memory model (Table 3.1 row 16).
    Parameters: nu, mu.
    Status: not yet implemented.
    """
    _stub('bessel_pair2')


# ── MODELS registry ───────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    'fbm': {
        'msd':         msd_fbm,
        'pdf':         pdf_fbm,
        'params':      ['H'],
        'n_params':    1,
        'row':         1,
        'status':      'available',
        'description': 'Fractional Brownian motion (Hurst exponent)',
        'reference':   'Table 3.1 row 1, Bernido & Carpio-Bernido (2015)',
    },
    'sin_half': {
        'msd':         msd_sin_half,
        'pdf':         None,
        'params':      [],
        'n_params':    0,
        'row':         2,
        'status':      'not_implemented',
        'description': 'sin^(1/2) memory kernel (no free parameters)',
        'reference':   'Table 3.1 row 2',
    },
    'cos_half': {
        'msd':         msd_cos_half,
        'pdf':         None,
        'params':      [],
        'n_params':    0,
        'row':         3,
        'status':      'not_implemented',
        'description': 'cos^(1/2) memory kernel (no free parameters)',
        'reference':   'Table 3.1 row 3',
    },
    'exponential': {
        'msd':         msd_exponential,
        'pdf':         pdf_exponential,
        'params':      ['mu', 'beta'],
        'n_params':    2,
        'row':         4,
        'status':      'available',
        'description': 'Power-law memory with exponential modulation',
        'reference':   'Table 3.1 row 4, Bernido & Carpio-Bernido (2015)',
    },
    'exp_whittaker': {
        'msd':         msd_exp_whittaker,
        'pdf':         None,
        'params':      ['mu', 'beta', 'nu'],
        'n_params':    3,
        'row':         5,
        'status':      'not_implemented',
        'description': 'Power-law memory with Whittaker W function',
        'reference':   'Table 3.1 row 5',
    },
    'bessel_K': {
        'msd':         msd_bessel_K,
        'pdf':         None,
        'params':      ['mu', 'beta'],
        'n_params':    2,
        'row':         6,
        'status':      'not_implemented',
        'description': 'Power-law memory with modified Bessel K',
        'reference':   'Table 3.1 row 6',
    },
    'hypergeom_F1': {
        'msd':         msd_hypergeom_F1,
        'pdf':         None,
        'params':      ['mu', 'beta', 'nu'],
        'n_params':    3,
        'row':         7,
        'status':      'not_implemented',
        'description': 'Power-law memory with Appell hypergeometric F1',
        'reference':   'Table 3.1 row 7',
    },
    'bessel_I': {
        'msd':         msd_bessel_I,
        'pdf':         None,
        'params':      ['mu', 'beta'],
        'n_params':    2,
        'row':         8,
        'status':      'not_implemented',
        'description': 'Power-law memory with modified Bessel I',
        'reference':   'Table 3.1 row 8',
    },
    'sine': {
        'msd':         msd_sine,
        'pdf':         pdf_sine,
        'params':      ['mu', 'nu'],
        'n_params':    2,
        'row':         9,
        'status':      'available',
        'description': 'Power-law memory with sine modulation',
        'reference':   'Table 3.1 row 9, Bernido & Carpio-Bernido (2015)',
    },
    'cosine': {
        'msd':         msd_cosine,
        'pdf':         pdf_cosine,
        'params':      ['mu', 'nu'],
        'n_params':    2,
        'row':         10,
        'status':      'available',
        'description': 'Power-law memory with cosine modulation',
        'reference':   'Table 3.1 row 10, Bernido & Carpio-Bernido (2015)',
    },
    'hypergeom_3F2': {
        'msd':         msd_hypergeom_3F2,
        'pdf':         None,
        'params':      ['mu', 'beta', 'lam'],
        'n_params':    3,
        'row':         11,
        'status':      'not_implemented',
        'description': 'Power-law with generalized hypergeometric 3F2',
        'reference':   'Table 3.1 row 11',
    },
    'csc_power': {
        'msd':         msd_csc_power,
        'pdf':         None,
        'params':      ['nu', 'c'],
        'n_params':    2,
        'row':         12,
        'status':      'not_implemented',
        'description': 'Cosecant power-law modulation',
        'reference':   'Table 3.1 row 12',
    },
    'cot_power': {
        'msd':         msd_cot_power,
        'pdf':         None,
        'params':      ['nu', 'c'],
        'n_params':    2,
        'row':         13,
        'status':      'not_implemented',
        'description': 'Cotangent power-law modulation',
        'reference':   'Table 3.1 row 13',
    },
    'inc_gamma': {
        'msd':         msd_inc_gamma,
        'pdf':         None,
        'params':      ['nu', 'mu'],
        'n_params':    2,
        'row':         14,
        'status':      'not_implemented',
        'description': 'Incomplete gamma modulation',
        'reference':   'Table 3.1 row 14',
    },
    'bessel_pair': {
        'msd':         msd_bessel_pair,
        'pdf':         None,
        'params':      ['nu'],
        'n_params':    1,
        'row':         15,
        'status':      'not_implemented',
        'description': 'Bessel function pair (no memory parameter)',
        'reference':   'Table 3.1 row 15',
    },
    'bessel_pair2': {
        'msd':         msd_bessel_pair2,
        'pdf':         None,
        'params':      ['nu', 'mu'],
        'n_params':    2,
        'row':         16,
        'status':      'not_implemented',
        'description': 'Bessel function pair with memory parameter',
        'reference':   'Table 3.1 row 16',
    },
    'dna': {
        'msd':         msd_dna,
        'pdf':         pdf_dna,
        'params':      ['a', 'b', 'c'],
        'n_params':    3,
        'row':         None,
        'status':      'available',
        'description': 'Exponentially decaying memory — DNA nucleotide '
                       'separation distances (plateau MSD shape)',
        'reference':   'Violanda et al. (2019), Phys. Scr. 94, 125006',
    },
}

_AVAILABLE_NAMES = [n for n, v in MODELS.items() if v['status'] == 'available']
_ALL_NAMES = list(MODELS.keys())

# ── Registry accessors ────────────────────────────────────────────────────────

def get_model(name: str) -> dict:
    """
    Return the model entry from the registry for a given model name.

    Parameters
    ----------
    name : str
        Model name (e.g. ``'cosine'``, ``'exponential'``).

    Returns
    -------
    dict
        Registry entry with keys ``'msd'``, ``'pdf'``, ``'params'``,
        ``'n_params'``, ``'row'``, ``'status'``, ``'description'``,
        ``'reference'``.

    Raises
    ------
    ValueError
        If ``name`` is not one of the 16 registered models.
    NotImplementedError
        If the model is registered but not yet implemented (stub).

    Examples
    --------
    >>> m = wn.get_model('cosine')
    >>> msd_fn = m['msd']
    """
    if name not in MODELS:
        raise ValueError(
            f"✗ Unknown model '{name}'.\n"
            f"Available: {', '.join(_ALL_NAMES)}\n"
            f"Run wn.list_models() for details."
        )
    info = MODELS[name]
    if info['status'] == 'not_implemented':
        raise NotImplementedError(
            f"Model '{name}' (row {info['row']}) is not yet implemented.\n"
            f"Available models: cosine, exponential, sine, fbm\n"
            f"Run wn.list_models() to see all 16 models."
        )
    return info


def list_models() -> None:
    """
    Print a formatted table of all 16 models from Table 3.1.

    Shows name, row number, parameters, implementation status,
    and description for each model.

    Examples
    --------
    >>> wn.list_models()
    """
    _PSYM = {
        'mu': 'mu', 'nu': 'nu', 'beta': 'beta',
        'H': 'H', 'lam': 'lam', 'c': 'c',
    }

    def _fmt_params(params: list[str]) -> str:
        return ', '.join(_PSYM.get(p, p) for p in params) or '(none)'

    def _fmt_status(status: str) -> str:
        return 'available' if status == 'available' else 'stub'

    # Column specs: (header, width)
    cols = [
        ('Name',        16),
        ('Row',          4),
        ('Params',      14),
        ('Status',      10),
        ('Description', 40),
    ]

    def _hline(left='+-', mid='-+-', right='-+') -> str:
        return left[0] + mid[1].join('-' * (w + 2) for _, w in cols) + right[-1]

    def _row(*values) -> str:
        parts = [f' {str(v):<{w}} ' for v, (_, w) in zip(values, cols)]
        return '|' + '|'.join(parts) + '|'

    sep = _hline(left='+-', mid='-+-', right='-+')
    print(sep)
    print(_row(*[h for h, _ in cols]))
    print(sep)
    # Table 3.1 models first (sorted by row), then extras (row=None) at the end
    def _sort_key(kv):
        r = kv[1]['row']
        return (r is None, r or 0)

    for name, info in sorted(MODELS.items(), key=_sort_key):
        p = _fmt_params(info['params'])
        s = _fmt_status(info['status'])
        d = info['description']
        if len(d) > 40:
            d = d[:37] + '...'
        row_display = 'N/A' if info['row'] is None else info['row']
        print(_row(name, row_display, p, s, d))
    print(sep)

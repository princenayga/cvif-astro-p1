"""
io/reader.py — CSV reading for the whitenoise package.

Handles the standard whitenoise CSV format::

    time [months], sunspot_number [count]
    1, 12.4
    2, 15.1

Column 1 is always time/index. Column 2 (and beyond) are observables.
Header format:  name [unit]   — space before bracket is optional.
"""

from __future__ import annotations

import os
import re

import numpy as np

# ── Validation constants ──────────────────────────────────────────────────────

_KNOWN_TIME_NAMES: set[str] = {
    'time', 't', 'index', 'idx', 'step', 'steps', 'sample',
    'samples', 'observation', 'observations', 'date', 'datetime',
    'timestamp', 'epoch', 'frame', 'lag', 'year', 'month', 'day',
    'hour', 'minute', 'second', 'yr', 'mo', 'hr', 'min', 'sec',
}

_NON_NEGATIVE_UNITS: set[str] = {
    'count', 'counts', 'number', 'numbers', 'freq', 'frequency',
    'intensity', 'flux', 'brightness', 'magnitude', 'population',
    'cases', 'events', 'price', 'usd', 'eur', 'ppm', 'ppb',
    'percent', '%', 'fraction', 'probability',
}

_HEADER_RE = re.compile(r'^\s*([^\[\]]+?)\s*(?:\[\s*([^\]]*?)\s*\])?\s*$')


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_header_column(header: str) -> tuple[str, str]:
    """
    Parse a single CSV header string into (name, unit), both lowercase.

    Handles all formatting variants::

        'time [months]'          → ('time', 'months')
        'time[months]'           → ('time', 'months')
        'Time [Months]'          → ('time', 'months')
        'sunspot_number [count]' → ('sunspot_number', 'count')
        'flux []'                → ('flux', '')
        'flux [  ]'              → ('flux', '')
        'value [unitless]'       → ('value', 'unitless')
        'distance'               → ('distance', '')
        '  time  [months]  '     → ('time', 'months')

    Parameters
    ----------
    header : str
        A single column header string.

    Returns
    -------
    name : str
        Column name, stripped and lowercased.
    unit : str
        Unit string, stripped and lowercased. Empty string if absent or blank.
    """
    m = _HEADER_RE.match(header)
    if not m:
        return header.strip().lower(), ''
    name = m.group(1).strip().lower()
    unit_raw = m.group(2)
    unit = unit_raw.strip().lower() if unit_raw is not None else ''
    return name, unit


def _make_axis_label(name: str, unit: str) -> str:
    """
    Build a matplotlib-ready axis label string.

    Parameters
    ----------
    name : str
        Column name (already lowercase).
    unit : str
        Unit string (already lowercase). Use '' or 'unitless' for no unit.

    Returns
    -------
    str
        ``name`` if unit is empty/unitless, else ``'name (unit)'``.

    Examples
    --------
    >>> _make_axis_label('sunspot_number', 'count')
    'sunspot_number (count)'
    >>> _make_axis_label('flux', '')
    'flux'
    >>> _make_axis_label('flux', 'unitless')
    'flux'
    """
    if not unit or unit == 'unitless':
        return name
    return f'{name} ({unit})'


def _validate(time: np.ndarray, values: np.ndarray, metadata: dict) -> None:
    """
    Run 3 non-fatal validation checks and print ⚠ warnings to stdout.

    Never raises; data is loaded regardless of warnings.

    Parameters
    ----------
    time : np.ndarray
    values : np.ndarray
    metadata : dict
        Must contain 'time_name', 'value_name', 'value_unit'.
    """
    # V1: Columns possibly swapped (time column has huge spread)
    if len(time) > 1:
        std_t = float(np.std(time))
        mean_t = float(np.mean(time))
        if std_t > 100.0 * (abs(mean_t) + 1e-10):
            print(
                f"⚠  Column order warning: the time column "
                f"('{metadata['time_name']}') has unusually large spread "
                f"(std={std_t:.2f}, mean={mean_t:.2f}).\n"
                f"   If your columns are swapped, re-save the CSV with time "
                f"in column 1 and your observable in column 2."
            )

    # V2: Time column name doesn't look like a time axis
    if metadata['time_name'].lower() not in _KNOWN_TIME_NAMES:
        print(
            f"⚠  Time column name warning: '{metadata['time_name']}' is not "
            f"a recognized time/index column name. Expected names like: "
            f"time, index, step, year, month, day, etc.\n"
            f"   If '{metadata['time_name']}' is not your time axis, check "
            f"column order."
        )

    # V3: Negative values in a unit that implies non-negative data
    value_unit = metadata['value_unit'].lower()
    if value_unit in _NON_NEGATIVE_UNITS and np.any(values < 0):
        n_neg = int(np.sum(values < 0))
        min_val = float(np.min(values))
        print(
            f"⚠  Value range warning: column '{metadata['value_name']}' has "
            f"unit '{metadata['value_unit']}' which typically implies "
            f"non-negative values, but {n_neg} negative value(s) were found "
            f"(min = {min_val:.4f}).\n"
            f"   If your data is already detrended, this is expected — ignore "
            f"this warning."
        )


def _build_metadata(
    time_name: str,
    time_unit: str,
    value_name: str,
    value_unit: str,
    source_file: str,
    n_points: int,
) -> dict:
    """Assemble the standard metadata dict returned by read functions."""
    return {
        'time_name':   time_name,
        'time_unit':   time_unit,
        'value_name':  value_name,
        'value_unit':  value_unit,
        'time_label':  _make_axis_label(time_name, time_unit),
        'value_label': _make_axis_label(value_name, value_unit),
        'source_file': source_file,
        'n_points':    n_points,
    }


def _read_raw_lines(path: str) -> list[str]:
    """Open file, strip blank lines, raise clean errors."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'✗ File not found: {path}')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw = fh.read()
    except Exception as exc:
        raise ValueError(f'✗ Could not read file "{path}": {exc}') from exc
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f'✗ File is empty: {path}')
    return lines


def _parse_header_line(header_line: str, min_cols: int = 2) -> list[tuple[str, str]]:
    """Split a header line on commas and parse each column. Returns list of (name, unit)."""
    cols = [c.strip() for c in header_line.split(',')]
    if len(cols) < min_cols:
        raise ValueError(
            f'✗ CSV must have at least {min_cols} columns '
            f'(time and observable). '
            f'Found {len(cols)} column(s) in header: "{header_line}"'
        )
    return [_parse_header_column(c) for c in cols]


def _parse_data_rows(
    lines: list[str],
    n_cols: int,
    col_indices: list[int],
) -> list[list[float]]:
    """
    Parse data rows into lists of floats.

    Parameters
    ----------
    lines : list[str]
        Data lines (no header).
    n_cols : int
        Expected number of comma-separated columns.
    col_indices : list[int]
        Which column indices to extract.

    Returns
    -------
    list of lists — one inner list per col_index.
    """
    columns: list[list[float]] = [[] for _ in col_indices]
    for row_num, line in enumerate(lines, start=2):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < n_cols:
            raise ValueError(
                f'✗ Row {row_num} has {len(parts)} value(s) but the header '
                f'declares {n_cols} column(s). Line: "{line}"'
            )
        for out_idx, col_idx in enumerate(col_indices):
            raw = parts[col_idx]
            try:
                columns[out_idx].append(float(raw))
            except ValueError:
                raise ValueError(
                    f'✗ Non-numeric value in row {row_num}, '
                    f'column {col_idx + 1}: "{raw}". '
                    f'All data cells must be numbers.'
                )
    return columns


# ── Public API ────────────────────────────────────────────────────────────────

def read_csv(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read a 2-column whitenoise-format CSV file.

    The expected format is::

        time [months], sunspot_number [count]
        1, 12.4
        2, 15.1
        3, 14.8

    Column 1 is always time/index. Column 2 is always the observable.
    Header strings follow the ``name [unit]`` convention; the space before
    ``[`` is optional and capitalisation is ignored.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    time : np.ndarray
        1-D float array of time/index values.
    values : np.ndarray
        1-D float array of observable values.
    metadata : dict
        Keys:

        * ``'time_name'``   — column 1 name, lowercased (e.g. ``'time'``)
        * ``'time_unit'``   — column 1 unit, lowercased (``''`` if absent)
        * ``'value_name'``  — column 2 name, lowercased (e.g. ``'sunspot_number'``)
        * ``'value_unit'``  — column 2 unit, lowercased (``''`` if absent)
        * ``'time_label'``  — axis-ready string (e.g. ``'time (months)'``)
        * ``'value_label'`` — axis-ready string (e.g. ``'sunspot_number (count)'``)
        * ``'source_file'`` — the ``path`` argument as given
        * ``'n_points'``    — number of data rows read

    Raises
    ------
    FileNotFoundError
        ``'✗ File not found: {path}'``
    ValueError
        ``'✗ {reason}'`` for malformed files (wrong column count,
        non-numeric data, empty file, etc.).

    Examples
    --------
    >>> time, values, meta = wn.read_csv('sunspot.csv')
    >>> meta['value_label']
    'sunspot_number (count)'
    >>> meta['time_label']
    'time (months)'
    """
    lines = _read_raw_lines(path)
    parsed_headers = _parse_header_line(lines[0], min_cols=2)
    time_name, time_unit = parsed_headers[0]
    value_name, value_unit = parsed_headers[1]

    data_lines = lines[1:]
    if not data_lines:
        raise ValueError(f'✗ File has a header but no data rows: "{path}"')

    columns = _parse_data_rows(data_lines, n_cols=len(parsed_headers), col_indices=[0, 1])

    time_arr = np.array(columns[0], dtype=float)
    values_arr = np.array(columns[1], dtype=float)

    metadata = _build_metadata(
        time_name, time_unit, value_name, value_unit, path, len(time_arr)
    )
    _validate(time_arr, values_arr, metadata)

    return time_arr, values_arr, metadata


def read_csv_multi(path: str) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """
    Read a multi-column whitenoise-format CSV file.

    Column 1 is the shared time axis. Every additional column is treated as
    a separate observable and returned as its own ``(time, values, metadata)``
    tuple. Used internally by ``batch_analyze()`` and available for advanced
    users who want to process multi-system CSVs manually.

    The expected format is::

        time [yr], co2 [ppm], temperature [°C]
        1958, 315.2, 14.1
        1959, 315.9, 14.0

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    list of (time, values, metadata) tuples
        One tuple per observable column (i.e. ``n_columns - 1`` tuples).
        Each ``time`` array is the same shared time axis.

    Raises
    ------
    FileNotFoundError
        ``'✗ File not found: {path}'``
    ValueError
        ``'✗ {reason}'`` for malformed files.

    Examples
    --------
    >>> datasets = wn.read_csv_multi('climate.csv')
    >>> len(datasets)      # 2 if CSV has time + 2 observable columns
    2
    >>> datasets[0][2]['value_name']
    'co2'
    """
    lines = _read_raw_lines(path)
    parsed_headers = _parse_header_line(lines[0], min_cols=2)
    time_name, time_unit = parsed_headers[0]
    n_cols = len(parsed_headers)

    data_lines = lines[1:]
    if not data_lines:
        raise ValueError(f'✗ File has a header but no data rows: "{path}"')

    all_col_indices = list(range(n_cols))
    columns = _parse_data_rows(data_lines, n_cols=n_cols, col_indices=all_col_indices)

    time_arr = np.array(columns[0], dtype=float)
    n_points = len(time_arr)

    results: list[tuple[np.ndarray, np.ndarray, dict]] = []
    for j in range(1, n_cols):
        val_name, val_unit = parsed_headers[j]
        val_arr = np.array(columns[j], dtype=float)
        meta = _build_metadata(
            time_name, time_unit, val_name, val_unit, path, n_points
        )
        _validate(time_arr, val_arr, meta)
        results.append((time_arr, val_arr, meta))

    return results

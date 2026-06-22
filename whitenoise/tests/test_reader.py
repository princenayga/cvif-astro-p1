"""
tests/test_reader.py — Self-contained tests for whitenoise/io/reader.py

Run with:
    python tests/test_reader.py

Each test prints "PASS" or "FAIL: {reason}".
Final line: "X/12 tests passed."
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

# Force UTF-8 output so ⚠ prints correctly on Windows (cp1252 terminals)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Allow running from the repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from whitenoise.io.reader import read_csv, read_csv_multi

# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_tmp(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(content)
    return path


def _run(name: str, fn) -> bool:
    """Run fn(); return True on pass, print result."""
    try:
        fn()
        print(f"PASS  {name}")
        return True
    except AssertionError as exc:
        print(f"FAIL  {name}: {exc}")
        return False
    except Exception as exc:
        print(f"FAIL  {name}: unexpected {type(exc).__name__}: {exc}")
        return False


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_standard_format():
    """Standard format with space before bracket."""
    path = _write_tmp(
        "time [months], sunspot_number [count]\n"
        "1,12.4\n2,15.1\n3,14.8"
    )
    try:
        t, v, m = read_csv(path)
        assert list(t) == [1.0, 2.0, 3.0],          f"time wrong: {list(t)}"
        assert list(v) == [12.4, 15.1, 14.8],        f"values wrong: {list(v)}"
        assert m['time_unit'] == 'months',            f"time_unit={m['time_unit']!r}"
        assert m['value_unit'] == 'count',            f"value_unit={m['value_unit']!r}"
        assert m['time_label'] == 'time (months)',    f"time_label={m['time_label']!r}"
        assert m['value_label'] == 'sunspot_number (count)', \
                                                      f"value_label={m['value_label']!r}"
        assert m['n_points'] == 3,                    f"n_points={m['n_points']}"
    finally:
        os.unlink(path)


def test_02_no_space_before_bracket():
    """No space before bracket."""
    path = _write_tmp("time[yr], co2[ppm]\n1958,315.2\n1959,315.9")
    try:
        _, _, m = read_csv(path)
        assert m['time_unit'] == 'yr',       f"time_unit={m['time_unit']!r}"
        assert m['value_unit'] == 'ppm',     f"value_unit={m['value_unit']!r}"
        assert m['value_label'] == 'co2 (ppm)', f"value_label={m['value_label']!r}"
    finally:
        os.unlink(path)


def test_03_capitalized_headers():
    """Capitalized headers are lowercased internally."""
    path = _write_tmp(
        "Time [Months], Sunspot_Number [Count]\n"
        "1,12.4\n2,15.1"
    )
    try:
        _, _, m = read_csv(path)
        assert m['time_name'] == 'time',         f"time_name={m['time_name']!r}"
        assert m['value_name'] == 'sunspot_number', \
                                                 f"value_name={m['value_name']!r}"
        assert m['time_unit'] == 'months',       f"time_unit={m['time_unit']!r}"
        assert m['value_unit'] == 'count',       f"value_unit={m['value_unit']!r}"
    finally:
        os.unlink(path)


def test_04_capitalized_no_space():
    """Capitalized headers without space before bracket."""
    path = _write_tmp("TIME[YR], CO2[PPM]\n1958,315.2\n1959,315.9")
    try:
        _, _, m = read_csv(path)
        assert m['time_name'] == 'time',      f"time_name={m['time_name']!r}"
        assert m['time_unit'] == 'yr',        f"time_unit={m['time_unit']!r}"
        assert m['value_label'] == 'co2 (ppm)', f"value_label={m['value_label']!r}"
    finally:
        os.unlink(path)


def test_05_unitless_empty_brackets():
    """Empty brackets → unit is empty string → no parentheses in label."""
    path = _write_tmp("time [days], normalized_flux []\n0,0.9987\n1,1.0023")
    try:
        _, _, m = read_csv(path)
        assert m['value_unit'] == '',                  f"value_unit={m['value_unit']!r}"
        assert m['value_label'] == 'normalized_flux',  f"value_label={m['value_label']!r}"
    finally:
        os.unlink(path)


def test_06_unitless_whitespace_inside_brackets():
    """Whitespace inside brackets → treated as empty."""
    path = _write_tmp("time [days], flux [  ]\n0,1.0\n1,2.0")
    try:
        _, _, m = read_csv(path)
        assert m['value_unit'] == '',  f"value_unit={m['value_unit']!r}"
        assert m['value_label'] == 'flux', f"value_label={m['value_label']!r}"
    finally:
        os.unlink(path)


def test_07_no_brackets_at_all():
    """Headers with no brackets → unit is empty, label has no parentheses."""
    path = _write_tmp("time, distance\n1,100.0\n2,200.0")
    try:
        _, _, m = read_csv(path)
        assert m['time_unit'] == '',    f"time_unit={m['time_unit']!r}"
        assert m['value_unit'] == '',   f"value_unit={m['value_unit']!r}"
        assert m['time_label'] == 'time',      f"time_label={m['time_label']!r}"
        assert m['value_label'] == 'distance', f"value_label={m['value_label']!r}"
    finally:
        os.unlink(path)


def test_08_read_csv_multi():
    """Multi-column CSV — returns one tuple per observable column."""
    path = _write_tmp(
        "time [yr], co2 [ppm], temperature [°C]\n"
        "1958,315.2,14.1\n"
        "1959,315.9,14.0"
    )
    try:
        result = read_csv_multi(path)
        assert len(result) == 2, f"Expected 2 tuples, got {len(result)}"
        assert result[0][2]['value_name'] == 'co2',         \
                                            f"name0={result[0][2]['value_name']!r}"
        assert result[1][2]['value_name'] == 'temperature', \
                                            f"name1={result[1][2]['value_name']!r}"
        assert result[0][2]['value_unit'] == 'ppm',         \
                                            f"unit0={result[0][2]['value_unit']!r}"
        # Both share the same time array
        import numpy as np
        assert list(result[0][0]) == [1958.0, 1959.0], f"time0 wrong"
        assert list(result[1][0]) == [1958.0, 1959.0], f"time1 wrong"
    finally:
        os.unlink(path)


def test_09_file_not_found():
    """Missing file raises FileNotFoundError with ✗ prefix."""
    try:
        read_csv('_this_file_does_not_exist_xyz.csv')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as exc:
        msg = str(exc)
        assert '✗' in msg,         f"No ✗ in message: {msg!r}"
        assert 'not found' in msg, f"'not found' missing from: {msg!r}"


def test_10_single_column_raises():
    """CSV with only 1 column raises ValueError with ✗ prefix."""
    path = _write_tmp("time [months]\n1\n2\n3")
    try:
        read_csv(path)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert '✗' in str(exc), f"No ✗ in error: {exc}"
    finally:
        os.unlink(path)


def test_11_v1_column_swap_warning():
    """V1 warning fires when time column has huge spread relative to mean.

    V1 condition: std > 100 * (|mean| + 1e-10)
    Fires when the 'time' column is zero-centered with nonzero spread,
    which suggests the user accidentally put an observable (oscillating
    around 0) in column 1 instead of a monotonic time axis.
    """
    # Use zero-centered oscillating values as the 'time' column so that
    # std >> |mean| and the V1 condition triggers.
    values_col1 = [-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500]
    values_col2 = list(range(1, 12))
    rows = '\n'.join(f'{t},{v}' for t, v in zip(values_col1, values_col2))
    path = _write_tmp(f"time [units], value []\n{rows}")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            t, v, m = read_csv(path)
        output = buf.getvalue()
        assert '⚠' in output,            f"No ⚠ in stdout: {output!r}"
        assert 'Column order' in output,  f"'Column order' missing: {output!r}"
        # Data still loaded
        assert len(t) == 11, f"Expected 11 rows, got {len(t)}"
    finally:
        os.unlink(path)


def test_12_v2_and_v3_warnings():
    """V2 fires on unrecognized time column name; V3 fires on negative counts."""
    # V2: time column named 'measurement' — not in KNOWN_TIME_NAMES
    path_v2 = _write_tmp("measurement [km], distance [km]\n1,100\n2,200")
    buf_v2 = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_v2):
            t2, v2, m2 = read_csv(path_v2)
        out2 = buf_v2.getvalue()
        assert '⚠' in out2,                  f"V2: no ⚠ in stdout: {out2!r}"
        assert 'Time column name' in out2,    f"V2: 'Time column name' missing: {out2!r}"
        assert len(t2) == 2,                  f"V2: data not loaded, got {len(t2)} rows"
    finally:
        os.unlink(path_v2)

    # V3: unit 'count' with negative values
    path_v3 = _write_tmp("time [days], events [count]\n1,-5.0\n2,3.0")
    buf_v3 = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_v3):
            t3, v3, m3 = read_csv(path_v3)
        out3 = buf_v3.getvalue()
        assert '⚠' in out3,             f"V3: no ⚠ in stdout: {out3!r}"
        assert 'Value range' in out3,   f"V3: 'Value range' missing: {out3!r}"
        assert len(t3) == 2,            f"V3: data not loaded"
    finally:
        os.unlink(path_v3)


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    ("01 Standard format with space",         test_01_standard_format),
    ("02 No space before bracket",            test_02_no_space_before_bracket),
    ("03 Capitalized headers",                test_03_capitalized_headers),
    ("04 Capitalized, no space",              test_04_capitalized_no_space),
    ("05 Unitless empty brackets",            test_05_unitless_empty_brackets),
    ("06 Unitless whitespace in brackets",    test_06_unitless_whitespace_inside_brackets),
    ("07 No brackets at all",                 test_07_no_brackets_at_all),
    ("08 read_csv_multi",                     test_08_read_csv_multi),
    ("09 FileNotFoundError",                  test_09_file_not_found),
    ("10 Single column raises ValueError",    test_10_single_column_raises),
    ("11 V1 column swap warning",             test_11_v1_column_swap_warning),
    ("12 V2 and V3 warnings",                 test_12_v2_and_v3_warnings),
]

if __name__ == '__main__':
    passed = sum(_run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print(f"\n{passed}/{total} tests passed.")
    sys.exit(0 if passed == total else 1)

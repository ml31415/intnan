"""Jitted callers of the public functions must compile the flat kernels via overload.

The pure python routes are covered by test_generic; here the same functions are called from
numba compiled code and compared against the python results, plus the TypingErrors for the
cases that stay python only (axis, keepdims, weights, ...).
"""

import numpy as np
import pytest

try:
    import numba as nb
    from numba.core.errors import TypingError
except ImportError:
    nb = None
    TypingError = Exception  # type: ignore[misc]

from .. import intnan_numba as inn

pytestmark = pytest.mark.skipif(nb is None, reason="numba not available")

_IMPLEMENTED = [
    "allnan",
    "anynan",
    "nancount",
    "nanfirst",
    "nanlast",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanptp",
    "nanprod",
    "nansum",
    "nanargmax",
    "nanargmin",
]


def _jit(call: str, **g):
    """Compile `call` (an expression over x) as a numba function."""
    ns: dict = {}
    exec(f"def _jitted(x):\n    return {call}", {"inn": inn, "np": np, **g}, ns)
    return nb.njit(ns["_jitted"])


def _make(dtype, shape, fortran=False):
    rng = np.random.default_rng(42)
    a = rng.integers(-50, 50, size=shape).astype(dtype)
    if np.dtype(dtype).kind == "f":
        a[rng.random(shape) < 0.3] = np.nan
    else:
        a.ravel()[::7] = np.dtype(dtype).type(inn.nanval(np.dtype(dtype)))
    return np.asfortranarray(a) if fortran else a


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize("shape,fortran", [((12,), False), ((3, 4), False), ((3, 4), True)])
@pytest.mark.parametrize("name", _IMPLEMENTED)
def test_flat_call_matches_python(name, dtype, shape, fortran):
    x = _make(dtype, shape, fortran)
    axis = ", axis=None" if name in ("nanfirst", "nanlast") else ""
    jitted = _jit(f"inn.{name}(x{axis})")
    np.testing.assert_array_equal(np.asarray(getattr(inn, name)(x, *([None] if axis else []))), np.asarray(jitted(x)))


def test_extras_match_python():
    x = _make(np.float64, (12,))
    y = _make(np.float64, (12,))
    assert inn.nanvar(x, ddof=1) == _jit("inn.nanvar(x, ddof=1)")(x)
    assert inn.nanstd(x, None, False, 1) == _jit("inn.nanstd(x, None, False, 1)")(x)
    np.testing.assert_array_equal(inn.nanmaximum(x, y), _jit("inn.nanmaximum(x, y)", y=y)(x))
    np.testing.assert_array_equal(inn.nanminimum(x, y), _jit("inn.nanminimum(x, y)", y=y)(x))
    assert inn.nanaverage(x) == _jit("inn.nanaverage(x)")(x)
    assert inn.nanpercentile(x, 25.0) == _jit("inn.nanpercentile(x, 25.0)")(x)
    assert inn.nanquantile(x, 0.25) == _jit("inn.nanquantile(x, 0.25)")(x)
    np.testing.assert_array_equal(inn.nanclip(x, -10, 10), _jit("inn.nanclip(x, -10, 10)")(x))


def test_layout_free_transposes_in_jit():
    x = _make(np.float64, (3, 4), fortran=True)
    assert inn.nanmedian(x) == _jit("inn.nanmedian(x)")(x)
    assert inn.allnan(x) is _jit("inn.allnan(x)")(x)


def test_int_marker_translation():
    for dtype in (np.int32, np.uint8):
        x = _make(dtype, (12,))
        np.testing.assert_array_equal(inn.nancumsum(x), _jit("inn.nancumsum(x)")(x))
        np.testing.assert_array_equal(inn.nancumprod(x), _jit("inn.nancumprod(x)")(x))


def test_clip_edge_cases_in_jit():
    x = _make(np.float64, (12,))
    np.testing.assert_array_equal(inn.nanclip(x, 3.0, -3.0), _jit("inn.nanclip(x, 3.0, -3.0)")(x))
    np.testing.assert_array_equal(inn.nanclip(x, None, 5), _jit("inn.nanclip(x, None, 5)")(x))
    with pytest.raises(TypingError, match="promote"):
        _jit("inn.nanclip(x, -2.5, 2)")(_make(np.int64, (12,)))


def test_degenerate_inputs():
    for x in (np.array([], dtype=np.float64), np.full(5, np.nan)):
        for name in ("nansum", "nanmean", "nanmax", "nancount", "nanmedian"):
            np.testing.assert_array_equal(np.asarray(getattr(inn, name)(x)), np.asarray(_jit(f"inn.{name}(x)")(x)))


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64, np.int32])
@pytest.mark.parametrize("layout", ["cc", "ff", "cf", "fc"])
def test_nanequal_nanclose_match_python(dtype, layout):
    x = _make(dtype, (3, 4))
    y = _make(dtype, (3, 4))
    order = "F" if layout[0] == "f" else "C"
    x = np.asarray(x, order=order)
    y = np.asarray(y, order="F" if layout[1] == "f" else "C")
    np.testing.assert_array_equal(inn.nanequal(x, y), _jit("inn.nanequal(x, y)", y=y)(x))
    np.testing.assert_array_equal(inn.nanclose(x, y), _jit("inn.nanclose(x, y)", y=y)(x))
    np.testing.assert_array_equal(inn.nanclose(x, y, delta=1e3), _jit("inn.nanclose(x, y, delta)", y=y, delta=1e3)(x))


def test_nanequal_nanclose_marker_semantics():
    # the missing marker is an ordinary value: equal markers match, tolerance applies to it too
    m = np.array([inn.nanval(np.dtype(np.int64)), 5, 0], dtype=np.int64)
    np.testing.assert_array_equal(inn.nanequal(m, m), _jit("inn.nanequal(x, x)")(m))
    assert inn.nanequal(m, m)[0]
    assert inn.nanclose(m, m)[0]  # |marker - marker| <= delta
    np.testing.assert_array_equal(inn.nanclose(m, m, delta=1e-300), _jit("inn.nanclose(x, x, delta)", delta=1e-300)(m))


def test_nanequal_nanclose_unsupported_in_jit():
    x = _make(np.float64, (3, 4))
    with pytest.raises(TypingError, match="same data type"):
        _jit("inn.nanequal(x, y)", y=_make(np.int32, (3, 4)))(x)
    with pytest.raises(TypingError, match="same data type"):
        _jit("inn.nanclose(x, y)", y=_make(np.int32, (3, 4)))(x)
    with pytest.raises((TypingError, ValueError), match="equal shapes"):
        _jit("inn.nanequal(x, y)", y=_make(np.float64, (4, 3)))(x)


def test_axis_and_kwargs_unsupported_in_jit():
    x = _make(np.float64, (12,))
    with pytest.raises(TypingError, match="axis and keepdims are not supported"):
        _jit("inn.nansum(x, 0)")(x)
    with pytest.raises(TypingError, match="compile time constants"):
        _jit("inn.nansum(x, ax)", ax=0)(x)  # runtime value, not a literal
    with pytest.raises(TypingError, match="not supported"):
        _jit("inn.nanmean(x, None, True)")(x)
    with pytest.raises(TypingError, match="axis and keepdims"):
        _jit("inn.nanfirst(x)")(x)  # default axis=0 routes to numpy in python
    with pytest.raises(TypingError, match="weights"):
        _jit("inn.nanaverage(x, None, w)", w=x)(x)
    with pytest.raises(TypingError, match="q must be a scalar"):
        _jit("inn.nanpercentile(x, q)", q=np.array([10.0, 90.0]))(x)
    with pytest.raises(TypingError, match="only 1d"):
        _jit("inn.nancumsum(x)")(x.reshape(3, 4))

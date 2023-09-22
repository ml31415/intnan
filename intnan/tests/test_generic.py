import itertools
import warnings

import numpy as np
import pytest

try:
    from types import SimpleNamespace
except ImportError:

    class SimpleNamespace(object):
        def __init__(self, **kwargs):
            self.__dict__.update(**kwargs)


from .. import intnan_np

try:
    from .. import intnan_numba
except ImportError:
    intnan_numba = None

_implementations = [impl for impl in [intnan_np, intnan_numba] if impl is not None]


@pytest.fixture(params=_implementations, ids=lambda impl: impl.__name__.split("_")[-1])
def inn(request):
    return request.param


def test_nanval(inn):
    assert inn.nanval(np.ones(10, dtype=np.int64)) == -(2**63)
    assert inn.nanval(np.ones(10, dtype=np.int32)) == -(2**31)
    assert np.isnan(inn.nanval(np.ones(10, dtype=np.float64)))
    assert np.isnan(inn.nanval(np.ones(10, dtype=np.float32)))


def test_asfloat(inn):
    np.testing.assert_array_equal(inn.asfloat(np.array([True, False])), np.array([1.0, 0.0]))
    np.testing.assert_array_equal(inn.asfloat(np.array([1.0, 0.0, np.nan])), np.array([1.0, 0.0, np.nan]))
    np.testing.assert_array_equal(inn.asfloat(np.array([1, 0, intnan_np.INTNAN64])), np.array([1.0, 0.0, np.nan]))


ninp_list = itertools.product(
    ["small", "large"],
    ["nonans", "nans", "allnans"],
    [np.int64, np.int32, np.float64, np.float32],
)


@pytest.fixture(params=ninp_list, ids=lambda x: "-".join((x[0], x[1], x[2].__name__)))
def ninp(request):
    sizestr, nanstate, dtype = request.param
    if sizestr == "small":
        size = 100
    else:
        size = 10000

    a = np.arange(size, dtype=dtype)
    a_nanmask = np.zeros_like(a, dtype=bool)
    warnings = "error"
    if nanstate == "nans":
        a_nanmask[::2] = True
    elif nanstate == "allnans":
        a_nanmask[:] = True
        warnings = "ignore"
    a[a_nanmask] = intnan_np.nanval(a)

    b = np.arange(size, dtype=dtype) + 1
    b_nanmask = np.zeros_like(b, dtype=bool)
    b_nanmask[::3] = True
    b[b_nanmask] = intnan_np.nanval(a)
    return SimpleNamespace(**locals())


@pytest.mark.parametrize(
    "val",
    [np.nan, intnan_np.INTNAN32, intnan_np.INTNAN64, b"", ""],
    ids=["nan", "INTNAN32", "INTNAN64", "empty bytes", "empty unicode"],
)
def test_isnan(inn, val):
    assert inn.isnan(val)


@pytest.mark.parametrize("val", [0, -1, "asdf"])
def test_not_isnan(inn, val):
    assert not inn.isnan(val)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_array_element_isnan(inn, dtype):
    nanval = inn.nanval(np.array([], dtype=dtype))
    arr = np.array([nanval], dtype=dtype)
    assert inn.isnan(arr[0])


def test_isnan_array(inn, ninp):
    np.testing.assert_array_equal(ninp.a_nanmask, inn.isnan(ninp.a))


def test_anynan(inn, ninp):
    assert inn.anynan(ninp.a) == (ninp.nanstate != "nonans")


def test_allnan(inn, ninp):
    assert inn.allnan(ninp.a) == (ninp.nanstate == "allnans")


@pytest.mark.parametrize("copy", (True, False))
def test_fix_invalid(inn, ninp, copy):
    x = ninp.a.copy()
    fixed = inn.fix_invalid(x, copy=copy)
    assert not inn.anynan(fixed)
    assert np.all(fixed[ninp.a_nanmask] == 0)
    assert np.all(fixed[~ninp.a_nanmask] == ninp.a[~ninp.a_nanmask])
    if copy:
        assert x is not fixed
    else:
        assert x is fixed


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_nanmax(inn, ninp):
    if ninp.nanstate != "allnans":
        assert inn.nanmax(ninp.a) == np.nanmax(ninp.a)
    else:
        np.testing.assert_equal(inn.nanmax(ninp.a), inn.nanval(ninp.a))


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_nanmin(inn, ninp):
    if ninp.nanstate == "nonans":
        assert inn.nanmin(ninp.a) == 0
    elif ninp.nanstate == "nans":
        assert inn.nanmin(ninp.a) == 1
    else:
        np.testing.assert_equal(inn.nanmin(ninp.a), inn.nanval(ninp.a))


def test_nanmaximum(inn, ninp):
    res = inn.nanmaximum(ninp.a, ninp.b)
    # Wherever a is nan, value from b needs to be picked
    np.testing.assert_array_equal(res[ninp.a_nanmask], ninp.b[ninp.a_nanmask])
    # Wherever b is nan, value from a needs to be picked
    np.testing.assert_array_equal(res[ninp.b_nanmask], ninp.a[ninp.b_nanmask])
    # Wherever both are nan, expect nanval
    assert inn.allnan(res[ninp.a_nanmask & ninp.b_nanmask])


def test_nanminimum(inn, ninp):
    res = inn.nanminimum(ninp.a, ninp.b)
    # Wherever a is nan, value from b needs to be picked
    np.testing.assert_array_equal(res[ninp.a_nanmask], ninp.b[ninp.a_nanmask])
    # Wherever b is nan, value from a needs to be picked
    np.testing.assert_array_equal(res[ninp.b_nanmask], ninp.a[ninp.b_nanmask])
    # Wherever both are nan, expect nanval
    assert inn.allnan(res[ninp.a_nanmask & ninp.b_nanmask])


def test_nansum(inn, ninp):
    assert inn.nansum(ninp.a) == np.sum(ninp.a[~inn.isnan(ninp.a)])


def test_nancumsum(inn, ninp):
    if ninp.nanstate == "allnans":
        ref = np.full_like(ninp.a, inn.nanval(ninp.a))
    else:
        ref = np.cumsum(inn.fix_invalid(ninp.a))
        nanval = inn.nanval(ninp.a)
        for i, val in enumerate(ninp.a):
            if inn.isnan(val):
                ref[i] = nanval
            else:
                break
    if issubclass(ninp.a.dtype.type, np.float32):
        rtol = 1e-4
    else:
        rtol = 1e-7
    np.testing.assert_allclose(inn.nancumsum(ninp.a), ref, rtol=rtol)


def test_nanprod(inn, ninp):
    if ninp.dtype == np.float32:
        ref_dtype = np.float64
    elif ninp.dtype == np.int32:
        ref_dtype = np.int64
    else:
        ref_dtype = ninp.dtype

    chunk_size = 200
    ref = np.prod(ninp.a[:chunk_size][~inn.isnan(ninp.a[:chunk_size])], dtype=ref_dtype)
    res = inn.nanprod(ninp.a[:chunk_size])
    assert np.isclose(res, ref, rtol=1e-6)


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_nanmean(inn, ninp):
    with warnings.catch_warnings():
        warnings.filterwarnings(ninp.warnings, module="numpy")
        ref = np.mean(ninp.a[~inn.isnan(ninp.a)])
        np.testing.assert_equal(inn.nanmean(ninp.a), ref)


@pytest.mark.parametrize("ddof", [0, 1])
def test_nanstd(inn, ninp, ddof, tolerance=1e-6):
    with warnings.catch_warnings():
        warnings.filterwarnings(ninp.warnings, module="numpy")
        ref = np.std(ninp.a[~inn.isnan(ninp.a)], ddof=ddof)
        np.testing.assert_allclose(inn.nanstd(ninp.a, ddof=ddof), ref, rtol=tolerance)


@pytest.mark.filterwarnings("ignore:Degrees of freedom")
def test_nanvar(inn, ninp, tolerance=1e-6):
    with warnings.catch_warnings():
        warnings.filterwarnings(ninp.warnings, module="numpy")
        ref = np.var(ninp.a[~inn.isnan(ninp.a)])
        np.testing.assert_allclose(inn.nanvar(ninp.a), ref, rtol=tolerance)


def test_nanequal(inn, ninp):
    clone = ninp.a.copy()
    assert np.all(inn.nanequal(ninp.a, clone))
    if ninp.nanstate == "allnans":
        clone[51] = 20
    else:
        clone[51] = inn.nanval(clone)
    assert np.count_nonzero(~inn.nanequal(ninp.a, clone)) == 1
    with np.errstate(invalid="ignore"):
        clone = clone.astype(np.int16)
    pytest.raises(TypeError, inn.nanequal, ninp.a, clone)


def test_nanclose(inn, ninp, tolerance=1e-9):
    clone = ninp.a.copy()
    assert np.all(inn.nanclose(ninp.a, clone, tolerance))

    if issubclass(ninp.a.dtype.type, np.floating):
        clone += np.random.random(ninp.a.shape) * tolerance / 10
        assert np.all(inn.nanclose(ninp.a, clone, tolerance))

    clone[50] = -5
    assert np.count_nonzero(~inn.nanclose(ninp.a, clone, tolerance)) == 1

    if ninp.nanstate == "allnans":
        clone[51] = 20
    else:
        clone[51] = inn.nanval(clone)
    assert np.count_nonzero(~inn.nanclose(ninp.a, clone, tolerance)) == 2
    with np.errstate(invalid="ignore"):
        clone = clone.astype(np.int16)
    pytest.raises(TypeError, inn.nanclose, ninp.a, clone, tolerance)

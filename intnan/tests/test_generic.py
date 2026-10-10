import itertools
import warnings

import numpy as np
import pytest

try:
    from types import SimpleNamespace
except ImportError:

    class SimpleNamespace:
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
    assert inn.nanval(np.dtype("longlong")) == -(2**63)
    assert inn.nanval(np.dtype("uint32")) == 0
    assert np.isnan(inn.nanval(np.ones(10, dtype=np.float64)))
    assert np.isnan(inn.nanval(np.ones(10, dtype=np.float32)))


def test_nanval_dtypes(inn):
    # kind based, independent of the platform dependent size of the C type 'long'
    for dt in (np.int8, np.int16, np.int32, np.int64, np.longlong):
        assert inn.nanval(np.dtype(dt)) == np.iinfo(dt).min
    for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
        assert inn.nanval(np.dtype(dt)) == 0
    for dt in (np.float16, np.float32, np.float64, np.longdouble):
        assert np.isnan(inn.nanval(np.dtype(dt)))
    assert inn.nanval(np.dtype("U5")) == ""
    assert inn.nanval(np.dtype("S5")) == b""
    assert inn.nanval(np.dtype("O")) is None
    assert inn.nanval(np.dtype(bool), -1) == -1


def test_asfloat(inn):
    np.testing.assert_array_equal(inn.asfloat(np.array([True, False])), np.array([1.0, 0.0]))
    np.testing.assert_array_equal(inn.asfloat(np.array([1.0, 0.0, np.nan])), np.array([1.0, 0.0, np.nan]))
    np.testing.assert_array_equal(inn.asfloat(np.array([1, 0, intnan_np.INTNAN64])), np.array([1.0, 0.0, np.nan]))


def test_asint(inn):
    res = inn.asint(np.array([1.0, np.nan, 3.0]))
    assert issubclass(res.dtype.type, np.integer)
    np.testing.assert_array_equal(res, np.array([1, intnan_np.INTNAN64, 3], dtype=np.int64))
    np.testing.assert_array_equal(inn.asint(np.array([True, False])), np.array([1, 0]))
    np.testing.assert_array_equal(inn.asint(np.array([1, intnan_np.INTNAN64])), np.array([1, intnan_np.INTNAN64]))


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


def test_nanargmax(inn, ninp):
    if ninp.nanstate == "allnans":
        pytest.raises(ValueError, inn.nanargmax, ninp.a)
    else:
        valid = ~inn.isnan(ninp.a)
        ref = int(np.flatnonzero(valid & (ninp.a == np.max(ninp.a[valid])))[0])
        assert inn.nanargmax(ninp.a) == ref


def test_nanargmin(inn, ninp):
    if ninp.nanstate == "allnans":
        pytest.raises(ValueError, inn.nanargmin, ninp.a)
    else:
        valid = ~inn.isnan(ninp.a)
        ref = int(np.flatnonzero(valid & (ninp.a == np.min(ninp.a[valid])))[0])
        assert inn.nanargmin(ninp.a) == ref


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
    ref = np.cumsum(inn.fix_invalid(ninp.a))
    nanval = inn.nanval(ref)
    if ninp.nanstate == "allnans":
        ref[:] = nanval
    else:
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


def test_nancumsum_int_promotion(inn):
    # Sums exceeding the int32 range must be accumulated in the platform integer, like numpy does
    a = np.array([2**30, 2**30, 2**30], dtype=np.int32)
    a[2] = intnan_np.nanval(a)
    ref = np.cumsum(inn.fix_invalid(a))
    np.testing.assert_array_equal(inn.nancumsum(a), ref)


def test_nancumprod(inn, ninp):
    ref = np.cumprod(inn.fix_invalid(ninp.a, fill_value=1))
    nanval = inn.nanval(ref)
    if ninp.nanstate == "allnans":
        ref[:] = nanval
    else:
        for i, val in enumerate(ninp.a):
            if inn.isnan(val):
                ref[i] = nanval
            else:
                break
    if issubclass(ninp.a.dtype.type, np.float32):
        rtol = 1e-4
    else:
        rtol = 1e-7
    np.testing.assert_allclose(inn.nancumprod(ninp.a), ref, rtol=rtol)


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


def test_nanmedian(inn, ninp):
    valid = ~inn.isnan(ninp.a)
    if ninp.nanstate == "allnans":
        ref = np.nan
    else:
        ref = np.median(ninp.a[valid])
    np.testing.assert_allclose(inn.nanmedian(ninp.a), ref, rtol=1e-6)


def test_nanmedian_valid_counts(inn):
    # odd number of valid values picks the middle element
    a = np.arange(6, dtype=np.int64)
    a[5] = intnan_np.nanval(a)
    np.testing.assert_equal(inn.nanmedian(a), 2)
    # even number of valid values averages the two middle elements
    a = np.arange(7, dtype=np.int64)
    a[6] = intnan_np.nanval(a)
    np.testing.assert_equal(inn.nanmedian(a), 2.5)


def test_nanarg_duplicates(inn):
    # argmax/argmin return the first occurrence of the extreme value
    a = np.array([1, 5, 3, 5, 2], dtype=np.int64)
    assert inn.nanargmax(a) == 1
    assert inn.nanargmin(a) == 0
    a[0] = intnan_np.nanval(a)
    assert inn.nanargmin(a) == 4


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


ninp2_list = itertools.product(
    ["nonans", "partial", "missingline"],
    [np.int64, np.int32, np.float64, np.float32],
)


@pytest.fixture(params=ninp2_list, ids=lambda x: "-".join((x[0], x[1].__name__)))
def nimat(request):
    nanstate, dtype = request.param
    a = np.arange(24, dtype=dtype).reshape(4, 6)
    mask = np.zeros_like(a, dtype=bool)
    if nanstate != "nonans":
        mask[1, ::2] = True
        mask[::2, 4] = True
    if nanstate == "missingline":
        mask[:, 5] = True  # one all-missing column
        mask[3, :] = True  # one all-missing row
    a[mask] = intnan_np.nanval(a)
    return SimpleNamespace(a=a, mask=mask, nanstate=nanstate, dtype=dtype)


def _accum_reference(inn, a, axis, prod):
    """Independent per-slice reference: accumulate valid values; positions before the first valid value stay missing."""
    nv = inn.nanval(a)
    mask = np.asarray(inn.isnan(a))
    vals = inn.fix_invalid(a, fill_value=1 if prod else 0)
    if axis is None:
        vals = vals.ravel()
        mask = mask.ravel()
        axis_eff = 0
    else:
        axis_eff = axis % vals.ndim
        vals = np.moveaxis(vals, axis_eff, 0)
        mask = np.moveaxis(mask, axis_eff, 0)
    ref = np.empty_like(vals)
    for idx in np.ndindex(vals.shape[1:]):
        acc = None
        for i in range(vals.shape[0]):
            if not mask[(i,) + idx]:
                acc = vals[(i,) + idx] if acc is None else (acc * vals[(i,) + idx] if prod else acc + vals[(i,) + idx])
            ref[(i,) + idx] = nv if acc is None else acc
    if axis is None:
        return ref
    return np.moveaxis(ref, 0, axis_eff)


def assert_like_ref(inn, res, ref, rtol=1e-6):
    """Compare an intnan result against a reference; missing positions must match."""
    res = np.asarray(res)
    ref = np.asarray(ref)
    res_missing = np.asarray(inn.isnan(res))
    ref_missing = np.asarray(inn.isnan(ref))
    np.testing.assert_array_equal(res_missing, ref_missing)
    np.testing.assert_allclose(res[~res_missing], ref[~ref_missing], rtol=rtol)


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nanmin_axis(inn, nimat, axis):
    assert_like_ref(inn, inn.nanmin(nimat.a, axis=axis), np.nanmin(inn.asfloat(nimat.a), axis=axis))


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nanmax_axis(inn, nimat, axis):
    assert_like_ref(inn, inn.nanmax(nimat.a, axis=axis), np.nanmax(inn.asfloat(nimat.a), axis=axis))


def test_nanmin_keepdims(inn, nimat):
    res = inn.nanmin(nimat.a, axis=1, keepdims=True)
    assert res.shape == (nimat.a.shape[0], 1)
    assert_like_ref(inn, res, np.nanmin(inn.asfloat(nimat.a), axis=1, keepdims=True))


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nansum_axis(inn, nimat, axis):
    assert_like_ref(inn, inn.nansum(nimat.a, axis=axis), np.nansum(inn.asfloat(nimat.a), axis=axis))


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nanprod_axis(inn, nimat, axis):
    ref = np.nanprod(inn.asfloat(nimat.a), axis=axis, dtype=np.float64)
    assert_like_ref(inn, inn.nanprod(nimat.a, axis=axis), ref)


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nanmean_axis(inn, nimat, axis):
    assert_like_ref(inn, inn.nanmean(nimat.a, axis=axis), np.nanmean(inn.asfloat(nimat.a), axis=axis))


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nanmedian_axis(inn, nimat, axis):
    assert_like_ref(inn, inn.nanmedian(nimat.a, axis=axis), np.nanmedian(inn.asfloat(nimat.a), axis=axis))


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_nanstd_axis(inn, nimat, axis, ddof):
    ref = np.nanstd(inn.asfloat(nimat.a), axis=axis, ddof=ddof)
    assert_like_ref(inn, inn.nanstd(nimat.a, axis=axis, ddof=ddof), ref)


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_nanvar_axis(inn, nimat, axis, ddof):
    ref = np.nanvar(inn.asfloat(nimat.a), axis=axis, ddof=ddof)
    assert_like_ref(inn, inn.nanvar(nimat.a, axis=axis, ddof=ddof), ref)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_nanargmax_axis(inn, nimat, axis):
    if nimat.nanstate == "missingline":
        # every reduction line hits the all-missing column or row
        pytest.raises(ValueError, inn.nanargmax, nimat.a, axis=axis)
        pytest.raises(ValueError, inn.nanargmin, nimat.a, axis=axis)
    else:
        ref = np.nanargmax(inn.asfloat(nimat.a), axis=axis)
        np.testing.assert_array_equal(inn.nanargmax(nimat.a, axis=axis), ref)
        ref = np.nanargmin(inn.asfloat(nimat.a), axis=axis)
        np.testing.assert_array_equal(inn.nanargmin(nimat.a, axis=axis), ref)


def test_nanargmax_int64_large_values(inn):
    # beyond the float64 mantissa precision a float detour would pick the wrong index
    a = np.array([2**53, 2**53 + 2, 2**53 + 1], dtype=np.int64)
    assert inn.nanargmax(a) == 1
    assert inn.nanargmin(a) == 0


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_anynan_axis(inn, nimat, axis):
    ref = np.any(inn.isnan(nimat.a), axis=axis)
    np.testing.assert_array_equal(inn.anynan(nimat.a, axis=axis), ref)


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_allnan_axis(inn, nimat, axis):
    ref = np.all(inn.isnan(nimat.a), axis=axis)
    np.testing.assert_array_equal(inn.allnan(nimat.a, axis=axis), ref)


def test_nanflags_axis_keepdims(inn, nimat):
    assert inn.anynan(nimat.a, axis=0, keepdims=True).shape == (1, nimat.a.shape[1])
    assert inn.allnan(nimat.a, axis=0, keepdims=True).shape == (1, nimat.a.shape[1])


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nancumsum_axis(inn, nimat, axis):
    assert_like_ref(inn, inn.nancumsum(nimat.a, axis=axis), _accum_reference(inn, nimat.a, axis, prod=False), rtol=1e-5)


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nancumprod_axis(inn, nimat, axis):
    assert_like_ref(inn, inn.nancumprod(nimat.a, axis=axis), _accum_reference(inn, nimat.a, axis, prod=True), rtol=1e-5)


def test_tuple_axis(inn, nimat):
    assert_like_ref(inn, inn.nansum(nimat.a, axis=(0, 1)), np.nansum(inn.asfloat(nimat.a), axis=(0, 1)))
    assert_like_ref(inn, inn.nanmin(nimat.a, axis=(0, 1)), np.nanmin(inn.asfloat(nimat.a), axis=(0, 1)))


def test_axis_unsupported_dtype(inn):
    a = np.array(["a", "", "b"])
    pytest.raises(ValueError, inn.nanmin, a, axis=0)
    a = np.array([1, None, 3], dtype=object)
    pytest.raises(ValueError, inn.nansum, a, axis=0)


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nanpercentile_axis(inn, nimat, axis):
    afloat = inn.asfloat(nimat.a)
    assert_like_ref(inn, inn.nanpercentile(nimat.a, 75, axis=axis), np.nanpercentile(afloat, 75, axis=axis))


def test_nanpercentile_median_parity(inn, nimat):
    np.testing.assert_allclose(inn.nanpercentile(nimat.a, 50, axis=0), inn.nanmedian(nimat.a, axis=0), rtol=1e-6)


def test_nanpercentile_requires_numeric(inn):
    pytest.raises(ValueError, inn.nanpercentile, np.array(["a", "b"]), 50)
    pytest.raises(ValueError, inn.nanquantile, np.array(["a", "b"]), 0.5)


@pytest.mark.parametrize("axis", [0, 1, None])
def test_nanquantile_axis(inn, nimat, axis):
    afloat = inn.asfloat(nimat.a)
    assert_like_ref(inn, inn.nanquantile(nimat.a, 0.25, axis=axis), np.nanquantile(afloat, 0.25, axis=axis))
    np.testing.assert_allclose(
        inn.nanquantile(nimat.a, 0.75, axis=axis), inn.nanpercentile(nimat.a, 75, axis=axis), rtol=1e-6
    )


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nanptp_axis(inn, nimat, axis):
    afloat = inn.asfloat(nimat.a)
    assert_like_ref(inn, inn.nanptp(nimat.a, axis=axis), np.nanmax(afloat, axis=axis) - np.nanmin(afloat, axis=axis))


def test_nanptp_all_missing(inn):
    a = np.array([1, 2, 3], dtype=np.int64)
    a[:] = inn.nanval(a)
    assert inn.nanptp(a) == inn.nanval(a)


@pytest.mark.parametrize("axis", [0, 1, None])
def test_nanaverage_axis(inn, nimat, axis):
    # without weights, nanaverage matches nanmean semantics
    assert_like_ref(inn, inn.nanaverage(nimat.a, axis=axis), np.nanmean(inn.asfloat(nimat.a), axis=axis))


def test_nanaverage_weighted(inn, ninp):
    weights = np.linspace(0.5, 2.0, ninp.a.size)
    valid = ~inn.isnan(ninp.a)
    if ninp.nanstate == "allnans":
        assert np.isnan(inn.nanaverage(ninp.a, weights=weights))
        return
    ref = np.average(ninp.a[valid].astype(np.float64), weights=weights[valid])
    np.testing.assert_allclose(inn.nanaverage(ninp.a, weights=weights), ref, rtol=1e-6)


def test_nanaverage_weighted_1d_axis(inn, nimat):
    weights = np.linspace(0.5, 2.0, nimat.a.shape[0])
    res, wsum = inn.nanaverage(nimat.a, axis=0, weights=weights, returned=True)
    valid = ~np.asarray(inn.isnan(nimat.a))
    w = np.where(valid, np.broadcast_to(weights.reshape(-1, 1), nimat.a.shape), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_avg = np.sum(np.where(valid, nimat.a, 0) * w, axis=0) / np.sum(w, axis=0)
    assert_like_ref(inn, res, ref_avg)
    np.testing.assert_allclose(wsum, np.sum(w, axis=0), rtol=1e-6)


def test_nanaverage_returned(inn, ninp):
    res, wsum = inn.nanaverage(ninp.a, returned=True)
    np.testing.assert_allclose(wsum, int(np.count_nonzero(~inn.isnan(ninp.a))))


def test_nanaverage_1d_weights_require_axis(inn, nimat):
    weights = np.linspace(0.5, 2.0, nimat.a.shape[0])
    pytest.raises(ValueError, inn.nanaverage, nimat.a, weights=weights)


def test_nanclip(inn, ninp):
    valid = ~inn.isnan(ninp.a)
    res = inn.nanclip(ninp.a, 10, 90)
    np.testing.assert_array_equal(res[valid], np.clip(ninp.a[valid], 10, 90))
    np.testing.assert_array_equal(inn.isnan(res), ~valid)


def test_nanclip_bounds_as_arrays(inn, ninp):
    lo = np.full(ninp.a.shape, 10, dtype=ninp.dtype)
    hi = np.full(ninp.a.shape, 90, dtype=ninp.dtype)
    lo[::2] = inn.nanval(lo)  # missing lower bound -> unbounded below
    hi[1::3] = inn.nanval(hi)  # missing upper bound -> unbounded above
    res = inn.nanclip(ninp.a, lo, hi)
    if issubclass(ninp.a.dtype.type, np.floating):
        extreme_lo, extreme_hi = -np.inf, np.inf
    else:
        ii = np.iinfo(ninp.dtype)
        extreme_lo, extreme_hi = ii.min, ii.max
    expected = np.clip(ninp.a, np.where(inn.isnan(lo), extreme_lo, lo), np.where(inn.isnan(hi), extreme_hi, hi))
    valid = ~inn.isnan(ninp.a)
    expected[~valid] = inn.nanval(expected)
    np.testing.assert_array_equal(res, expected)


def test_nanclip_one_sided(inn, ninp):
    valid = ~inn.isnan(ninp.a)
    res = inn.nanclip(ninp.a, None, 90)
    np.testing.assert_array_equal(res[valid], np.minimum(ninp.a[valid], 90))
    res = inn.nanclip(ninp.a, 10, None)
    np.testing.assert_array_equal(res[valid], np.maximum(ninp.a[valid], 10))


def test_nanclip_missing_scalar_bound(inn, ninp):
    valid = ~inn.isnan(ninp.a)
    res = inn.nanclip(ninp.a, np.nan, 90)
    np.testing.assert_array_equal(res[valid], np.minimum(ninp.a[valid], 90))


def test_nanclip_float_bounds_promote(inn):
    a = np.array([1, 5, 10], dtype=np.int32)
    a[1] = inn.nanval(a)
    res = inn.nanclip(a, 2.5, 7.5)
    # np.clip promotes to float64, missing values become NaN
    np.testing.assert_array_equal(res, np.array([2.5, np.nan, 7.5]))
    assert inn.isnan(res[1])


def test_nanclip_requires_numeric(inn):
    pytest.raises(ValueError, inn.nanclip, np.array(["a", "b"]), 1, 2)
    pytest.raises(ValueError, inn.nanclip, np.arange(3), "a", 2)


def _first_last_reference(inn, a, mask, axis, last):
    """Independent per-slice reference for nanfirst/nanlast."""
    nv = inn.nanval(a)
    ax = axis % a.ndim
    shape = list(a.shape)
    del shape[ax]
    ref = np.empty(shape, dtype=a.dtype)
    for idx in np.ndindex(*shape):
        src = list(idx)
        src.insert(ax, slice(None))
        col = a[tuple(src)]
        colmask = mask[tuple(src)]
        order = range(len(col) - 1, -1, -1) if last else range(len(col))
        pick = nv
        for i in order:
            if not colmask[i]:
                pick = col[i]
                break
        ref[idx] = pick
    return ref


def test_nancount(inn, ninp):
    assert inn.nancount(ninp.a) == int(np.count_nonzero(~inn.isnan(ninp.a)))


@pytest.mark.parametrize("axis", [0, 1, -1, None])
def test_nancount_axis(inn, nimat, axis):
    ref = np.sum(~np.asarray(inn.isnan(nimat.a)), axis=axis)
    np.testing.assert_array_equal(inn.nancount(nimat.a, axis=axis), ref)


def test_nancount_keepdims(inn, nimat):
    assert inn.nancount(nimat.a, axis=0, keepdims=True).shape == (1, nimat.a.shape[1])


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_nanfirst_axis(inn, nimat, axis):
    res = inn.nanfirst(nimat.a, axis=axis)
    ref = _first_last_reference(inn, nimat.a, nimat.mask, axis, last=False)
    assert_like_ref(inn, res, ref)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_nanlast_axis(inn, nimat, axis):
    res = inn.nanlast(nimat.a, axis=axis)
    ref = _first_last_reference(inn, nimat.a, nimat.mask, axis, last=True)
    assert_like_ref(inn, res, ref)


def test_nanfirst_last_flat(inn, nimat):
    flat_valid = np.flatnonzero(~nimat.mask.ravel())
    if len(flat_valid):
        assert inn.nanfirst(nimat.a, axis=None) == nimat.a.ravel()[flat_valid[0]]
        assert inn.nanlast(nimat.a, axis=None) == nimat.a.ravel()[flat_valid[-1]]


def test_nanfirst_last_all_missing(inn):
    a = np.arange(3, dtype=np.int64)
    a[:] = inn.nanval(a)
    assert inn.nanfirst(a) == inn.nanval(a)
    assert inn.nanlast(a) == inn.nanval(a)


def test_object_arrays(inn):
    a = np.array([1, None, 3], dtype=object)
    assert inn.nanmax(a) == 3
    assert inn.nanmin(a) == 1
    assert inn.nancount(a) == 2
    assert inn.nanptp(a) == 2
    # functions reducing along an axis require numeric dtypes
    pytest.raises(ValueError, inn.nanfirst, a)


def test_nanpercentile_array_q(inn, nimat):
    afloat = inn.asfloat(nimat.a)
    res = inn.nanpercentile(nimat.a, [25, 50, 75], axis=0)
    assert_like_ref(inn, res, np.nanpercentile(afloat, [25, 50, 75], axis=0))


def test_nanpercentile_scalar_positions(inn):
    a = np.arange(101)
    assert inn.nanpercentile(a, 0) == 0
    assert inn.nanpercentile(a, 100) == 100
    assert inn.nanpercentile(a, 50) == 50.0

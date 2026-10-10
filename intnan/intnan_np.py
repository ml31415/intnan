"""
Module for handling integer arrays with missing values.
Missing values are handled as special values and functions
to skip them in processing are provided here.

The special nan values are chosen depending on the array type.
Large negative values are used, so that especially python indexing
(from the end) is unlikely to work.
"""

from typing import Any, cast, overload

import numpy as np
import numpy.typing as npt

__all__ = [
    "NANVALS",
    "INTNAN32",
    "INTNAN64",
    "nanval",
    "isnan",
    "fix_invalid",
    "asfloat",
    "asint",
    "allnan",
    "anynan",
    "nancount",
    "nanfirst",
    "nanlast",
    "nanmin",
    "nanmax",
    "nanargmax",
    "nanargmin",
    "nanmaximum",
    "nanminimum",
    "nanclip",
    "nanmean",
    "nanmedian",
    "nanstd",
    "nanvar",
    "nanpercentile",
    "nanquantile",
    "nanptp",
    "nanaverage",
    "nansum",
    "nancumsum",
    "nancumprod",
    "nanprod",
    "nanequal",
    "nanclose",
]

type NanValue = float | int | str | bytes | None
type Axis = int | tuple[int, ...] | None

# Dtype kinds for which reductions along axes are supported
_NUMERIC_KINDS = "iuf"

INTNAN32 = np.iinfo("int32").min  # -2147483648
INTNAN64 = np.iinfo("int64").min  # -9223372036854775808

# NAN values keyed by dtype char. Note: the size of the C types behind the
# chars 'l' and 'L' (long) is platform dependent (64 bit on Linux/macOS,
# 32 bit on Windows), so this mapping is only reliable on Linux/macOS.
# Use nanval(), which works on dtype kinds and is platform independent.
NANVALS: dict[str, NanValue] = dict(
    d=np.nan,
    f=np.nan,
    e=np.nan,
    g=np.nan,
    S=b"",
    U="",
    l=INTNAN64,
    q=INTNAN64,
    i=INTNAN32,
    I=0,
    b=-1,
    h=-1,
    B=0,
    H=0,
    L=0,
    Q=0,
    O=None,
)


def nanval(x: npt.NDArray | npt.DTypeLike, default: NanValue = 0) -> NanValue:
    """Return the corresponding NAN value for a column or type"""
    dtype = x.dtype if isinstance(x, np.ndarray) else np.dtype(x)
    if dtype.kind == "f":
        return np.nan
    elif dtype.kind == "i":
        return np.iinfo(dtype).min
    elif dtype.kind == "u":
        return 0
    elif dtype.kind == "U":
        return ""
    elif dtype.kind == "S":
        return b""
    elif dtype.kind == "O":
        return None
    return default


@overload
def isnan(x: npt.NDArray) -> npt.NDArray[np.bool_]: ...
@overload
def isnan(x: np.generic | float | int | complex | str | bytes | None) -> bool: ...
def isnan(x: npt.NDArray | np.generic | float | int | complex | str | bytes | None) -> bool | npt.NDArray[np.bool_]:
    if isinstance(x, np.ndarray):
        nv = nanval(x)
        if nv is np.nan:
            return np.isnan(x)
        elif nv is None:
            return np.array([val is None for val in x], dtype=np.bool_)
        else:
            return x == nv
    elif x in {np.nan, None, b"", "", INTNAN32, INTNAN64}:
        return True
    else:
        try:
            return bool(np.isnan(x))
        except TypeError:
            # Non-numeric values are never NaN
            return False


def fix_invalid(x: npt.NDArray, copy: bool = True, fill_value: Any = 0) -> npt.NDArray:
    nv = nanval(x)
    if nv is np.nan:
        if copy:
            return np.where(np.isnan(x), fill_value, x)
        else:
            x[np.isnan(x)] = fill_value
            return x
    elif nv is None:
        ret = np.zeros_like(x)
        x_flat = x.flat
        for i in range(x.size):
            if x_flat[i] is None:
                ret[i] = fill_value
            else:
                ret[i] = x_flat[i]
        return ret
    else:
        if copy:
            return np.where(x == nv, fill_value, x)
        else:
            x[x == nv] = fill_value
            return x


def asfloat(x: npt.NDArray) -> npt.NDArray[np.floating]:
    if issubclass(x.dtype.type, np.floating):
        return x.copy()
    elif issubclass(x.dtype.type, np.bool_):
        return np.array(x, dtype=float)
    return fix_invalid(x, fill_value=np.nan)


def asint(x: npt.NDArray) -> npt.NDArray[np.integer]:
    if issubclass(x.dtype.type, np.integer):
        return x.copy()
    elif issubclass(x.dtype.type, np.bool_):
        return np.array(x, dtype=int)
    nv = cast(int, nanval(int))
    return np.nan_to_num(x, nan=nv, posinf=nv, neginf=nv).astype(int)


def anynan(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Test if any value is missing; with axis, test per slice."""
    nv = nanval(x)
    if axis is None and not keepdims:
        if nv is None:
            return any(val is None for val in x.flat)
        if nv is np.nan:
            return bool(np.any(np.isnan(x)))
        return bool(nv in x)
    return np.any(isnan(x), axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def allnan(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Test if all values are missing; with axis, test per slice."""
    nv = nanval(x)
    if axis is None and not keepdims:
        if nv is None:
            return all(val is None for val in x.flat)
        if nv is np.nan:
            return bool(np.all(np.isnan(x)))
        return bool(np.all(x == nv))
    return np.all(isnan(x), axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def nancount(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Number of valid values; with axis, count per slice."""
    mask = isnan(x)
    if axis is None and not keepdims:
        return int(np.count_nonzero(~mask))
    return np.sum(~mask, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]


def nanfirst(x: npt.NDArray, axis: int | None = 0) -> Any:
    """First valid value along axis (default 0, None for flattened); all-missing slices yield the missing value."""
    nv = nanval(x)
    mask = ~isnan(x)
    if axis is None:
        flat = np.flatnonzero(mask.ravel())
        return x.ravel()[flat[0]] if len(flat) else nv
    _check_axis_support(x, axis)
    if not np.any(mask):
        return nv
    first = np.argmax(mask, axis=axis)
    ret = np.take_along_axis(x, np.expand_dims(first, axis), axis=axis)
    ret = np.squeeze(ret, axis=axis)
    has_valid = np.any(mask, axis=axis)
    return np.where(has_valid, ret, cast("float | int", nv))


def nanlast(x: npt.NDArray, axis: int | None = -1) -> Any:
    """Last valid value along axis (default -1, None for flattened); all-missing slices yield the missing value."""
    nv = nanval(x)
    mask = ~isnan(x)
    if axis is None:
        flat = np.flatnonzero(mask.ravel())
        return x.ravel()[flat[-1]] if len(flat) else nv
    _check_axis_support(x, axis)
    if not np.any(mask):
        return nv
    last = x.shape[axis] - 1 - np.argmax(np.flip(mask, axis=axis), axis=axis)
    ret = np.take_along_axis(x, np.expand_dims(last, axis), axis=axis)
    ret = np.squeeze(ret, axis=axis)
    has_valid = np.any(mask, axis=axis)
    return np.where(has_valid, ret, cast("float | int", nv))  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def _extreme_value(x: npt.NDArray, high: bool) -> Any:
    """Value that never wins a min/max reduction against any valid entry."""
    if x.dtype.kind == "f":
        return np.inf if high else -np.inf
    ii = np.iinfo(x.dtype)
    return ii.max if high else ii.min


def _check_axis_support(x: npt.NDArray, axis: Axis) -> None:
    if axis is not None and x.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"axis is only supported for numeric dtypes, not for {x.dtype!r}")


def _squeeze_reduced(ret: npt.NDArray, axis: Axis, keepdims: bool) -> Any:
    """Undo the internal keepdims=True shape of a reduction."""
    if keepdims:
        return ret
    if axis is None:
        return ret[()]
    return np.squeeze(ret, axis=axis)


def _masked_mean(x: npt.NDArray, mask: npt.NDArray[np.bool_], axis: Axis) -> npt.NDArray:
    """Mean over the valid values with reduced dimensions kept; all-missing slices yield NaN."""
    valid = np.sum(~mask, axis=axis, keepdims=True)
    total = np.sum(x, axis=axis, where=~mask, dtype=np.float64, keepdims=True)
    return np.divide(total, valid, out=np.full_like(total, np.nan), where=valid > 0)


def _leading_invalid(
    x: npt.NDArray, first: npt.NDArray, has_valid: npt.NDArray[np.bool_], axis: int
) -> npt.NDArray[np.bool_]:
    """Boolean mask of the positions before the first valid value along axis."""
    ax = axis if axis >= 0 else axis + x.ndim
    grid_shape = [1] * x.ndim
    grid_shape[ax] = x.shape[ax]
    pos = np.arange(x.shape[ax], dtype=np.intp).reshape(grid_shape)
    lead = pos < np.expand_dims(first, ax)
    return lead | ~np.expand_dims(has_valid, ax)


def nanmax(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Maximum over the valid values; all-missing slices yield the missing value."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nanmax(x, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload
    mask = isnan(x)
    if axis is None and not keepdims:
        try:
            return np.max(x[~mask])
        except ValueError as e:
            if "zero-size" in str(e):
                return nv
            raise
    _check_axis_support(x, axis)
    filled = np.where(mask, _extreme_value(x, high=False), x)
    ret = np.max(filled, axis=axis, keepdims=True)
    valid = np.sum(~mask, axis=axis, keepdims=True)
    ret = np.where(valid == 0, cast("float | int", nv), ret)
    return _squeeze_reduced(ret, axis, keepdims)


def nanmin(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Minimum over the valid values; all-missing slices yield the missing value."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nanmin(x, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload
    mask = isnan(x)
    if axis is None and not keepdims:
        try:
            return np.min(x[~mask])
        except ValueError as e:
            if "zero-size" in str(e):
                return nv
            raise
    _check_axis_support(x, axis)
    filled = np.where(mask, _extreme_value(x, high=True), x)
    ret = np.min(filled, axis=axis, keepdims=True)
    valid = np.sum(~mask, axis=axis, keepdims=True)
    ret = np.where(valid == 0, cast("float | int", nv), ret)
    return _squeeze_reduced(ret, axis, keepdims)


def nanargmax(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Index of the maximum over the valid values; raises on all-missing slices like numpy."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nanargmax(x, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload
    mask = isnan(x)
    filled = np.where(mask, _extreme_value(x, high=False), x)
    if axis is None and not keepdims:
        if not np.any(~mask):
            raise ValueError("All-NaN slice encountered")
        return np.argmax(filled)
    _check_axis_support(x, axis)
    if not np.all(np.any(~mask, axis=axis, keepdims=True)):
        raise ValueError("All-NaN slice encountered")
    return np.argmax(filled, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def nanargmin(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Index of the minimum over the valid values; raises on all-missing slices like numpy."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nanargmin(x, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload
    mask = isnan(x)
    filled = np.where(mask, _extreme_value(x, high=True), x)
    if axis is None and not keepdims:
        if not np.any(~mask):
            raise ValueError("All-NaN slice encountered")
        return np.argmin(filled)
    _check_axis_support(x, axis)
    if not np.all(np.any(~mask, axis=axis, keepdims=True)):
        raise ValueError("All-NaN slice encountered")
    return np.argmin(filled, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def nanmaximum(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Does the same as numpy.maximum (element-wise maximum operation of two arrays) but ignores NaNs"""
    z = np.maximum(x, y)
    badx = isnan(x)
    bady = isnan(y)
    z[badx] = y[badx]
    z[bady] = x[bady]
    return z


def nanminimum(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Does the same as numpy.minimum (element-wise minimum operation of two arrays) but ignores NaNs"""
    z = np.minimum(x, y)
    badx = isnan(x)
    bady = isnan(y)
    z[badx] = y[badx]
    z[bady] = x[bady]
    return z


def nansum(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Sum over the valid values."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nansum(x, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload
    mask = isnan(x)
    if axis is None and not keepdims:
        return np.sum(x[~mask])
    _check_axis_support(x, axis)
    return np.sum(fix_invalid(x), axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def nanprod(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Product over the valid values."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nanprod(x, axis=axis, keepdims=keepdims, dtype=np.float64)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload
    mask = isnan(x)
    if axis is None and not keepdims:
        return np.prod(x[~mask])
    _check_axis_support(x, axis)
    return np.prod(fix_invalid(x, fill_value=1), axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def nancumsum(x: npt.NDArray, axis: int | None = None) -> npt.NDArray:
    """Cumulative sum over the valid values; positions before the first valid value stay missing."""
    mask = isnan(x)
    if axis is None:
        ret = np.cumsum(fix_invalid(x))
        nv_ret = nanval(ret)
        if np.any(mask):
            good_idx = np.flatnonzero(~mask.ravel())
            if len(good_idx) == 0:
                ret[:] = nv_ret
            elif good_idx[0] > 0:
                ret[: good_idx[0]] = nv_ret
        return ret
    _check_axis_support(x, axis)
    ret = np.cumsum(fix_invalid(x), axis=axis)
    first = np.argmax(~mask, axis=axis)
    has_valid = np.any(~mask, axis=axis)
    lead = _leading_invalid(x, first, has_valid, axis)
    return np.where(lead, cast("float | int", nanval(ret)), ret)


def nancumprod(x: npt.NDArray, axis: int | None = None) -> npt.NDArray:
    """Cumulative product over the valid values; positions before the first valid value stay missing."""
    mask = isnan(x)
    if axis is None:
        ret = np.cumprod(fix_invalid(x, fill_value=1))
        nv_ret = nanval(ret)
        if np.any(mask):
            good_idx = np.flatnonzero(~mask.ravel())
            if len(good_idx) == 0:
                ret[:] = nv_ret
            elif good_idx[0] > 0:
                ret[: good_idx[0]] = nv_ret
        return ret
    _check_axis_support(x, axis)
    ret = np.cumprod(fix_invalid(x, fill_value=1), axis=axis)
    first = np.argmax(~mask, axis=axis)
    has_valid = np.any(~mask, axis=axis)
    lead = _leading_invalid(x, first, has_valid, axis)
    return np.where(lead, cast("float | int", nanval(ret)), ret)


def nanmean(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Arithmetic mean over the valid values; all-missing slices yield NaN."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nanmean(x, axis=axis, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload
    mask = isnan(x)
    if axis is None and not keepdims:
        with np.errstate(invalid="ignore"):
            return np.mean(x[~mask])
    _check_axis_support(x, axis)
    return _squeeze_reduced(_masked_mean(x, mask, axis), axis, keepdims)


def nanmedian(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Median over the valid values; all-missing slices yield NaN."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nanmedian(x, axis=axis, keepdims=keepdims)
    if axis is None and not keepdims:
        with np.errstate(invalid="ignore"):
            return np.median(x[~isnan(x)])
    _check_axis_support(x, axis)
    return np.nanmedian(asfloat(x), axis=axis, keepdims=keepdims)


def nanvar(x: npt.NDArray, axis: Axis = None, ddof: int = 0, keepdims: bool = False) -> Any:
    """Variance over the valid values; slices without enough valid values yield NaN."""
    nv = nanval(x)
    if nv is np.nan:
        return np.nanvar(x, axis=axis, ddof=ddof, keepdims=keepdims)  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload
    mask = isnan(x)
    if axis is None and not keepdims:
        with np.errstate(invalid="ignore"):
            return np.var(x[~mask], ddof=ddof)
    _check_axis_support(x, axis)
    valid = np.sum(~mask, axis=axis, keepdims=True)
    mu = _masked_mean(x, mask, axis)
    dev = np.where(mask, 0.0, x - mu)
    s2 = np.sum(dev * dev, axis=axis, keepdims=True)
    dof = valid - ddof
    ret = np.divide(s2, dof, out=np.full_like(s2, np.nan), where=dof > 0)
    return _squeeze_reduced(ret, axis, keepdims)


def nanstd(x: npt.NDArray, axis: Axis = None, ddof: int = 0, keepdims: bool = False) -> Any:
    """Standard deviation over the valid values; slices without enough valid values yield NaN."""
    return np.sqrt(nanvar(x, axis=axis, ddof=ddof, keepdims=keepdims))


def nanpercentile(
    x: npt.NDArray,
    q: npt.ArrayLike,
    axis: Axis = None,
    out: npt.NDArray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Any:
    """Percentiles over the valid values; all-missing slices yield NaN. Always returns float64."""
    if x.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"nanpercentile requires numeric dtypes, not for {x.dtype!r}")
    return np.nanpercentile(
        np.asarray(asfloat(x), dtype=np.float64),
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def nanquantile(
    x: npt.NDArray,
    q: npt.ArrayLike,
    axis: Axis = None,
    out: npt.NDArray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Any:
    """Quantiles over the valid values; all-missing slices yield NaN. Always returns float64."""
    if x.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"nanquantile requires numeric dtypes, not for {x.dtype!r}")
    return np.nanquantile(
        np.asarray(asfloat(x), dtype=np.float64),
        q,
        axis=axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )  # type: ignore[call-overload]  # numpy-stubs: no union-axis + bool keepdims overload


def nanptp(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    """Peak-to-peak (maximum - minimum) over the valid values; all-missing slices yield the missing value."""
    nv = nanval(x)
    if x.dtype.kind == "O":
        _check_axis_support(x, axis)
        if allnan(x):
            return nv
        return cast("float | int", nanmax(x)) - cast("float | int", nanmin(x))
    mask = isnan(x)
    mx = nanmax(x, axis=axis, keepdims=True)
    mn = nanmin(x, axis=axis, keepdims=True)
    valid = np.sum(~mask, axis=axis, keepdims=True)
    ret = np.where(valid == 0, cast("float | int", nv), mx - mn)
    return _squeeze_reduced(ret, axis, keepdims)


def _broadcast_weights(weights: npt.ArrayLike | None, x: npt.NDArray, axis: Axis) -> npt.NDArray:
    """Broadcast weights against x, supporting 1d weights along the reduction axis like np.average."""
    w = np.asarray(weights if weights is not None else 1.0, dtype=np.float64)
    if w.ndim == 1 and x.ndim > 1:
        if not isinstance(axis, int):
            raise ValueError("1d weights require an integer axis for multidimensional arrays")
        if w.shape[0] != x.shape[axis]:
            raise ValueError("weights must have the same length as the reduced axis")
        shape = [1] * x.ndim
        shape[axis] = w.shape[0]
        return np.broadcast_to(w.reshape(shape), x.shape)
    return np.broadcast_to(w, x.shape)


def nanaverage(
    x: npt.NDArray,
    axis: Axis = None,
    weights: npt.ArrayLike | None = None,
    returned: bool = False,
    keepdims: bool = False,
) -> Any:
    """Weighted mean over the valid values; weights at missing positions are excluded, all-missing slices yield NaN.

    Accumulation happens in float64, so the result is always a float array.
    """
    _check_axis_support(x, axis)
    mask = isnan(x)
    w = np.where(mask, 0, _broadcast_weights(weights, x, axis))
    xw = np.where(mask, 0, x) * w
    wsum = np.sum(w, axis=axis, keepdims=True)  # type: ignore[call-overload]
    total = np.sum(xw, axis=axis, keepdims=True)  # type: ignore[call-overload]
    ret = np.divide(total, wsum, out=np.full_like(total, np.nan), where=wsum > 0)
    ret = _squeeze_reduced(ret, axis, keepdims)
    if returned:
        return ret, _squeeze_reduced(wsum, axis, keepdims)
    return ret


def _clip_bound(bound: npt.ArrayLike | None, x: npt.NDArray, low: bool) -> Any:
    """Translate a clip bound for np.clip, treating missing bounds as unbounded."""
    if bound is None:
        return _extreme_value(x, high=not low)
    b = np.asarray(bound)
    if b.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"clip bounds must be numeric or None, not {b.dtype!r}")
    missing = isnan(b)
    if not np.any(missing):
        return b
    return np.where(missing, _extreme_value(x, high=not low), b)


def nanclip(x: npt.NDArray, a_min: npt.ArrayLike | None = None, a_max: npt.ArrayLike | None = None) -> npt.NDArray:
    """Clip the valid values like np.clip; missing values are preserved and missing bounds mean unbounded.

    Like np.clip, float bounds promote integer arrays to float, in which case missing values
    are marked with NaN afterwards.
    """
    if x.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(f"nanclip requires numeric dtypes, not for {x.dtype!r}")
    ret = np.clip(x, _clip_bound(a_min, x, low=True), _clip_bound(a_max, x, low=False))
    ret[isnan(x)] = nanval(ret)
    return ret


def nanequal(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray[np.bool_]:
    """Treat NaN as an ordinary value when comparing for equality."""
    if x.dtype != y.dtype:
        raise TypeError(f"nanequal requires same data type: {x.dtype} != {y.dtype}")
    if issubclass(x.dtype.type, np.floating) and issubclass(y.dtype.type, np.floating):
        return np.isclose(x, y, 0, 0, equal_nan=True)
    else:
        return x == y


def nanclose(x: npt.NDArray, y: npt.NDArray, delta: float = np.finfo(float).eps) -> npt.NDArray[np.bool_]:
    if x.dtype != y.dtype:
        raise TypeError(f"nanclose requires same data type: {x.dtype} != {y.dtype}")
    if issubclass(x.dtype.type, np.integer):
        return np.isclose(x, y, atol=delta, equal_nan=True)
    elif issubclass(x.dtype.type, str):
        return x == y
    else:
        return np.isclose(x, y, atol=delta, equal_nan=True)

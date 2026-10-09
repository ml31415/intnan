"""Numba accelerated implementations of the intnan functions.

Automatically selected on import when numba is available.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import numba as nb
import numpy as np
import numpy.typing as npt

from . import intnan_np as _impl
from .intnan_np import (
    INTNAN32,
    INTNAN64,
    NANVALS,
    Axis,
    __all__,
    asfloat,
    asint,
    isnan,
    nanclose,
    nanequal,
    nanval,
)


def _via_impl(axis: Any, keepdims: bool = False) -> bool:
    """True when the axis-aware numpy implementation has to be used instead of the JIT kernel."""
    return axis is not None or keepdims


def nancalc(func: Callable[..., Any]) -> Callable[..., Any]:
    jfunc = nb.njit(func, cache=True)

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nv = nanval(args[0])
        args = args + (nv,)
        return jfunc(*args, **kwargs)

    return wrapped


@nb.njit(cache=True)
def isnan_vec(x: npt.NDArray, nan: Any) -> npt.NDArray[np.bool_]:
    return (x == nan) | (x != x)


@nancalc
def fix_invalid(x: npt.NDArray, nan: Any, copy: bool = True, fill_value: Any = 0) -> npt.NDArray:
    if copy:
        ret = np.empty_like(x)
    else:
        ret = x
    for i in nb.prange(len(x.flat)):
        if isnan_vec(x.flat[i], nan):
            ret.flat[i] = fill_value
        else:
            ret.flat[i] = x.flat[i]
    return ret


@nancalc
def _allnan(x: npt.NDArray, nan: Any) -> bool:
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            return False
    return True


@nancalc
def _anynan(x: npt.NDArray, nan: Any) -> bool:
    for x_ in x.flat:
        if isnan_vec(x_, nan):
            return True
    return False


@nancalc
def _nanmax(x: npt.NDArray, nan: Any) -> Any:
    cmp_val = nan
    for x_ in x.flat:
        if isnan_vec(x_, nan):
            continue
        if isnan_vec(cmp_val, nan):
            cmp_val = x_
        elif x_ > cmp_val:
            cmp_val = x_
    return cmp_val


@nancalc
def _nanmin(x: npt.NDArray, nan: Any) -> Any:
    cmp_val = nan
    for x_ in x.flat:
        if isnan_vec(x_, nan):
            continue
        if isnan_vec(cmp_val, nan):
            cmp_val = x_
        elif x_ < cmp_val:
            cmp_val = x_
    return cmp_val


@nancalc
def _nanargmax(x: npt.NDArray, nan: Any) -> Any:
    idx = -1
    cmp_val = nan
    for i, x_ in enumerate(x.flat):
        if isnan_vec(x_, nan):
            continue
        if isnan_vec(cmp_val, nan) or x_ > cmp_val:
            cmp_val = x_
            idx = i
    if idx == -1:
        raise ValueError("All-NaN slice encountered")
    return idx


@nancalc
def _nanargmin(x: npt.NDArray, nan: Any) -> Any:
    idx = -1
    cmp_val = nan
    for i, x_ in enumerate(x.flat):
        if isnan_vec(x_, nan):
            continue
        if isnan_vec(cmp_val, nan) or x_ < cmp_val:
            cmp_val = x_
            idx = i
    if idx == -1:
        raise ValueError("All-NaN slice encountered")
    return idx


@nancalc
def nanmaximum(x: npt.NDArray, y: npt.NDArray, nan: Any) -> npt.NDArray:
    if len(x) != len(y):
        raise ValueError("input arrays must be of equal length")
    ret = np.full_like(x, nan)
    for i in nb.prange(len(x)):
        if isnan_vec(x[i], nan):
            ret[i] = y[i]
        elif isnan_vec(y[i], nan):
            ret[i] = x[i]
        elif x[i] > y[i]:
            ret[i] = x[i]
        else:
            ret[i] = y[i]
    return ret


@nancalc
def nanminimum(x: npt.NDArray, y: npt.NDArray, nan: Any) -> npt.NDArray:
    if len(x) != len(y):
        raise ValueError("input arrays must be of equal length")
    ret = np.full_like(x, nan)
    for i in nb.prange(len(x)):
        if isnan_vec(x[i], nan):
            ret[i] = y[i]
        elif isnan_vec(y[i], nan):
            ret[i] = x[i]
        elif x[i] < y[i]:
            ret[i] = x[i]
        else:
            ret[i] = y[i]
    return ret


@nancalc
def _nansum(x: npt.NDArray, nan: Any) -> Any:
    ret = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            ret += x_
    return ret


@nancalc
def _nanprod(x: npt.NDArray, nan: Any) -> Any:
    ret = 1
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            ret *= x_
    return ret


def _nancumsum(x: npt.NDArray, nan: Any) -> npt.NDArray:
    ret = np.full_like(x, nan)
    val = nan
    started = False
    for i, x_ in enumerate(x.flat):
        if not isnan_vec(x_, nan):
            if started:
                val = val + x_
            else:
                val = x_
                started = True
        if started:
            ret[i] = val
    return ret


_jnancumsum = nb.njit(_nancumsum, cache=True)


def _platform_int(x: npt.NDArray) -> npt.NDArray:
    """Upcast integer arrays below platform integer width, like numpy accumulations do, and translate the marker."""
    if issubclass(x.dtype.type, np.integer) and x.dtype.itemsize < 8:
        up = x.astype(np.int64)
        up[up == nanval(x)] = cast("int", nanval(up))
        return up
    return x


def nancumsum(x: npt.NDArray, axis: int | None = None) -> npt.NDArray:
    if axis is None and x.ndim == 1:
        return _jnancumsum(_platform_int(x), nanval(_platform_int(x)))
    return _impl.nancumsum(x, axis=axis)


def _nancumprod(x: npt.NDArray, nan: Any) -> npt.NDArray:
    ret = np.full_like(x, nan)
    val = nan
    for i, x_ in enumerate(x.flat):
        if not isnan_vec(x_, nan):
            if isnan_vec(val, nan):
                val = x_
            else:
                val = val * x_
        if not isnan_vec(val, nan):
            ret[i] = val
    return ret


_jnancumprod = nb.njit(_nancumprod, cache=True)


def nancumprod(x: npt.NDArray, axis: int | None = None) -> npt.NDArray:
    if axis is None and x.ndim == 1:
        return _jnancumprod(_platform_int(x), nanval(_platform_int(x)))
    return _impl.nancumprod(x, axis=axis)


@nancalc
def _nanmean(x: npt.NDArray, nan: Any) -> Any:
    ret = 0.0
    cnt = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            cnt += 1
            ret += x_
    return np.divide(ret, cnt)


def _nanvar(x: npt.NDArray, nan: Any, ddof: int = 0) -> Any:
    ret = 0.0
    cnt = 0
    # Inline that loop from nanmean, so that we can reuse cnt
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            cnt += 1
            ret += x_
    if cnt == ddof or cnt == 0:
        return np.nan
    mean = ret / cnt
    ex_mean = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            inc = x_ - mean
            ex_mean += inc * inc
    return ex_mean / (cnt - ddof)


_jnanvar = nb.njit(_nanvar, cache=True)


def nanvar(x: npt.NDArray, axis: Axis = None, ddof: int = 0, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanvar(x, axis=axis, ddof=ddof, keepdims=keepdims)
    return _jnanvar(x, nanval(x), ddof=ddof)


@nancalc
def _nanmedian(x: npt.NDArray, nan: Any) -> Any:
    cnt = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            cnt += 1
    if cnt == 0:
        return np.nan
    tmp = np.empty(cnt, dtype=x.dtype)
    i = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            tmp[i] = x_
            i += 1
    tmp.sort()
    if cnt % 2 == 1:
        return tmp[cnt // 2]
    return (tmp[cnt // 2 - 1] + tmp[cnt // 2]) / 2


@nancalc
def _nanstd(x: npt.NDArray, nan: Any, ddof: int = 0) -> Any:
    return np.sqrt(_jnanvar(x, nan, ddof=ddof))


def nanstd(x: npt.NDArray, axis: Axis = None, ddof: int = 0, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanstd(x, axis=axis, ddof=ddof, keepdims=keepdims)
    return _nanstd(x, ddof=ddof)


def allnan(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.allnan(x, axis=axis, keepdims=keepdims)
    return _allnan(x)


def anynan(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.anynan(x, axis=axis, keepdims=keepdims)
    return _anynan(x)


def nanmax(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanmax(x, axis=axis, keepdims=keepdims)
    return _nanmax(x)


def nanmin(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanmin(x, axis=axis, keepdims=keepdims)
    return _nanmin(x)


def nanargmax(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanargmax(x, axis=axis, keepdims=keepdims)
    return _nanargmax(x)


def nanargmin(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanargmin(x, axis=axis, keepdims=keepdims)
    return _nanargmin(x)


def nansum(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nansum(x, axis=axis, keepdims=keepdims)
    return _nansum(x)


def nanprod(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanprod(x, axis=axis, keepdims=keepdims)
    return _nanprod(x)


def nanmean(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanmean(x, axis=axis, keepdims=keepdims)
    return _nanmean(x)


def nanmedian(x: npt.NDArray, axis: Axis = None, keepdims: bool = False) -> Any:
    if _via_impl(axis, keepdims):
        return _impl.nanmedian(x, axis=axis, keepdims=keepdims)
    return _nanmedian(x)

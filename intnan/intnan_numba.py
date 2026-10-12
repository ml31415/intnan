"""Numba accelerated implementations of the intnan functions.

Automatically selected on import when numba is available.
"""

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

import numba as nb
import numpy as np
import numpy.typing as npt

from . import intnan_np as _impl
from .intnan_np import (
    _NUMERIC_KINDS,
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


def _memory_order_view(x: npt.NDArray) -> npt.NDArray:
    """Return a view over x whose flat order is its memory order.

    Scanning a Fortran ordered array in flat order walks it against its strides, which costs
    a cache miss per element; the transposed view is C ordered, so the same scan is sequential.
    Only valid for operations that do not depend on the flat element order.
    """
    if x.ndim > 1 and not x.flags.c_contiguous and x.flags.f_contiguous:
        return x.T
    return x


def dispatch_ndim(
    func: Callable[..., Any] | None = None, *, axis_default: int | None = None, layout_free: bool = False
) -> Any:
    """Wrap a flat JIT kernel into the axis-aware public function of the same name.

    Calls with axis or keepdims (and object arrays) are routed to the numpy implementation of
    the same name; the kernel handles the flattened case and receives the missing value of the
    input appended after x. Kernels that other jitted functions call internally must be
    exposed through a separate private dispatcher instead.

    Kernels of layout free functions do not depend on the flat element order, so a Fortran
    ordered input is handed to them transposed to keep the scan in memory order. Keep the
    check inline rather than calling _memory_order_view, as the per call overhead shows.
    """

    def build(kernel: Callable[..., Any]) -> Callable[..., Any]:
        impl: Any = getattr(_impl, kernel.__name__)
        with_keepdims = "keepdims" in inspect.signature(impl).parameters

        def wrapper(
            x: npt.NDArray, axis: Axis = axis_default, keepdims: bool = False, *args: Any, **kwargs: Any
        ) -> Any:
            # axis/keepdims/object routing inline, to keep the per-call overhead low
            if axis is not None or keepdims or x.dtype.kind == "O":
                if with_keepdims:
                    return impl(x, *args, axis=axis, keepdims=keepdims, **kwargs)
                return impl(x, *args, axis=axis, **kwargs)
            # the ndim test first, so that the flat case only costs one attribute lookup
            if layout_free and x.ndim > 1 and not x.flags.c_contiguous and x.flags.f_contiguous:
                x = x.T
            return kernel(x, nanval(x), *args, **kwargs)

        wrapper.__name__ = kernel.__name__
        wrapper.__qualname__ = kernel.__name__
        wrapper.__doc__ = impl.__doc__
        return wrapper

    return build if func is None else build(func)


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


@nb.njit(cache=True)
def _fix_invalid(x: npt.NDArray, nan: Any, copy: bool = True, fill_value: Any = 0) -> npt.NDArray:
    if copy:
        ret = np.empty_like(x)
    else:
        ret = x
    if x.ndim == 1:
        for i in range(len(x)):
            if isnan_vec(x[i], nan):
                ret[i] = fill_value
            else:
                ret[i] = x[i]
    else:
        for i in nb.prange(len(x.flat)):
            if isnan_vec(x.flat[i], nan):
                ret.flat[i] = fill_value
            else:
                ret.flat[i] = x.flat[i]
    return ret


def fix_invalid(x: npt.NDArray, copy: bool = True, fill_value: Any = 0) -> npt.NDArray:
    """Replace the missing values of x by fill_value, in place when copy is False."""
    view = _memory_order_view(x)  # elementwise, so the flat element order does not matter
    ret = _fix_invalid(view, nanval(x), copy=copy, fill_value=fill_value)
    if view is x:
        return ret
    return x if not copy else ret.T


@dispatch_ndim(layout_free=True)
@nb.njit(cache=True)
def allnan(x: npt.NDArray, nan: Any) -> bool:
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            return False
    return True


@dispatch_ndim(layout_free=True)
@nb.njit(cache=True)
def anynan(x: npt.NDArray, nan: Any) -> bool:
    for x_ in x.flat:
        if isnan_vec(x_, nan):
            return True
    return False


@dispatch_ndim(layout_free=True)
@nb.njit(cache=True)
def nanmax(x: npt.NDArray, nan: Any) -> Any:
    cmp_val = nan
    for x_ in x.flat:
        if isnan_vec(x_, nan):
            continue
        if isnan_vec(cmp_val, nan):
            cmp_val = x_
        elif x_ > cmp_val:
            cmp_val = x_
    return cmp_val


@dispatch_ndim(layout_free=True)
@nb.njit(cache=True)
def nanmin(x: npt.NDArray, nan: Any) -> Any:
    cmp_val = nan
    for x_ in x.flat:
        if isnan_vec(x_, nan):
            continue
        if isnan_vec(cmp_val, nan):
            cmp_val = x_
        elif x_ < cmp_val:
            cmp_val = x_
    return cmp_val


@dispatch_ndim
@nb.njit(cache=True)
def nanargmax(x: npt.NDArray, nan: Any) -> Any:
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


@dispatch_ndim
@nb.njit(cache=True)
def nanargmin(x: npt.NDArray, nan: Any) -> Any:
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


@dispatch_ndim
@nb.njit(cache=True)
def nansum(x: npt.NDArray, nan: Any) -> Any:
    ret = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            ret += x_
    return ret


@dispatch_ndim
@nb.njit(cache=True)
def nanprod(x: npt.NDArray, nan: Any) -> Any:
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


@dispatch_ndim
@nb.njit(cache=True)
def nanmean(x: npt.NDArray, nan: Any) -> Any:
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


@dispatch_ndim
@nb.njit(cache=True)
def nanvar(x: npt.NDArray, nan: Any, ddof: int = 0) -> Any:
    return _jnanvar(x, nan, ddof=ddof)


@dispatch_ndim(layout_free=True)
@nb.njit(cache=True)
def nanmedian(x: npt.NDArray, nan: Any) -> Any:
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


@dispatch_ndim
@nb.njit(cache=True)
def nanstd(x: npt.NDArray, nan: Any, ddof: int = 0) -> Any:
    return np.sqrt(_jnanvar(x, nan, ddof=ddof))


@dispatch_ndim(layout_free=True)
@nb.njit(cache=True)
def nancount(x: npt.NDArray, nan: Any) -> Any:
    cnt = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            cnt += 1
    return cnt


@dispatch_ndim(layout_free=True)
@nb.njit(cache=True)
def nanptp(x: npt.NDArray, nan: Any) -> Any:
    mx = nan
    mn = nan
    for x_ in x.flat:
        if isnan_vec(x_, nan):
            continue
        if isnan_vec(mx, nan):
            mx = x_
            mn = x_
            continue
        if x_ > mx:
            mx = x_
        if x_ < mn:
            mn = x_
    if isnan_vec(mx, nan):
        return nan
    return mx - mn


@dispatch_ndim(axis_default=0)
@nb.njit(cache=True)
def nanfirst(x: npt.NDArray, nan: Any) -> Any:
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            return x_
    return nan


@dispatch_ndim(axis_default=-1)
@nb.njit(cache=True)
def nanlast(x: npt.NDArray, nan: Any) -> Any:
    ret = nan
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            ret = x_
    return ret


@nb.njit(cache=True)
def _nanaverage(x: npt.NDArray, nan: Any) -> Any:
    total = 0.0
    cnt = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            total += x_
            cnt += 1
    return np.divide(total, cnt)


def nanaverage(
    x: npt.NDArray,
    axis: Axis = None,
    weights: npt.ArrayLike | None = None,
    returned: bool = False,
    keepdims: bool = False,
) -> Any:
    if x.dtype.kind not in _NUMERIC_KINDS or axis is not None or keepdims or weights is not None or returned:
        return _impl.nanaverage(x, axis=axis, weights=weights, returned=returned, keepdims=keepdims)
    return _nanaverage(x, nanval(x))


@nb.njit(cache=True)
def _nanpercentile(x: npt.NDArray, nan: Any, q: float, percent: bool) -> Any:
    cnt = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            cnt += 1
    if cnt == 0:
        return np.nan
    tmp = np.empty(cnt, dtype=np.float64)
    i = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            tmp[i] = x_
            i += 1
    tmp.sort()
    pos = q / 100 * (cnt - 1) if percent else q * (cnt - 1)
    if pos < 0.0:
        pos = 0.0
    last = cnt - 1
    if pos > last:
        pos = float(last)
    lo = int(pos)
    if lo >= last:
        return float(tmp[last])
    return tmp[lo] + (tmp[lo + 1] - tmp[lo]) * (pos - lo)


def nanpercentile(
    x: npt.NDArray,
    q: npt.ArrayLike,
    axis: Axis = None,
    out: npt.NDArray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Any:
    if (
        x.dtype.kind not in _NUMERIC_KINDS
        or axis is not None
        or keepdims
        or out is not None
        or overwrite_input
        or method != "linear"
        or not np.isscalar(q)
    ):
        return _impl.nanpercentile(
            x, q, axis=axis, out=out, overwrite_input=overwrite_input, method=method, keepdims=keepdims
        )
    # the values are collected and sorted, so the flat element order does not matter
    return _nanpercentile(_memory_order_view(x), nanval(x), q, True)


def nanquantile(
    x: npt.NDArray,
    q: npt.ArrayLike,
    axis: Axis = None,
    out: npt.NDArray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> Any:
    if (
        x.dtype.kind not in _NUMERIC_KINDS
        or axis is not None
        or keepdims
        or out is not None
        or overwrite_input
        or method != "linear"
        or not np.isscalar(q)
    ):
        return _impl.nanquantile(
            x, q, axis=axis, out=out, overwrite_input=overwrite_input, method=method, keepdims=keepdims
        )
    # the values are collected and sorted, so the flat element order does not matter
    return _nanpercentile(_memory_order_view(x), nanval(x), q, False)


@nb.njit(cache=True)
def _nanclip(x: npt.NDArray, nan: Any, lo: Any, hi: Any) -> npt.NDArray:
    ret = np.empty_like(x)
    for i in nb.prange(len(x.flat)):
        xi = x.flat[i]
        # Pick the value into one variable and keep a single store site: numba emits
        # roughly 14x slower code for a loop with three or more ret.flat[i] assignments,
        # while two of them (as in fix_invalid) are still as fast as direct indexing.
        val = xi
        if isnan_vec(xi, nan):
            val = nan
        elif xi < lo:
            val = lo
        elif xi > hi:
            val = hi
        ret.flat[i] = val
    return ret


def nanclip(x: npt.NDArray, a_min: npt.ArrayLike | None = None, a_max: npt.ArrayLike | None = None) -> npt.NDArray:
    if x.dtype.kind not in _NUMERIC_KINDS:
        return _impl.nanclip(x, a_min=a_min, a_max=a_max)
    lo = _impl._clip_bound(a_min, x, low=True)
    hi = _impl._clip_bound(a_max, x, low=False)
    if (isinstance(lo, np.ndarray) and lo.ndim > 0) or (isinstance(hi, np.ndarray) and hi.ndim > 0):
        return _impl.nanclip(x, a_min=lo, a_max=hi)
    lo = lo.item() if isinstance(lo, np.ndarray) else lo
    hi = hi.item() if isinstance(hi, np.ndarray) else hi
    if (isinstance(lo, float) or isinstance(hi, float)) and x.dtype.kind in "iu":
        return _impl.nanclip(x, a_min=a_min, a_max=a_max)  # float bounds promote, like np.clip
    if lo > hi:
        return _impl.nanclip(x, a_min=a_min, a_max=a_max)  # keep the np.clip edge-case semantics
    view = _memory_order_view(x)  # elementwise, so the flat element order does not matter
    ret = _nanclip(view, nanval(x), lo, hi)
    return ret if view is x else ret.T

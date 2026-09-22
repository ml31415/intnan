"""A bunch of small functions that replace and improve former usage of numexpr and bottleneck"""

from collections.abc import Callable
from functools import wraps
from typing import Any

import numba as nb
import numpy as np
import numpy.typing as npt

from .intnan_np import (
    INTNAN32,
    INTNAN64,
    NANVALS,
    __all__,
    asfloat,
    isnan,
    nanclose,
    nanequal,
    nanval,
)


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
        if isnan_vec(x[i], nan):
            ret.flat[i] = fill_value
        else:
            ret.flat[i] = x[i]
    return ret


@nancalc
def allnan(x: npt.NDArray, nan: Any) -> bool:
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            return False
    return True


@nancalc
def anynan(x: npt.NDArray, nan: Any) -> bool:
    for x_ in x.flat:
        if isnan_vec(x_, nan):
            return True
    return False


@nancalc
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


@nancalc
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
def nansum(x: npt.NDArray, nan: Any) -> Any:
    ret = 0
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            ret += x_
    return ret


@nancalc
def nanprod(x: npt.NDArray, nan: Any) -> Any:
    ret = 1
    for x_ in x.flat:
        if not isnan_vec(x_, nan):
            ret *= x_
    return ret


@nancalc
def nancumsum(x: npt.NDArray, nan: Any) -> npt.NDArray:
    ret = np.full_like(x, nan)
    val = nan
    for i, x_ in enumerate(x.flat):
        if not isnan_vec(x_, nan):
            if isnan_vec(val, nan):
                val = 0
            val += x_
        if not isnan_vec(val, nan):
            ret[i] = val
    return ret


@nancalc
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
nanvar = nancalc(_nanvar)


@nancalc
def nanstd(x: npt.NDArray, nan: Any, ddof: int = 0) -> Any:
    return np.sqrt(_jnanvar(x, nan, ddof=ddof))

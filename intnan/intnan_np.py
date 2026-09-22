"""
Module for handling integer arrays with missing values.
Missing values are handled as special values and functions
to skip them in processing are provided here.

The special nan values are chosen depending on the array type.
Large negative values are used, so that especially python indexing
(from the end) is unlikely to work.
"""

from typing import Any, TypeAlias, cast, overload

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
    "allnan",
    "anynan",
    "nanmin",
    "nanmax",
    "nanargmax",
    "nanargmin",
    "nanmaximum",
    "nanminimum",
    "nanmean",
    "nanmedian",
    "nanstd",
    "nanvar",
    "nansum",
    "nancumsum",
    "nancumprod",
    "nanprod",
    "nanequal",
    "nanclose",
]

NanValue: TypeAlias = float | int | str | bytes | None

INTNAN32 = np.iinfo("int32").min  # -2147483648
INTNAN64 = np.iinfo("int64").min  # -9223372036854775808
NANVALS: dict[str, NanValue] = dict(
    d=np.nan,
    f=np.nan,
    e=np.nan,
    S=b"",
    U="",
    l=INTNAN64,
    q=INTNAN32,
    i=INTNAN32,
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
    return NANVALS.get(dtype.char, default)


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


def anynan(x: npt.NDArray) -> bool:
    nv = nanval(x)
    if nv is np.nan:
        return bool(np.any(np.isnan(x)))
    elif nv is None:
        return any(val is None for val in x.flat)
    else:
        return bool(nv in x)


def allnan(x: npt.NDArray) -> bool:
    nv = nanval(x)
    if nv is np.nan:
        return bool(np.all(np.isnan(x)))
    elif nv is None:
        return all(val is None for val in x.flat)
    else:
        return bool(np.all(x == nv))


def nanmax(x: npt.NDArray) -> Any:
    nv = nanval(x)
    if nv is np.nan:
        return np.nanmax(x)
    else:
        try:
            return np.max(x[x != nv])
        except ValueError as e:
            if "zero-size" in str(e):
                return nv
            else:
                raise


def nanmin(x: npt.NDArray) -> Any:
    nv = nanval(x)
    if nv is np.nan:
        return np.nanmin(x)
    else:
        try:
            return np.min(x[x != nv])
        except ValueError as e:
            if "zero-size" in str(e):
                return nv
            else:
                raise


def nanargmax(x: npt.NDArray) -> Any:
    return np.nanargmax(asfloat(x))


def nanargmin(x: npt.NDArray) -> Any:
    return np.nanargmin(asfloat(x))


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


def nansum(x: npt.NDArray) -> Any:
    nv = nanval(x)
    if nv is np.nan:
        return np.nansum(x)
    else:
        return np.sum(x[x != nv])


def nanprod(x: npt.NDArray) -> Any:
    nv = nanval(x)
    if nv is np.nan:
        return np.nanprod(x, dtype=np.float64)
    else:
        return np.prod(x[x != nv])


def nancumsum(x: npt.NDArray) -> npt.NDArray:
    nv = nanval(x)
    result = np.cumsum(fix_invalid(x))

    if anynan(x):
        # cumsum is undefined before the first valid number appears, so we need to replace
        # the nans starting from the beginning of the array
        # TODO: Finding the first instance of a value this way is quite some overhead
        good_idx = np.where(~isnan(x))[0]
        if len(good_idx) > 0:
            result[: good_idx[0]] = nv
        else:
            result[:] = nv
    return result


def nancumprod(x: npt.NDArray) -> npt.NDArray:
    nv = nanval(x)
    result = np.cumprod(fix_invalid(x, fill_value=1))

    if anynan(x):
        # cumprod is undefined before the first valid number appears, so we need to replace
        # the nans starting from the beginning of the array
        good_idx = np.where(~isnan(x))[0]
        if len(good_idx) > 0:
            result[: good_idx[0]] = nv
        else:
            result[:] = nv
    return result


def nanmean(x: npt.NDArray) -> Any:
    nv = nanval(x)
    if nv is np.nan:
        return np.nanmean(x)
    else:
        with np.errstate(invalid="ignore"):
            return np.mean(x[x != nv])


def nanmedian(x: npt.NDArray) -> Any:
    nv = nanval(x)
    if nv is np.nan:
        return np.nanmedian(x)
    else:
        with np.errstate(invalid="ignore"):
            return np.median(x[x != nv])


def nanvar(x: npt.NDArray, ddof: int = 0) -> Any:
    nv = nanval(x)
    if nv is np.nan:
        return np.nanvar(x, ddof=ddof)
    else:
        with np.errstate(invalid="ignore"):
            return np.var(x[x != nv], ddof=ddof)


def nanstd(x: npt.NDArray, ddof: int = 0) -> Any:
    nv = nanval(x)
    if nv is np.nan:
        return np.nanstd(x, ddof=ddof)
    else:
        with np.errstate(invalid="ignore"):
            return np.std(x[x != nv], ddof=ddof)


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

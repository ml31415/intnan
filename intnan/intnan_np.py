"""
Module for handling integer arrays with missing values.
Missing values are handled as special values and functions
to skip them in processing are provided here.

The special nan values are chosen depending on the array type.
Large negative values are used, so that especially python indexing 
(from the end) is unlikely to work.
"""

import numpy as np

__all__ = ['NANVALS', 'INTNAN32', 'INTNAN64', 'nanval', 'isnan', 'fix_invalid', 'asfloat',
           'allnan', 'anynan', 'nanmin', 'nanmax', 'nanmaximum', 'nanminimum',
           'nanmean', 'nanstd', 'nanvar', 'nansum', 'nancumsum', 'nanprod',
           'nanequal', 'nanclose']

INTNAN32 = np.iinfo('int32').min  # -2147483648
INTNAN64 = np.iinfo('int64').min  # -9223372036854775808
NANVALS = dict(d=np.nan, f=np.nan, e=np.nan, S=b'', l=INTNAN64, q=INTNAN32, i=INTNAN32,
               b=-1, h=-1, B=0, H=0, L=0, Q=0, O=None)


def nanval(x):
    """ Return the corresponding NAN value for a column """
    return NANVALS.get(x.dtype.char)


def isnan(x):
    if isinstance(x, np.ndarray):
        nanval = NANVALS.get(x.dtype.char, 0)
        if nanval is np.nan:
            return np.isnan(x)
        elif nanval is None:
            return np.array([val is None for val in x])
        else:
            return x == nanval
    elif x in {np.nan, None, '', INTNAN32, INTNAN64}:
        return True
    else:
        try:
            return np.isnan(x)
        except TypeError:
            return False


def fix_invalid(x, copy=True, fill_value=0):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        if copy:
            return np.where(np.isnan(x), fill_value, x)
        else:
            x[np.isnan(x)] = fill_value
            return x
    elif nanval is None:
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
            return np.where(x == nanval, fill_value, x)
        else:
            x[x == nanval] = fill_value
            return x


def asfloat(x):
    if isinstance(x.dtype.type, np.floating):
        return x.copy()
    return fix_invalid(x, fill_value=np.nan)


def anynan(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.any(np.isnan(x))
    elif nanval is None:
        return any(val is None for val in x.flat)
    else:
        return nanval in x


def allnan(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.all(np.isnan(x))
    elif nanval is None:
        return all(val is None for val in x.flat)
    else:
        return np.all(x == nanval)


def nanmax(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.nanmax(x)
    else:
        try:
            return np.max(x[x != nanval])
        except ValueError as e:
            if 'zero-size' in str(e):
                return nanval
            else:
                raise


def nanmin(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.nanmin(x)
    else:
        try:
            return np.min(x[x != nanval])
        except ValueError as e:
            if 'zero-size' in str(e):
                return nanval
            else:
                raise


def nanmaximum(x, y):
    """ Does the same as numpy.maximum (element-wise maximum operation of two arrays) but ignores NaNs """
    z = np.maximum(x, y)
    badx = isnan(x)
    bady = isnan(y)
    z[badx] = y[badx]
    z[bady] = x[bady]
    return z


def nanminimum(x, y):
    """ Does the same as numpy.minimum (element-wise minimum operation of two arrays) but ignores NaNs """
    z = np.minimum(x, y)
    badx = isnan(x)
    bady = isnan(y)
    z[badx] = y[badx]
    z[bady] = x[bady]
    return z


def nansum(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.nansum(x)
    else:
        return np.sum(x[x != nanval])


def nanprod(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.nanprod(x, dtype=np.float64)
    else:
        return np.prod(x[x != nanval])


def nancumsum(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    result = np.cumsum(fix_invalid(x))

    if anynan(x):
        # cumsum is undefined before the first valid number appears, so we need to replace
        # the nans starting from the beginning of the array
        # TODO: Finding the first instance of a value this way is quite some overhead
        good_idx = np.where(~isnan(x))[0]
        if len(good_idx) > 0:
            result[:good_idx[0]] = nanval
        else:
            result[:] = nanval
    return result


def nanmean(x):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.nanmean(x)
    else:
        with np.errstate(invalid='ignore'):
            return np.mean(x[x != nanval])


def nanvar(x, ddof=0):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.nanvar(x, ddof=ddof)
    else:
        with np.errstate(invalid='ignore'):
            return np.var(x[x != nanval], ddof=ddof)


def nanstd(x, ddof=0):
    nanval = NANVALS.get(x.dtype.char, 0)
    if nanval is np.nan:
        return np.nanstd(x, ddof=ddof)
    else:
        with np.errstate(invalid='ignore'):
            return np.std(x[x != nanval], ddof=ddof)


def nanequal(x, y):
    """Treat NaN as an ordinary value when comparing for equality."""
    if x.dtype != y.dtype:
        raise TypeError("nanequal requires same data type: %s != %s" % (x.dtype, y.dtype))
    if issubclass(x.dtype.type, np.floating) and issubclass(y.dtype.type, np.floating):
        return np.isclose(x, y, 0, 0, equal_nan=True)
    else:
        return x == y


def nanclose(x, y, delta=np.finfo(float).eps):
    if x.dtype != y.dtype:
        raise TypeError("nanclose requires same data type: %s != %s" % (x.dtype, y.dtype))
    if issubclass(x.dtype.type, np.integer):
        return np.isclose(x, y, atol=delta, equal_nan=True)
    elif issubclass(x.dtype.type, str):
        return x == y
    else:
        return np.isclose(x, y, atol=delta, equal_nan=True)

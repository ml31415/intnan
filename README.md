[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/ml31415/intnan/python-package.yml?branch=master&logo=github&style=flat)](https://github.com/ml31415/intnan/actions)
[![Supported Versions](https://img.shields.io/pypi/pyversions/intnan.svg)](https://pypi.org/project/intnan)
[![PyPI](https://img.shields.io/pypi/v/intnan.svg?style=flat)](https://pypi.org/project/intnan/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# intnan

Integer data types lack special values for `-inf`, `inf` and `NaN`. Especially
`NaN` as an indication for missing data would be useful in many scientific contexts.

Of course there is `numpy.ma.MaskedArray` around for the very same reason. Nevertheless,
it might sometimes be annoying to carry a separate mask array around. And in those cases,
using a set of `numpy`-compatible functions for the same job will do just fine.

This package provides such an implementation for several standard `numpy` functions, that 
treat integer arrays in such a way, that the lowest negative integer resembles `NaN`.

The library provides an implementation using only standard `numpy` functions and
another implementation using `numba`, for functions that allow major speed gains. 
The `numba` implementation is automatically selected, when it is available for import.

## functions

The following list of functions is provided by `intnan`.

- nanval(x)
- isnan(x)
- fix_invalid(x, copy=True, fill_value=0)
- asfloat(x)
- anynan(x)
- allnan(x)
- nanmax(x)
- nanmin(x)
- nanmaximum(x, y)
- nanminimum(x, y)
- nansum(x)
- nanprod(x)
- nancumsum(x)
- nanmean(x)
- nanvar(x, ddof=0)
- nanstd(x, ddof=0)
- nanequal(x, y)
- nanclose(x, y, delta=sys.float_info.epsilon)

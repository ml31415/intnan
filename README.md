[![GitHub Workflow CI Status](https://img.shields.io/github/actions/workflow/status/ml31415/intnan/python-package.yml?branch=master&logo=github&style=flat)](https://github.com/ml31415/intnan/actions)
[![Supported Versions](https://img.shields.io/pypi/pyversions/intnan.svg)](https://pypi.org/project/intnan)
[![PyPI](https://img.shields.io/pypi/v/intnan.svg?style=flat)](https://pypi.org/project/intnan/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# intnan

Integer data types lack special values for `-inf`, `inf` and `NaN`. Especially
`NaN` as an indication for missing data would be useful in many scientific contexts.

Of course there is `numpy.ma.MaskedArray` around for the very same reason.
Nevertheless, it might sometimes be annoying to carry a separate mask array
around — and masked arrays are notoriously slow. In those cases, using a set of
`numpy`-compatible functions for the same job will do just fine.

This package provides such an implementation for a set of standard `numpy`
functions, treating integer arrays in such a way, that a designated sentinel
value resembles `NaN`:

- For signed integer types, the lowest negative value (`np.iinfo(dtype).min`)
  is used as the missing value, e.g. `-2147483648` for `int32`. Large negative
  values are chosen deliberately, so that python indexing (from the end) is
  unlikely to run into them accidentally.
- For unsigned integer types, the value `0` is used.
- For float and string types, the corresponding `NaN`, empty bytes or empty
  string values are used.
- For object arrays, `None` is used.

## Installation

```bash
pip install intnan
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install intnan
```

The package requires Python 3.12+ and works with numpy 2.4+. All public
functions are fully type annotated using `numpy.typing`.

## Usage

Simply import the package and use the provided functions like their `numpy`
counterparts:

```python
import numpy as np
import intnan

a = np.array([1, -(2**31), 3], dtype=np.int32)  # -(2**31) marks a missing value

intnan.isnan(a)  # array([False,  True, False])
intnan.nansum(a)  # 4
intnan.nanmean(a)  # 2.0
intnan.fix_invalid(a)  # array([1, 0, 3], dtype=int32)
```

## Functions

The following functions are provided by `intnan`. Where applicable, their
semantics mirror the corresponding `numpy` function, with missing values
ignored instead of propagated.

Missing value handling:

- `nanval(x)` — return the missing value for a given array or data type
- `isnan(x)` — boolean mask of missing values, works on arrays and scalars
- `fix_invalid(x, copy=True, fill_value=0)` — replace missing values
- `asfloat(x)` — convert to a float array, missing values become `NaN`
- `asint(x)` — convert to an integer array, missing values become the integer missing value
- `anynan(x)`, `allnan(x)` — test for the presence of missing values

Reductions:

- `nanmax(x)`, `nanmin(x)` and their index counterparts `nanargmax(x)`, `nanargmin(x)`
- `nansum(x)`, `nanprod(x)`, `nancumsum(x)`, `nancumprod(x)`
- `nanmean(x)`, `nanmedian(x)`, `nanvar(x, ddof=0)`, `nanstd(x, ddof=0)`

Element-wise binary operations:

- `nanmaximum(x, y)`, `nanminimum(x, y)` — as `np.maximum`/`np.minimum`, but
  picking the valid value wherever one input is missing

Comparison:

- `nanequal(x, y)` — element-wise equality, treating missing values as an
  ordinary value
- `nanclose(x, y, delta=sys.float_info.epsilon)` — element-wise closeness with
  tolerance `delta`

## Performance

The library ships two interchangeable implementations:

- `intnan_np` — based purely on vectorized `numpy` operations
- `intnan_numba` — JIT-compiled with [numba](https://numba.pydata.org/) for
  functions that allow major speed gains

Both provide the identical API. On import, the `numba` implementation is
automatically selected whenever numba is installed and importable; otherwise
the `numpy` implementation is used. This makes numba an optional runtime
dependency. Compiled numba kernels are cached on disk, so no recompilation
overhead occurs after the first use.

To get the accelerated implementation, simply install numba alongside:

```bash
pip install intnan numba
```

## Development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/ml31415/intnan
cd intnan
uv sync          # create virtualenv and install all dependencies
uv run pre-commit install  # optional: run lint, format and type checks on every commit
uv run pytest    # run the test suite
uv run ruff check . && uv run ruff format --check .  # lint and format check
uv run mypy      # type check
```

Tests are run against both implementations and a range of dtypes
(`int32`, `int64`, `float32`, `float64`) in the CI on Python 3.12 through 3.14.

## License

BSD 3-Clause, see [LICENSE.txt](LICENSE.txt).

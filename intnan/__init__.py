from .intnan_np import *

try:
    from .intnan_numba import *
except ImportError:
    pass


try:
    # Version is added only when packaged
    from ._version import __version__
except ImportError:
    try:
        from setuptools_scm import get_version
    except ImportError:
        __version__ = "0.0.0"
    else:
        __version__ = get_version(root="..", relative_to=__file__)
        del get_version

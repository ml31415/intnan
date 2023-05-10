from .intnan_np import *

try:
    from .intnan_numba import *
except ImportError:
    pass

from . import _version
__version__ = _version.get_versions()['version']

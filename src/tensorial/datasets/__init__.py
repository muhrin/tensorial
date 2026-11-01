"""Datasets built on top of :mod:`tensorial.gcnn.data`.

This package simply re-exports the public API so users can import as
``tensorial.datasets.<name>``.
"""

from . import qm9, rmd17
from .qm9 import *
from .rmd17 import *

__all__ = qm9.__all__ + rmd17.__all__

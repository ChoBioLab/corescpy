# __init__.py
# pylint: disable=unused-import

from . import utils as tl
from . import processing as pp
from . import analysis as ax
from . import visualization as pl
from .class_sc import Omics
from .class_crispr import corescpy
from .class_spatial import Spatial

import sys

sys.modules.update({f"{__name__}.{m}": globals()[m]
                    for m in ["ax", "pl", "pp", "tl",
                              "Omics", "Crispr", "Spatial"]})

__all__ = [
    "ax", "pl", "pp", "tl", "Omics", "Crispr", "Spatial"
]

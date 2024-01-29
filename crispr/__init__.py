# __init__.py
# pylint: disable=unused-import

from . import analysis as ax
from . import visualization as pl
from . import processing as pp
from . import utils as tl
from .class_sc import Omics
from .class_crispr import Crispr
from .class_spatial import Spatial

import sys

sys.modules.update({f"{__name__}.{m}": globals()[m]
                    for m in ["ax", "pl", "pp", "tl",
                              "Omics", "Crispr", "Spatial"]})

__all__ = [
    "ax", "pl", "pp", "tl", "Omics", "Crispr", "Spatial"
]

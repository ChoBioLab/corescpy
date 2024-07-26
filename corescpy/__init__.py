# __init__.py
# pylint: disable=unused-import

import sys
from .constants import get_panel_constants
from .class_sc import Omics
from .class_crispr import Crispr
from .class_spatial import Spatial
from . import utils as tl
from . import processing as pp
from . import analysis as ax
from . import visualization as pl
from . import class_crispr, class_sc, class_spatial, defaults

mod = ["ax", "pl", "pp", "tl", "Omics", "Crispr", "Spatial"]
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in mod})

__all__ = [
    "ax", "pl", "pp", "tl", "Omics", "Crispr", "Spatial",
    "processing", "analysis", "visualization", "utils",
    "class_sc", "class_crispr", "class_spatial", "defaults",
    "get_panel_constants"
]

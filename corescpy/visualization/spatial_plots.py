#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting and other image display and manipulation for spatial data.

@author: E. N. Aslinger
"""

from warnings import warn
from matplotlib import pyplot as plt
import tifffile
import numpy as np


def plot_tiff(file_tiff, levels=None, size=16, kind=None):
    """Plot .tiff file (`kind` argument only for title, e.g., DAPI)."""
    with tifffile.TiffFile(file_tiff) as t:
        lvls = np.arange(len(t.series[0].levels))  # available levels
    levels = lvls if levels is None else [levels] if isinstance(
        levels, str) else list(levels)  # levels -> list
    if any((i not in lvls for i in levels)):
        warn("Dropping levels not found in TIFF: "
             f"{set(levels).difference(set(lvls))}")
    for i in levels:
        with tifffile.TiffFile(file_tiff) as t:
            image = t.series[0].levels[i].asarray()
        plt.imshow(image, cmap="binary")
        plt.title(f"Level {i}" + str(f" {kind}" if kind else ""), size=size)
        plt.axis("Scaled")
        plt.show()

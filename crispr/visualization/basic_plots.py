#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member line-too-long
"""
Visualizing CRISPR experiment analysis results.

@author: E. N. Aslinger
"""

import pertpy as pt
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sb
# import cowplot
import warnings
import math
import pandas as pd
import numpy as np


def plot_by_cluster(adata, genes, method_cluster=None, plot_types="all"):
    if not isinstance(plot_types, str):
        raise TypeError("plot_types must be a string.")
    plot_types = plot_types.lower()  # so not case-sensitive
    if method is None:
        if "leiden" in adata.uns: 
            method = "leiden" 
        elif "louvain" in adata.uns: 
            method = "louvain"
        else:
            raise ValueError("No clustering method found in object.")
        warnings.warn("Clustering method unspecified. Using {method}.")
    figs = {}
    if plot_types == "all" or "violin" in plot_types:
        figs["violin"] = sc.pl.violin(adata, genes, groupby=method_cluster)
        figs["violin_stacked"] = sc.pl.stacked_violin(
            adata, genes, groupby=method_cluster, rotation=90)
    if plot_types == "all" or "dot" in plot_types:
        figs["dot"] = sc.pl.dotplot(adata, genes, groupby='leiden')
        

def square_grid(number):
    """Return row-column dimensions that most closely approximate a square."""
    if isinstance(number, (np.ndarray, list, set, tuple, pd.Series)):
        number = len(number)  # if provided actual object, calculate length
    rows = int(np.sqrt(number))  # number of rows (try to get close to square grid)
    cols = math.ceil(number / rows)  # number of columns
    return rows, cols
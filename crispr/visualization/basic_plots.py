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
    """Make plots facetted/split by cell type/cluster."""
    if not isinstance(plot_types, str):
        raise TypeError("plot_types must be a string.")
    plot_types = plot_types.lower()  # so not case-sensitive
    if method_cluster is None:
        if "leiden" in adata.uns: 
            method_cluster = "leiden" 
        elif "louvain" in adata.uns: 
            method_cluster = "louvain"
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
    return figs
        

def square_grid(number):
    """Return row-column dimensions (approximately a square)."""
    if isinstance(number, (np.ndarray, list, set, tuple, pd.Series)):
        number = len(number)  # if provided actual object, calculate length
    rows = int(np.sqrt(number))  # number of rows
    cols = math.ceil(number / rows)  # number of columns
    return rows, cols


# def plot_qc_vars(adata):
#     for k in qc_vars:
#         try:
#             figs[f"qc_pct_counts_{k}_hist"] = seaborn.histplot(
#                 adata.obs[pct_counts_vars[k]])
#         except Exception as err:
#             figs[f"qc_pct_counts_{k}_hist"] = err
#             print(err)
#         try:
#             figs["qc_metrics_violin"] = sc.pl.violin(
#                 adata[assay] if assay else adata, 
#                 ["n_genes_by_counts", "total_counts"] + pct_counts_vars,
#                 jitter=0.4, multi_panel=True)
#         except Exception as err:
#             figs["qc_metrics_violin"] = err
#             print(err)
#         for v in pct_counts_vars + ["n_genes_by_counts"]:
#             try:
#                 figs[f"qc_{v}_scatter"] = sc.pl.scatter(
#                     adata[assay] if assay else adata, x="total_counts", y=v)
#             except Exception as err:
#                 figs[f"qc_{v}_scatter"] = err
#                 print(err)
#     try:
#         figs["qc_log"] = seaborn.jointplot(
#             data=adata[assay].obs if assay else adata.obs,
#             x="log1p_total_counts", y="log1p_n_genes_by_counts", kind="hex")
#     except Exception as err:
#         figs["qc_log"] = err
#         print(err)
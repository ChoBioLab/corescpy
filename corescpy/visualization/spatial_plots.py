#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting and other image display and manipulation for spatial data.

@author: E. N. Aslinger
"""

from warnings import warn
from matplotlib import pyplot as plt
import tifffile
import traceback as tb
import scanpy as sc
import tangram as tg
import pandas as pd
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


def plot_integration_spatial(adata_sp, adata_sp_new, adata_sc=None,
                             col_cell_type=None, ad_map=None,
                             df_compare=None, plot_genes=None):
    figs = {}
    col_cell_type, col_cell_type_sp = [None, None] if (
        col_cell_type) is None else [col_cell_type, col_cell_type] if (
            isinstance(col_cell_type, str)) else col_cell_type
    if adata_sc and col_cell_type:
        tg.plot_cell_annotation_sc(adata_sp, list(pd.unique(adata_sc.obs[
            col_cell_type])), perc=0.02)  # annotations spatial plot
        if col_cell_type_sp not in adata_sp.obs:
            col_cell_type_sp = None  # if not present, ignore for plotting
        try:
            if col_cell_type_sp:
                fig, axs = plt.subplots(1, 2, figsize=(20, 5))
                sc.pl.spatial(adata_sp, color=col_cell_type_sp, alpha=0.7,
                              frameon=False, show=False, ax=axs[0])  # spatial
            sc.pl.umap(adata_sc, color=col_cell_type, size=10, frameon=False,
                       show=False if col_cell_type_sp else True,
                       ax=axs[1] if col_cell_type_sp else None)  # UMAP
            plt.tight_layout()
            figs["clusters"] = fig
        except Exception:
            print(tb.format_exc(), "\n\n", "Plotting UMAPs/spatial failed!")
    if ad_map:
        tg.plot_training_scores(ad_map, bins=20, alpha=0.5)  # train score
        figs["scores_training"] = plt.gcf()
    if df_compare:
        tg.plot_auc(df_compare)  # area under the curve
        figs["auc"] = plt.gcf()
    if plot_genes:
        tg.plot_genes_sc(plot_genes, adata_measured=adata_sp,
                         adata_predicted=adata_sp_new, perc=0.02)
        figs["genes"] = plt.gcf()
    return figs

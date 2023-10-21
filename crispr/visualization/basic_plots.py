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

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


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
    cols = rows if rows * 2 == number else math.ceil(
        number / rows)  # number of columns
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


def plot_umap(adata, col_cell_type="leiden", title="UMAP", color=None, 
              legend_loc="on_data", genes=None, col_gene_symbols=None, 
              cell_types_circle=None, **kwargs):
    """Make UMAP-based plots."""
    figs = {}
    if "cmap" in kwargs:  # in case use wrong form of argument
        kwargs["color_map"] = kwargs.pop("cmap")
    kwargs = {"color_map": COLOR_MAP, "palette": COLOR_PALETTE, 
              "frameon": False, "legend_loc": "on_data", 
              "vcenter": 0, **kwargs}
    if "X_umap" in adata.obsm or col_cell_type in adata.obs.columns:
        print("\n<<< PLOTTING UMAP >>>")
        try:
            figs["clustering"] = sc.pl.umap(
                adata, color=col_cell_type, return_fig=True, 
                title=title,  **kwargs)  # UMAP ~ cell type
        except Exception as err:
            warnings.warn(f"{err}\n\nCould not plot UMAP clusters.")
            figs["clustering"] = err
        if genes is not None:
            if not isinstance(genes, (list, np.ndarray)) and (
                genes is None or genes == "all"):
                genes = list(pd.unique(adata.var_names))  # gene names
            else:  # if unspecified, random subset of genes
                if isinstance(genes, (int, float)):
                    genes = list(pd.Series(adata.var_names).sample(genes))
            print("\n<<< PLOTTING GEX ON UMAP >>>")
            try:
                figs["clustering_gene_expression"] = sc.pl.umap(
                    adata, title=genes, return_fig=True, 
                    gene_symbols=col_gene_symbols, color=genes,
                    legend_loc=legend_loc, **kwargs)  # UMAP ~ GEX
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot GEX UMAP.")
                figs["clustering_gene_expression"] = err
        if color is not None:
            print(f"\n<<< PLOTTING {color} on UMAP >>>")
            try:
                figs[f"clustering_{color}"] = sc.pl.umap(
                    adata, title=title if title else None, 
                    return_fig=True, color=color, **kwargs)  # UMAP ~ GEX
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot UMAP ~ {color}.")
                figs[f"clustering_{color}"] = err
        if cell_types_circle and "X_umap" in adata.obsm:
            figs["circled"], axu = plt.subplots(figsize=(3, 3))
            sc.pl.umap(adata, color=[col_cell_type], ax=axu, show=False)
            for h in cell_types_circle:  # circle cell type
                locs = adata[adata.obs[col_cell_type] == h, :].obsm['X_umap']
                coordinates = [locs[:, i].mean() for i in [0, 1]]
                circle = plt.Circle(tuple(coordinates), 1.5, color="r", 
                                    clip_on=False, fill=False)  # circle
                axu.add_patch(circle)
            # l_1 = axu.get_legend()  # save original Legend
            # l_1.set_title(lab_cluster)
            # # Make a new Legend for the mark
            # l_2 = axu.legend(handles=[Line2D(
            #     [0],[0],marker="o", color="k", markerfacecolor="none", 
            #     markersize=12, markeredgecolor="r", lw=0, 
            #     label="selected")], frameon=False, 
            #                 bbox_to_anchor=(3,1), title='Annotation')
            #     # Add back the original Legend (was overwritten by new)
            # _ = plt.gca().add_artist(l_1)
        return figs
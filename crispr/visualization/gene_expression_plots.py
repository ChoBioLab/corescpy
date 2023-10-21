#!/usr/bin/env python3_layers
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
import crispr as cr
import pandas as pd
import numpy as np

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


def plot_gex(adata, col_cell_type=None, title=None, 
             col_gene_symbols=None, 
             genes=None, marker_genes_dict=None,
             genes_highlight=None, kind="all",
             **kws_plots):
    """
    Make gene expression violin, heatmap.
    
    Pass additional keyword arguments to `sc.pl.violin` and 
    `sc.pl.heatmap` by specifying the in dictionaries in 
    the arguments "kws_heat" and/or "kws_violin" 
    (i.e., variable arguments **kws_plots).
    
    Specify "dot," "heat," or "violin," a list of these,
        or "all" to choose which types to plot.
    """
    figs = {}
    if isinstance(kind, str):
        kind = ["dot", "heat", "violin"] if kind == "all" else [kind.lower()]
    kind = [x.lower() for x in kind]
    names_layers = cr.pp.get_layer_dict()
    kws_hm, kws_violin, kws_matrix, kws_dot = [kws_plots[f"kws_{x}"] if (
        x in kws_plots) else {} for x in ["heat", "violin", "matrix", "dot"]]
    if not isinstance(genes, (list, np.ndarray)) and (
        genes is None or genes == "all"):
        genes = list(pd.unique(adata.var_names))  # gene names
    else:  # if unspecified, random subset of genes
        if isinstance(genes, (int, float)):
            genes = list(pd.Series(adata.var_names).sample(genes))
    
    # Heatmap(s)
    if "heat" in kind or "heatmap" in kind or "hm" in kind:
        print("\n<<< PLOTTING GEX (Heatmap) >>>")
        kws_hm = {**{"dendrogram": True, "show_gene_labels": True}, **kws_hm}
        if "cmap" not in kws_hm:
            kws_hm.update({"cmap": COLOR_MAP})
        for i in list([None] + list(adata.layers)):
            lab = f"gene_expression_{i}" if i else "gene_expression"
            hm_title = f"Gene Expression: {i}" if i else "Gene Expression"
            if title: 
                hm_title += f" ({title})"
            try:
                figs[lab] = sc.pl.heatmap(
                    adata, genes, layer=i, show=False, layer=i, 
                    gene_symbols=col_gene_symbols, **kws_hm)  # heatmap
                # axes_gex[j].set_title(i.capitalize() if i else None)
                figs[lab] = plt.gcf(), figs[lab]
                figs[lab][0].suptitle(hm_title)
                # figs[lab][0].supxlabel("Gene")
                figs[lab][0].show()
            except Exception as err:
                warnings.warn(
                    f"{err}\n\nCould not plot GEX heatmap ('{hm_title}').")
                figs[lab] = err
    
    # Violin Plots
    if "violin" in kind:
        print("\n<<< PLOTTING GEX (Violin) >>>")
        if "color_map" in kws_violin:
            kws_violin["cmap"] = kws_violin.pop("color_map")
        kws_violin.update({"cmap": COLOR_MAP, **kws_violin})
        if "groupby" in kws_violin or "col_cell_type" in kws_violin:
            lab_cluster = kws_violin.pop(
                "groupby" if "groupby" in kws_violin else "col_cell_type")
        else:
            lab_cluster = col_cell_type
        if lab_cluster not in adata.obs:
            lab_cluster = None   # None if cluster label N/A in `.obs`
        for i in zip(["dendrogram", "swap_axes", "cmap"], 
                    [True, False, COLOR_MAP]):
            if i[0] not in kws_violin:  # add default arguments
                kws_violin.update({i[0]: i[1]})
        for i in [None] + list(adata.layers.keys()):
            try:
                lab = f"gene_expression_violin_{i}"
                title_gexv = title if title else "Gene Expression"
                if i:
                    lab += "_" + str(i)
                    if not title:
                        title_gexv = f"{title_gexv} ({i})"
                figs[lab] = sc.pl.stacked_violin(
                    adata, marker_genes_dict if marker_genes_dict else genes,
                    groupby=lab_cluster if lab_cluster in adata.obs else None, 
                    layer=i, return_fig=True, gene_symbols=col_gene_symbols, 
                    title=title_gexv, show=False, **kws_violin)  # violin plot
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster)
                figs[lab].show()
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot GEX violins.")
                figs[lab] = err
            
    # Dot Plot
    if "dot" in kind:
        print("\n<<< PLOTTING GEX (Dot) >>>")
        try:
            figs["gene_expression_dot"] = sc.pl.dotplot(
                adata, genes, lab_cluster, show=False, use_raw=False)
            try:
                if genes_highlight is not None:
                    for x in figs["gene_expression_dot"][
                        "mainplot_ax"].get_xticklabels():
                        # x.set_style("italic")
                        if x.get_text() in genes_highlight:
                            x.set_color('#A97F03')
            except Exception as error:
                print(error, "Could not highlight gene name.")
        except Exception as err:
            warnings.warn(f"{err}\n\nCould not plot GEX violins.")
            figs["gene_expression_dot"] = err

    # Matrix Plots
    if "matrix" in kind:
        print("\n<<< PLOTTING GEX (Matrix) >>>")
        if "color_map" in kws_matrix:
            kws_matrix["cmap"] = kws_matrix.pop("color_map")
        if "cmap" not in kws_matrix:
            kws_matrix.update({"cmap": COLOR_MAP})
        if "groupby" in kws_matrix or "col_cell_type" in kws_matrix:
            lab_cluster = kws_matrix.pop(
                "groupby" if "groupby" in kws_matrix else "col_cell_type")
        for i in zip(["dendrogram", "swap_axes", "cmap"], 
                        [True, False, COLOR_MAP]):
            if i[0] not in kws_matrix:  # add default arguments
                kws_matrix.update({i[0]: i[1]})
        for i in [None] + list(adata.layers):
            lab = f"gene_expression_matrix"
            title_gexm = title if title else "Gene Expression"
            if i:
                lab += "_" + str(i)
                if not title:
                    title_gexm = f"{title_gexm} ({i})"
            bar_title = "Expression"
            if i == names_layers["scaled"]:
                bar_title += " (Mean Z-Score)"
            try:
                figs[lab] = sc.pl.matrixplot(
                    adata, genes, layer=i, return_fig=True, 
                    groupby=lab_cluster if lab_cluster in adata.obs else None,
                    title=title_gexm, gene_symbols=col_gene_symbols, 
                    **{"colorbar_title": bar_title}, **kws_matrix, show=False)
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster_mat)
                figs[lab].show()
            except Exception as err:
                print(f"{err} in plotting GEX matrix for label {i}")
                figs[lab] = err
    return figs
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
from matplotlib import colors
import seaborn as sb
# import cowplot
import warnings
import copy
import crispr as cr
import pandas as pd
import numpy as np

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


def plot_gex(adata, col_cell_type=None, title=None, 
             col_gene_symbols=None, layer=None,
             genes=None, marker_genes_dict=None,
             genes_highlight=None, kind="all",
             kws_heat=None, kws_violin=None, kws_matrix=None, kws_dot=None):
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
    kws_heat, kws_violin, kws_matrix, kws_dot = [
        x if x else {} for x in [kws_heat, kws_violin, kws_matrix, kws_dot]]
    kws_heat = {**{"dendrogram": True, "show_gene_labels": True}, **kws_heat}
    if not isinstance(genes, (list, np.ndarray)) and (
        genes is None or genes == "all"):
        genes = list(pd.unique(adata.var_names))  # gene names
    else:  # if unspecified, random subset of genes
        if isinstance(genes, (int, float)):
            genes = list(pd.Series(adata.var_names).sample(genes))
    if adata.var.index.names[0] == col_gene_symbols:
        col_gene_symbols = None
    
    # Heatmap(s)
    if "heat" in kind or "heatmap" in kind or "hm" in kind:
        print("\n<<< PLOTTING GEX (Heatmap) >>>")
        if "cmap" not in kws_heat:
            kws_heat.update({"cmap": COLOR_MAP})
        if layer is not None:
            layers = [layer] if isinstance(layer, str) else layer
        else:
            layers = [kws_heat["layer"]] if "layer" in kws_heat else list(
                [None] + list(adata.layers))  # layer(s) to plot
        for i in layers:
            lab = f"heat{str('_' + str(i) if i else '')}"
            title_h = kws_heat["title"] if "title" in kws_heat else \
                title if title else "Gene Expression"
            title_h = f"{title_h} ({i})" if i else title_h
            try:
                sc.tl.dendrogram(adata, col_cell_type)
                figs[lab] = sc.pl.heatmap(
                    adata, marker_genes_dict if marker_genes_dict else genes, 
                    col_cell_type, show=False, gene_symbols=col_gene_symbols, 
                    **{**kws_heat, "layer": i})  # heatmap
                # axes_gex[j].set_title(i.capitalize() if i else None)
                figs[lab] = plt.gcf(), figs[lab]
                figs[lab][0].suptitle(title_h)
                # figs[lab][0].supxlabel("Gene")
                figs[lab][0].show()
            except Exception as err:
                warnings.warn(
                    f"{err}\n\nCould not plot GEX heatmap ('{title_h}').")
                figs[lab] = err
    
    # Violin Plots
    if "violin" in kind:
        print("\n<<< PLOTTING GEX (Violin) >>>")
        if "color_map" in kws_violin:
            kws_violin["cmap"] = kws_violin.pop("color_map")
        kws_violin.update({"cmap": COLOR_MAP, **kws_violin})
        kws_violin_o = copy.deepcopy(kws_violin)
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
        if layer is not None:
            layers = [layer] if isinstance(layer, str) else layer
        else:
            layers = [kws_violin["layer"]] if "layer" in kws_violin else list(
                [None] + list(adata.layers))  # layer(s) to plot
        for i in layers:
            lab = f"violin{str('_' + str(i) if i else '')}"
            title_v = kws_violin["title"] if "title" in kws_violin else \
                title if title else "Gene Expression"
            title_v = f"{title_v} ({i})" if i else title_v
            # Stacked Violin
            try:
                figs[lab + "_stacked"] = sc.pl.stacked_violin(
                    adata, marker_genes_dict if marker_genes_dict else genes,
                    groupby=lab_cluster if lab_cluster in adata.obs else None, 
                    return_fig=True, gene_symbols=col_gene_symbols, 
                    show=False, **{**kws_violin, "layer": i, "title": title_v}
                    )  # violin (stacked)
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster)
                figs[lab + "_stacked"].show()
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot GEX stacked violins.")
                figs[lab + "_stacked"] = err
            # Normal Violin (Genes=Panels)
            try:
                figs[lab] = sc.pl.violin(
                    adata, marker_genes_dict if marker_genes_dict else genes,
                    groupby=lab_cluster if lab_cluster in adata.obs else None, 
                    show=False, **{"rotation": 90, 
                                   **kws_violin_o, "layer": i})  # violin
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster)
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot GEX violins.")
                figs[lab] = err
            
    # Dot Plot
    if "dot" in kind:
        print("\n<<< PLOTTING GEX (Dot) >>>")
        if "groupby" in kws_dot or "col_cell_type" in kws_dot:
            lab_cluster = kws_dot.pop(
                "groupby" if "groupby" in kws_dot else "col_cell_type")
        else:
            lab_cluster = col_cell_type
        if lab_cluster not in adata.obs:
            lab_cluster = None   # None if cluster label N/A in `.obs`
        try:
            title_d = kws_dot["title"] if "title" in kws_dot else \
                title if title else "Gene Expression"
            figs["dot"] = sc.pl.DotPlot(
                adata, marker_genes_dict if marker_genes_dict else genes, 
                lab_cluster, gene_symbols=col_gene_symbols, 
                **{**kws_dot, "title": title_d})  # dot plot
            try:
                if genes_highlight is not None:
                    for x in figs["dot"][
                        "mainplot_ax"].get_xticklabels():
                        # x.set_style("italic")
                        if x.get_text() in genes_highlight:
                            x.set_color('#A97F03')
            except Exception as error:
                print(error, "Could not highlight gene name.")
            figs["dot"].show()
        except Exception as err:
            warnings.warn(f"{err}\n\nCould not plot GEX violins.")
            figs["dot"] = err

    # Matrix Plots
    if "matrix" in kind:
        print("\n<<< PLOTTING GEX (Matrix) >>>")
        if "groupby" in kws_matrix or "col_cell_type" in kws_matrix:
            lab_cluster = kws_matrix.pop(
                "groupby" if "groupby" in kws_matrix else "col_cell_type")
        else:
            lab_cluster = col_cell_type
        if lab_cluster not in adata.obs:
            lab_cluster = None   # None if cluster label N/A in `.obs`
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
        if layer is not None:
            layers = [layer] if isinstance(layer, str) else layer
        else:
            layers = [kws_matrix["layer"]] if "layer" in kws_matrix else list(
                [None] + list(adata.layers))  # layer(s) to plot
        for i in layers:
            lab = f"matrix{str('_' + str(i) if i else '')}"
            title_m = kws_matrix["title"] if "title" in kws_matrix else \
                title if title else "Gene Expression"
            title_m = f"{title_m} ({i})" if i else title_m
            bar_title = "Expression"
            if i == names_layers["scaled"]:
                bar_title += " (Mean Z-Score)"
            try:
                figs[lab] = sc.pl.matrixplot(
                    adata, genes, return_fig=True, 
                    groupby=lab_cluster if lab_cluster in adata.obs else None,
                    gene_symbols=col_gene_symbols, colorbar_title=bar_title, 
                    **{**kws_matrix, "layer": i, "title": title_m}, 
                    show=False)  # matrix plot
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster_mat)
                figs[lab].show()
            except Exception as err:
                warnings.warn(f"{err} in plotting GEX matrix for label {i}")
                figs[lab] = err
    return figs


def plot_umap_multi(adata, genes, title=None, **kwargs):
    """Plot multiple continuous features (e.g, genes) on same UMAP."""
    fxs = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, 
           plt.cm.Purples, plt.cm.Oranges, plt.cm.Greys]
    if len(genes) > len(fxs):
        warnings.warn("More genes than colors. Splitting plot.")
        ggg = np.array_split(genes, np.arange(len(fxs), len(genes), len(fxs)))
        for g in ggg:
            axis_list = plot_umap_multi(
                adata, g, title=None, **kwargs)
        return axis_list
    cmaps = []
    for i in np.arange(len(genes)):
        colors2 = fxs[i](np.linspace(0, 1, 128))
        colors_comb = np.vstack([colors2])
        mymap = colors.LinearSegmentedColormap.from_list(
            'my_colormap', colors_comb)
        my_cmap = mymap(np.arange(mymap.N))
        my_cmap[:,-1] = np.linspace(0, 1, mymap.N)
        my_cmap = colors.ListedColormap(my_cmap)
        cmaps += [my_cmap]
    # fig = plt.figure.Figure()
    if "ax" in kwargs:
        axis = kwargs["ax"]
        _ = kwargs.pop("ax")
    else:
        axis = None
    for i in np.arange(len(genes)):
        axis = sc.pl.umap(adata, color=genes[i], 
                          ax=axis,  # use previous axis
                          title=title, show=False, return_fig=None, 
                          colorbar_loc=None, color_map=cmaps[i], **kwargs)
        c_b = plt.colorbar(
            axis.collections[i], ax=axis, pad=0.05, aspect=30, 
             orientation="horizontal", ticklocation="top",
            )
        c_b.ax.spines[["left", "right", "top"]].set_visible(False)
        c_b.minorticks_off()
        c_b.set_ticklabels(c_b.get_ticks(), rotation=270, fontdict={
            "fontsize": 3})  # tick labels font size
        c_b.set_ticklabels([c_b.get_ticks()[0]] + [""] * (
            len(c_b.get_ticks()) - 2) + ["{1:0.{0}f}\"".format(int(
                c_b.get_ticks()[-1] % 1 > 0), c_b.get_ticks()[-1])], 
            rotation=270, fontdict={"fontsize": 3})  # tick labels font size
        c_b.set_label(genes[i], rotation=0, loc="center", fontdict={
            "fontsize": 8}, labelpad=4, rotation_mode="anchor"
                      )  # colorbar title
        c_b.ax.xaxis.set_ticks_position("top")
    return axis


def plot_umap_split(adata, split_by, color="leiden", 
                    ncol=2, nrow=None, figsize=None, **kwargs):
    """Plot UMAP ~ group (from Scanpy GitHub issues/2333)."""
    categories = adata.obs[split_by].cat.categories
    if nrow is None:
        nrow = int(np.ceil(len(categories) / ncol))
    if figsize is None:
        figsize = (5 * ncol, 4 * nrow)
    fig, axs = plt.subplots(nrow, ncol, figsize=figsize)
    axs = axs.flatten()
    for i, cat in enumerate(categories):
        if not isinstance(color, str) and len(color) > 0:
            plot_umap_multi(adata[adata.obs[split_by] == cat], color,
                            ax=axs[i], **{"title": cat, **kwargs})
        else:
            sc.pl.umap(adata[adata.obs[split_by] == cat], color=color, 
                       ax=axs[i], show=False, **{"title": cat, **kwargs})
    plt.tight_layout()
    return fig, axs
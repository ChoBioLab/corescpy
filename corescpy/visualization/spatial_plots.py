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
import squidpy as sq
import spatialdata
import tangram as tg
import pandas as pd
import numpy as np
import corescpy as cr


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


def plot_spatial(adata, color="leiden", col_segment=None, figsize=20,
                 spatial_key=cr.pp.SPATIAL_KEY, key_image=None,
                 library_id=None, col_sample_id=None,
                 wspace=0.1, shape=None, cmap=None,
                 title=None, title_offset=0, **kwargs):
    """Plot spatial by clusters, transcripts, batches, etc."""
    ann = adata.copy()
    if isinstance(figsize, (int, float)):
        figsize = (figsize, figsize)
    libid = library_id if library_id else list(ann.uns[
        spatial_key].keys()) if spatial_key in ann.uns else None
    if col_sample_id:
        libid = list(set(libid).intersection(set(adata.obs[
            col_sample_id].unique())))
    if isinstance(ann, spatialdata.SpatialData) and shape:
        warn("Can't currently use `shape` parameter with SpatialData.")
        shape = None
    cgs = kwargs.pop("col_gene_symbols", None)
    kws = cr.tl.merge(dict(figsize=figsize, shape=shape, cmap=cmap,
                           return_ax=True, library_key=col_sample_id,
                           library_id=libid, color=color, alt_var=cgs,
                           wspace=wspace), kwargs)  # keyword arguments
    img_names = list(ann.uns[spatial_key][libid if isinstance(
        libid, str) else libid[0]]["images"].keys())
    kws["img_res_key"] = key_image if key_image else "hires" if (
        "hires" in img_names) else img_names[0]

    # Plot
    try:
        with plt.rc_context({"figure.constrained_layout.use": True}):
            try:
                fig = sq.pl.spatial_segment(ann, col_segment, **kws) if (
                    col_segment) else sq.pl.spatial_scatter(ann, **kws)
            except Exception:  # remove Leiden colors if => Squidpy bug
                for c in color:
                    _ = ann.uns.pop(f"{c}_colors", None)
                fig = sq.pl.spatial_segment(ann, col_segment, **kws) if (
                    col_segment) else sq.pl.spatial_scatter(ann, **kws)
    except Exception:
        fig = str(tb.format_exc())
        print(fig)

    # Modify (e.g, Title)
    try:
        fig.figure.suptitle(title, y=1 - title_offset)
    except Exception:
        pass
    return fig


def plot_integration_spatial(adata_sp, adata_sp_new=None, adata_sc=None,
                             col_cell_type=None, ad_map=None, cmap="magma",
                             df_compare=None, plot_genes=None, perc=0.01,
                             col_annotation=None, figsize=5,
                             plot_density=False, **kwargs):
    """Plot integration (see `corescpy.pp.integrate_spatial`)."""
    figs = {}
    col_cell_type, col_cell_type_sp = [None, None] if (
        col_cell_type) is None else [col_cell_type, col_cell_type] if (
            isinstance(col_cell_type, str)) else col_cell_type
    if isinstance(figsize, (int, float)):
        figsize = (figsize, figsize)
    if adata_sc and col_cell_type:
        # tg.plot_cell_annotation_sc(adata_sp, list(pd.unique(adata_sc.obs[
        #     col_cell_type])), perc=0.02)  # annotations spatial plot
        if col_cell_type_sp not in adata_sp.obs:
            col_cell_type_sp = None  # if not present, ignore for plotting
        try:
            if col_cell_type_sp:
                fig, axs = plt.subplots(1, 2, figsize=figsize)
                plot_spatial(adata_sp, color=col_cell_type_sp, ax=axs[0])
            sc.pl.umap(adata_sc, color=col_cell_type, size=10, frameon=False,
                       show=False if col_cell_type_sp else True,
                       ax=axs[1] if col_cell_type_sp else None,
                       legend_loc="on data")  # UMAP
            plt.tight_layout()
            figs["clusters"] = plt.gcf()
        except Exception:
            figs["clusters"] = tb.format_exc()
        if plot_density is True:
            print(tb.format_exc(), "\n\n", "UMAP/spatial plots failed!")
            try:  # plot integration-inferred spatial cell types
                tmp, dfp, _ = cr.pp.construct_obs_spatial_integration(
                    adata_sp.copy(), adata_sc.copy(), col_cell_type,
                    perc=perc)  # spatial AnnData + cell type density
                figs["spatial_density"] = plot_spatial(
                    tmp, color=list(pd.unique(adata_sc.obs[
                        col_cell_type])), cmap=cmap)  # cell types plot
            except Exception:
                figs["spatial_density"] = tb.format_exc()
                print(tb.format_exc(), "\n\n", "UMAP/spatial plots failed!")
    if ad_map is not None:  # plot training scores
        tg.plot_training_scores(ad_map, bins=20, alpha=0.5)  # train score
        figs["scores_training"] = plt.gcf()
    if df_compare is not None:  # plot AUC for predictions
        figs["auc"] = plot_auc_spatial_integration(
            df_compare, title="Prediction on Test Transcriptome")  # AUC
        plt.gcf()
    if plot_genes is not None and adata_sp_new is not None:  # plot GEX
        tg.plot_genes_sc(plot_genes, adata_measured=adata_sp,
                         adata_predicted=adata_sp_new, perc=0.02)
        figs["genes"] = plt.gcf()
    if col_annotation is not None:  # predicted cell types spatial plot
        figs["spatial"] = plot_spatial(
            adata_sp_new, color=col_annotation, **kwargs)  # plot
    return figs


def plot_auc_spatial_integration(df_all_genes, test_genes=None, title=""):
    """
        Plots auc curve which is used to evaluate model performance.
        Fixed version of Tangram's `plot_auc` function.

    Args:
        df_all_genes (Pandas DataFrame): Returned by
            `tangram.compare_spatial_geneexp(adata_ge, adata_sp)`.
        test_genes (list): List of test genes, if not given,
            test_genes will be set to genes where 'is_training'
            field is False.

    Returns:
        None
    """
    metric_dict, ((pol_xs, pol_ys), (x_s, y_s)) = tg.utils.eval_metric(
        df_all_genes, test_genes)
    fig = plt.figure()
    plt.figure(figsize=(6, 5))
    plt.plot(pol_xs, pol_ys, c="r")
    plt.scatter(x_s, y_s, alpha=0.5, edgecolors="face")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(.5)
    plt.xlabel("Score")
    plt.ylabel("Spatial Sparsity")
    plt.tick_params(axis="both", labelsize=8)
    plt.title(title)
    textstr = "auc_score={}".format(np.round(metric_dict["auc_score"], 3))
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    # place a text box in upper left in axes coords
    plt.text(0.03, 0.1, textstr, fontsize=12,
             verticalalignment="top", bbox=props)
    return fig

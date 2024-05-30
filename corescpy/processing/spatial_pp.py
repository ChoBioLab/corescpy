#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Processing spatial data.

Functions adapted from
'https://www.10xgenomics.com/analysis-guides/performing-3d-nucleus-
segmentation-with-cellpose-and-generating-a-feature-cell-matrix'
and `stlearn` tutorials.

@author: E. N. Aslinger
"""

import tifffile
import csv
import os
import re
import traceback
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
# import stlearn as st
import spatialdata_io as sdio
import scipy.sparse as sparse
import scipy.io as sio
import subprocess
import tangram as tg
import pandas as pd
import numpy as np
# from corescpy.visualization import plot_integration_spatial
# from corescpy.utils import merge
import corescpy as cr

# Define constant.
# z-slices are 3 microns apart
Z_SLICE_MICRON = 3
SPATIAL_KEY = "spatial"
SPATIAL_IMAGE_KEY_SEP = "___"
STORE_UNS_SQUIDPY = True  # for back-compatibility with Squidpy
# store images from SpatialData object in SpatialData.table.uns (AnnData.uns)


def _get_control_probe_names():
    """Get Xenium control probe names."""
    return ["NegControlCodeword", "NegControlProbe", "Unassigned"]


def read_spatial(file_path, file_path_spatial=None, file_path_image=None,
                 visium=False, spatial_key="spatial", library_id=None,
                 col_gene_symbols="gene_symbols", prefix=None, gex_only=False,
                 col_sample_id="library_key_spatial", n_jobs=1, **kwargs):
    """Read Xenium or Visium spatial data into an AnnData object."""
    # missing_fps = file_path_spatial is None and visium is False
    # uns_spatial = kwargs.pop("uns_spatial", None)
    # _ = kwargs.pop("spatial", None)
    # if col_sample_id is None:
    #     col_sample_id = "library_key_spatial"
    # if missing_fps:
    #     f_s = os.path.join(os.path.dirname(file_path), "cells.csv")
    #     file_path_spatial = f_s if os.path.exists(
    #         f_s) else f_s + ".gz" if os.path.exists(f_s + ".gz") else None
    # if file_path_image is not None:
    #     os.path.abspath(file_path_image)  # absolute for reproducibility
    # if isinstance(visium, dict) or visium is True:
    #     if not isinstance(visium, dict):
    #         visium = {}  # unpack file path & arguments
    #     adata = sq.read.visium(file_path, **visium)  # read Visium
    # else:
    #     if isinstance(file_path, (str, os.PathLike)):
    #         file_path = os.path.abspath(
    #             file_path)  # absolute path for reproducibility
    #         adata = sc.read_10x_mtx(
    #             file_path, var_names=col_gene_symbols, cache=True,
    #             gex_only=gex_only, prefix=prefix)  # read 10x
    #     else:
    #         adata = file_path.copy()
    #     print(f"\n*** Retrieving spatial data from {file_path_spatial}\n")
    #     comp = "gzip" if ".gz" in file_path_spatial[-3:] else None
    #     dff = pd.read_csv(file_path_spatial, compression=comp, index_col=0)
    #     adata.obs = adata.obs.join(dff, how="left")
    #     adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]
    #                                       ].copy().to_numpy()  # coordinates
    # if spatial_key not in adata.uns:
    #     if uns_spatial is not None and "scalefactors" in uns_spatial:
    #         uns_spatial["scalefactors"] = merge({
    #             "tissue_hires_scalef": 1, "spot_diameter_fullres": 0.5
    #             }, uns_spatial["scalefactors"])
    #     adata.uns[spatial_key] = {library_id: merge(
    #         {"images": {}, "metadata": {
    #             "file_path": file_path, "source_image_path": file_path_image
    #             }},  uns_spatial)}  # default .uns merge w/ specified
    # if col_sample_id not in adata.obs:
    #     adata.obs.loc[:, col_sample_id] = library_id
    if isinstance(visium, dict) or visium is True:
        if not isinstance(visium, dict):
            visium = {}  # unpack file path & arguments
        adata = sq.read.visium(file_path, **visium)  # read Visium
    else:
        # sdata = sdio.xenium(file_path, n_jobs=n_jobs)
        # adata = sdata.table
        # adata.uns["sdata"] = sdata
        if library_id is None:
            print(f"\n*** USING FILE PATH {file_path} as library ID.\n")
            library_id = str(file_path)
        adata = sdio.xenium(file_path, n_jobs=n_jobs)
        if STORE_UNS_SQUIDPY:
            adata = update_spatial_uns(adata, library_id, col_sample_id)
    return adata


def update_spatial_uns(adata, library_id, col_sample_id, rna_only=False):
    """Copy SpatialData.images to .table.uns (Squidpy-compatible)."""
    imgs = {}
    if "images" in dir(adata):
        for x in adata.images:
            scales = [int(i.split("scale")[1]) for i in adata.images[x] if (
                "scale") in i] if "focus" in x else []
            for i in adata.images[x]:
                key = f"{library_id}{SPATIAL_IMAGE_KEY_SEP}{x}_{i}"
                imgs[key] = sq.im.ImageContainer(
                    adata.images[x][i].image, library_id=library_id)
                if len(scales) > 0 and "scale" in i and str(i.split(
                        "scale")[1]) == str(min(scales)):
                    imgs["hires"] = imgs[key]  # Squidpy-compatible
    if rna_only is True:
        # if col_sample_id in adata.table.obs:
        #     rna = adata.table[adata.table.obs[col_sample_id] == library_id]
        rna = adata.table if "table" in dir(adata) else adata
        rna.uns[SPATIAL_KEY] = {library_id: {"images": imgs}}
        # rna.uns[SPATIAL_KEY]["library_id"] = library_id
        return rna
    else:
        adata.table.uns[SPATIAL_KEY] = {library_id: {"images": imgs}}
        # adata.table.uns[SPATIAL_KEY]["library_id"] = library_id
        if col_sample_id not in adata.table.obs:
            adata.table.obs.loc[:, col_sample_id] = library_id
        return adata


def create_spot_grid(adata, col_cell_type, n_spots, layer="counts", n_jobs=1,
                     title="Grid Label Transfer", cmap=None, kws_plot=True):
    """Create Visium-like data from Xenium data."""
    pass
    # print(f"\n\n{n_spots} by {n_spots} has {n_spots * n_spots} spots\n\n")
    # if kws_plot is True:  # if no keywords to pass but still want plotting
    #     kws_plot = {}
    # if layer is not None:
    #     adata.X = adata.layers[layer].copy()
    # st.pp.normalize_total(adata)  # total count-normalize
    # grid = st.tl.cci.grid(adata, n_row=n_spots, n_col=n_spots,
    #                       use_label=col_cell_type, n_cpus=n_jobs)
    # if kws_plot is not None:  # plot label transfer; adata vs. grid
    #     f_s = kws_plot.pop("figsize", None)
    #     fig, axes = plt.subplots(ncols=2, figsize=f_s if f_s else (20, 8))
    #     for i, a in enumerate([grid, adata]):
    #         plot_space(a, col_cell_type, cmap=cmap, fig=fig, axes=axes[i],
    #                    titles=[col_cell_type, "Grid (Dominant Spots)"][i],
    #                    show=False, **kws_plot)  # plot adata or grid
    #     fig.suptitle(title)
    #     plt.show()
    # return grid, (fig, axes)


def plot_space(grid, color, groups=None,
               cmap=None, size=10, figsize=None, show=True,
               fig=None, axes=None, titles=None, suptitle=None, **kwargs):
    """
    Plot data in spatial coordinates (alternative, stlearn way).

    Notes
    -----
    For additional options for plotting keyword arguments to pass,
    https://stlearn.readthedocs.io/en/latest/\
        stlearn.pl.cluster_plot.html
    """
    pass
    # color = [color] if isinstance(color, str) else list(color)  # color by
    # titles = [titles] * len(color) if isinstance(
    #     titles, str) else color if titles is None else list(titles)  # title
    # if groups is not None:
    #     kwargs.update({"list_clusters": groups})  # only plot certain groups
    # if axes is None:  # determine rows & columns
    #     nrows = kwargs.pop("nrows", 1 if len(color) == 1 else int(
    #         np.sqrt(len(color))))
    #     ncols = kwargs.pop("ncols", 2 if len(color) == 2 else math.ceil(
    #         len(color) / nrows) if nrows * 2 != len(color) else nrows)
    #     fig, axes = plt.subplots(nrows, ncols, squeeze=False,
    #                              figsize=figsize if figsize else (20, 8))
    #     axes = axes.flatten()
    # if not isinstance(axes, (list, np.ndarray)):
    #     axes = [axes]
    # for i, c in enumerate(color):
    #     c_m = cmap if cmap else "default_102" if len(
    #         grid.obs[c].unique()) > 40 else "jana_40" if len(
    #             grid.obs[c].unique()) > 28 else 28 if len(
    #                 grid.obs[c].unique()) > 20 else "default"  # cmap
    #     st.pl.cluster_plot(grid, use_label=c, cmap=c_m, size=size, fig=fig,
    #                        ax=axes[i], show_plot=False, **kwargs)
    #     axes[i].set_title(titles[i])
    # if suptitle is not None:
    #     fig.suptitle(suptitle)
    # if show is True:
    #     plt.show()
    # return fig, axes


def impute_spatial(adata_sp, adata_sc, col_cell_type,
                   col_annotation="tangram_prediction",
                   markers=None, gene_to_lowercase=False, device="cpu",
                   learning_rate=0.1, num_epochs=1000,
                   density_prior=None, mode="cells", perc=0.01,
                   plot=True, plot_genes=None, plot_density=False,
                   seed=0, inplace=False, **kwargs):
    """
    Impute scRNA-seq GEX & annotations onto spatial data.

    Args:
        adata_sp (AnnData): Spatial data object.
        adata_sc (AnnData): sc-RNA-seq data (AnnData object).
        col_cell_type (str, optional): Either a string indicating the
            cell type column shared between spatial and scRNA-seq,
            or a list [scRNA-seq, spatial cell type column].
            The spatial cell type column is currently only
            used for plotting.
        col_annotation (str, optional): Name of the column in which
            to store the predicted cell type in `adata_sp.obs`.
        markers (int | list, optional): Either a number of random
            genes to use for training mapping, or a list of genes.
            Defaults to None (use all).
        gene_to_lowercase (bool, optional): Turn genes to all lowercase
            to reconcile capitalization differences? Defaults to False.
        num_epochs (int, optional): Number of epochs for training.
            Defaults to 500.
        device (str, optional): Use "cpu" or "gpu" (or other specified
            device; "gpu" is automatically changed to "cuda:0").
            Defaults to "cpu".
        density_prior (str | None, optional): None, "rna_count_based".
            or "uniform". Defaults to None.
        mode (str, optional): Map by "cells" or "clusters" or
            "constrained"? It is recommended to use "clusters" when
            the spatial and sc-RNA-seq data come from different
            subjects/specimens. Defaults to "cells".
        plot (bool, optional): Plot? Defaults to True.
        perc (float, optional): Percentile for colormap.
            Defaults to 0.01.
        plot_genes (list, optional): Genes of interest to focus on
            for certain plots. Defaults to None.
        seed (int, optional): Random seed for reproducibility.
            Defaults to 0.
        inplace (bool, optional): Modify data objects in-place? Copy
            if False. Defaults to True.
        kwargs (Any, optional): Additional keyword arguments to pass
            to `tangram.map_cells_to_space()`. If contains `key_added`
            argument, will use that key within `adata_sc.uns` for
            ranked gene markers (if present) instead of re-running
            `scanpy.tl.rank_genes_groups()`; otherwise, will store
            new ranking under "rank_genes_groups_<col_cell_type>".

    Returns:
        tuple: New spatial data, old spatial data, old scRNA-seq data,
            mapping result object, comparison dataframe
    """
    if device == "gpu":
        device = "cuda:0"
    kws = {"suffix": kwargs.pop("suffix", None)
           }  # to construct .obs; suffix, density columns for each cell type
    if inplace is False:
        adata_sc, adata_sp = adata_sc.copy(), adata_sp.copy()
    col_cell_type, col_cell_type_sp = [col_cell_type, col_cell_type] if (
        isinstance(col_cell_type, str)) else col_cell_type
    if col_annotation is None:
        col_annotation = col_cell_type
    if col_cell_type_sp not in adata_sp.obs:
        col_cell_type_sp = None  # if not present, ignore for plotting
    if mode == "clusters":  # if mapping ~ clusters rather than cells...
        kwargs["cluster_label"] = col_cell_type  # ...must give label column
    key = kwargs.pop("key_added", f"rank_genes_groups_{col_cell_type}" if (
        "rank_genes_groups" in adata_sp.uns) else "rank_genes_groups")

    # Obtain Marker/Training Genes for Whole Transcriptome Cell Types
    if key not in adata_sc.uns:  # if need to rank genes (not already done)
        sc.tl.rank_genes_groups(adata_sc, groupby=col_cell_type,
                                use_raw=False, key_added=key)  # rank markers
    if isinstance(markers, (int, float)) or markers is None:
        # if makers not a list of pre-specified genes
        mks = set(np.unique(pd.DataFrame(adata_sc.uns[key]["names"]).melt(
            ).value.values)).intersection(set(adata_sp.var_names))
        if isinstance(markers, (int, float)):  # if markers = #...
            markers = list(pd.Series(list(mks)).sample(
                int(markers)))  # ...random subset of overlapping markers
        else:  # if markers = None...
            markers = list(mks)  # ...use all overlapping genes

    # Preprocessing
    tg.pp_adatas(adata_sc, adata_sp, genes=markers,
                 gene_to_lowercase=gene_to_lowercase)  # preprocess
    if "uniform_density" not in adata_sp.obs:  # issue with Tangram?
        adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[
            0]) / adata_sp.X.shape[0]  # uniform density calculation -> .obs
    if "rna_count_based_density" not in adata_sp.obs:  # issue with Tangram?
        ct_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()  # cts per spot
        adata_sp.obs["rna_count_based_density"] = ct_spot / np.sum(ct_spot)

    # Spatially Map Cells (Calculate Densities/Probabilities for Cell Types)
    ad_map = tg.map_cells_to_space(
        adata_sc, adata_sp, mode=mode, device=device, random_state=seed,
        learning_rate=learning_rate, num_epochs=num_epochs,
        density_prior=density_prior, **kwargs)  # map cells on spatial spots
    tg.project_cell_annotations(
        ad_map, adata_sp, annotation=col_cell_type)  # clusters -> space
    c_l = col_cell_type if mode == "clusters" else None

    # Project/Impute Gene Expression onto Spatial Data
    adata_sp_new = project_genes_m(ad_map, adata_sc, cluster_label=c_l,
                                   gene_to_lowercase=gene_to_lowercase)  # GEX
    adata_sp_new.obsm["tangram_ct_pred"] = adata_sp.obsm[
        "tangram_ct_pred"].loc[adata_sp_new.obs.index]  # predictions -> new
    df_compare = tg.compare_spatial_geneexp(adata_sp_new, adata_sp, adata_sc)

    # Assign Deterministic Labels Based on Densities/Probabilities
    tmp, dfp, preds = construct_obs_spatial_imputation(
        adata_sp_new, adata_sc.copy(), col_cell_type, perc=perc,
        col_annotation=col_annotation, **kws)  # normalized densities; labels
    adata_sp_new.obsm["tangram"] = tmp.obs[dfp.columns]
    adata_sp_new.obs = adata_sp_new.obs.join(preds)

    # Add Back `.uns` & `.obsm` Missing from New Spatial Data Object
    for x in set(adata_sp.uns.keys()).difference(adata_sp_new.uns.keys()):
        adata_sp_new.uns[x] = adata_sp.uns[x]
    for x in set(adata_sp.obsm.keys()).difference(adata_sp_new.obsm.keys()):
        adata_sp_new.obsm[x] = adata_sp.obsm[x]

    # Plot
    if plot is True:  # plotting
        try:
            figs = cr.pp.plot_integration_spatial(
                adata_sp, adata_sp_new, adata_sc=None,
                col_cell_type=[col_cell_type, col_cell_type_sp],
                col_annotation=col_annotation, plot_density=plot_density,
                ad_map=ad_map, df_compare=df_compare, plot_genes=plot_genes)
        except Exception:
            print(traceback.format_exc(), "\n\n", "Plotting failed!")
            figs = traceback.format_exc()
    else:
        figs = None
    return adata_sp_new, adata_sp, adata_sc, ad_map, df_compare, figs


def project_genes_m(adata_map, adata_sc, cluster_label=None, scale=True,
                    gene_to_lowercase=False):
    """
    Transfer gene expression from the single cell onto space.
    Modfied from Tangram's `project_genes()`.

    Args:
        adata_map (AnnData): Mapping data.
        adata_sc (AnnData): Single-cell data
        cluster_label (AnnData): Optional. Should be consistent with
            the 'cluster_label' argument passed to
            `map_cells_to_space` function.
        scale (bool): Optional. Should be consistent with the 'scale'
            argument passed to `map_cells_to_space` function.

    Returns:
        AnnData: spot-by-gene AnnData containing spatial gene
            expression from the single cell data.
    """

    # put all var index to lower case to align
    if gene_to_lowercase is True:
        adata_sc.var.index = [g.lower() for g in adata_sc.var.index]

    # make varnames unique for adata_sc
    adata_sc.var_names_make_unique()

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)

    if cluster_label:
        adata_sc = tg.mapping_utils.adata_to_cluster_expression(
            adata_sc, cluster_label, scale=scale)

    if not adata_map.obs.index.equals(adata_sc.obs.index):
        raise ValueError("The two AnnDatas need to have same `obs` index.")
    if hasattr(adata_sc.X, "toarray"):
        adata_sc.X = adata_sc.X.toarray()
    x_space = adata_map.X.T @ adata_sc.X
    adata_ge = sc.AnnData(
        X=x_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
    )
    training_genes = adata_map.uns["train_genes_df"].index.values
    adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
    return adata_ge


def construct_obs_spatial_imputation(adata_sp, adata_sc, col_cell_type,
                                     perc=0, suffix=None,
                                     col_annotation="tangram_prediction"):
    """
    Helper function to normalize densities/probabilities & transfer
    cell type prediction labels from Tangram (modified).
    Also incorporates code from
    https://github.com/broadinstitute/Tangram/issues/96.
    """
    adata = adata_sp.copy()
    annots = list(pd.unique(adata_sc.obs[col_cell_type]))
    dfp = adata.obsm["tangram_ct_pred"][annots]
    adata.obs = adata.obs.drop(list(dfp.columns), axis=1, errors="ignore")
    dfp = dfp.reindex(adata.obs.index)
    dfp = dfp.clip(dfp.quantile(perc), dfp.quantile(1 - perc), axis=1)  # clip
    dfp = (dfp - dfp.min()) / (dfp.max() - dfp.min())  # normalize
    if suffix:
        dfp = dfp.add_suffix(" ({})".format(suffix))
    adata.obs = pd.concat([adata.obs, dfp], axis=1)
    preds = pd.DataFrame()
    preds[col_annotation] = adata.obsm["tangram_ct_pred"].idxmax(axis=1)
    preds["tangram_score"] = adata.obsm["tangram_ct_pred"].apply(
        lambda x: max(x), axis=1)  # scores
    return adata, dfp, preds


def map_transcripts_to_cells(file_transcripts="transcripts.parquet"):
    """Map Xenium transcripts to cells using CellPose."""
    pass


def segment(directory, nuc_exp=10, file_cellpose=None, file_transcript=None,
            rep_interval=10000, qv_cutoff=20, pix_size=1.7):
    """Segment cells in spatial data."""
    # Check for existence of input file.
    if (not os.path.exists(file_cellpose)):
        raise ValueError(
            f"Specified CellPose output file ({file_cellpose}) not found!")
    if (not os.path.exists(file_transcript)):
        raise ValueError(
            "Specified parquet file ({file_transcript}) not found!")
    if (os.path.exists(directory)):
        raise ValueError(f"Output folder {directory} already exists!")

    # Define additional constants
    nuc_exp_pixel = nuc_exp / pix_size
    nuc_exp_slice = nuc_exp / Z_SLICE_MICRON

    # Read Cellpose segmentation mask
    seg_data = np.load(file_cellpose, allow_pickle=True).item()
    mask_array = seg_data["masks"]
    # Use regular expression to extract dimensions from mask_array.shape
    patt = "\((?P<z_size>\d+), (?P<y_size>\d+), (?P<x_size>\d+)"  # noqa: W605
    m = re.match(patt, str(mask_array.shape))
    mask_dims = {key: int(m.groupdict()[key]) for key in m.groupdict()}

    # Read 5 columns from transcripts Parquet file
    transcripts_df = pd.read_parquet(
        file_transcript, columns=["feature_name", "x_location", "y_location",
                                  "z_location", "qv"])

    # Find distinct set of features.
    features = np.unique(transcripts_df["feature_name"])

    # Create lookup dictionary
    feature_to_index = dict()
    for index, val in enumerate(features):
        feature_to_index[str(val, "utf-8")] = index

    # Find distinct set of cells. Discard first entry which is 0 (non-cell)
    cells = np.unique(mask_array)[1:]

    # Create a cells x features data frame, initialized with 0
    matrix = pd.DataFrame(0, index=range(len(features)), columns=cells,
                          dtype=np.int32)

    # Iterate through all transcripts
    for index, row in transcripts_df.iterrows():
        if index % rep_interval == 0:
            print(index, "transcripts processed.")
        feature = str(row["feature_name"], "utf-8")

        # Ignore transcript below user-specified cutoff
        if row["qv"] < qv_cutoff:
            continue

        # Convert transcript locations from physical space to image space
        x_pixel = row["x_location"] / pix_size
        y_pixel = row["y_location"] / pix_size
        z_slice = row["z_location"] / Z_SLICE_MICRON

        # Add guard rails to make sure lookup falls within image boundaries.
        x_pixel = min(max(0, x_pixel), mask_dims["x_size"] - 1)
        y_pixel = min(max(0, y_pixel), mask_dims["y_size"] - 1)
        z_slice = min(max(0, z_slice), mask_dims["z_size"] - 1)

        # Look up cell_id assigned by Cellpose. Array is in ZYX order.
        cell_id = mask_array[round(z_slice)][round(y_pixel)][round(x_pixel)]

        # If cell_id is 0, Cellpose did not assign the pixel to a cell.
        # Need to perform neighborhood search. See if nearest nucleus is
        # within user-specified distance.
        if cell_id == 0:
            # Define neighborhood boundary for 3D ndarray slicing.
            # Take image boundary into consideration to avoid negative index.
            z_neighborhood_min_slice = max(0, round(z_slice - nuc_exp_slice))
            z_neighborhood_max_slice = min(mask_dims["z_size"], round(
                z_slice + nuc_exp_slice + 1))
            y_neighborhood_min_pixel = max(0, round(y_pixel - nuc_exp_pixel))
            y_neighborhood_max_pixel = min(mask_dims["y_size"], round(
                y_pixel + nuc_exp_pixel + 1))
            x_neighborhood_min_pixel = max(0, round(x_pixel - nuc_exp_pixel))
            x_neighborhood_max_pixel = min(mask_dims["x_size"], round(
                x_pixel + nuc_exp_pixel+1))

            # Call helper function to see if nearest nucleus is
            # w/i user-specified distance.
            cell_id = nearest_cell(
                x_pixel, y_pixel, z_slice, x_neighborhood_min_pixel,
                y_neighborhood_min_pixel, z_neighborhood_min_slice,
                mask_array[
                    z_neighborhood_min_slice: z_neighborhood_max_slice,
                    y_neighborhood_min_pixel: y_neighborhood_max_pixel,
                    x_neighborhood_min_pixel: x_neighborhood_max_pixel],
                nuc_exp=nuc_exp)

        # If cell_id is not 0 at this point, it means the transcript is
        # associated with a cell
        if cell_id != 0:
            # Increment count in feature-cell matrix
            matrix.at[feature_to_index[feature], cell_id] += 1

    # Call a helper function to create Seurat and
    # Scanpy compatible MTX output
    write_sparse_mtx(directory, matrix, cells, features)


def nearest_cell(x_pixel, y_pixel, z_slice,
                 x_neighborhood_min_pixel, y_neighborhood_min_pixel,
                 z_neighborhood_min_slice, mask_array,
                 pix_size=1.7, nuc_exp=10):
    """
    Check if nearest nucleus is within user-specified distance.

    If function returns 0, it means no suitable nucleus was found.
    """
    # For Euclidean distance, we need to convert z-slice to z-micron
    slice_to_pixel = Z_SLICE_MICRON / pix_size
    # When we take a neighborhood slice of mask_array,
    # all indices start at (0,0,0). This INDEX_SHIFT is necessary to
    # reconstruct coordinates from original mask_array.
    ix_shift = np.array([z_neighborhood_min_slice, y_neighborhood_min_pixel,
                         x_neighborhood_min_pixel])
    min_dist = nuc_exp / pix_size
    cell_id = 0

    # Enumerate through all points in the neighborhood
    for index, cell in np.ndenumerate(mask_array):
        # Current point is not assigned to a nucleus.
        if cell == 0:
            continue
        # Current point IS assigned to a nucleus. But is it w/i nuc_exp_pixel?
        else:
            img_loc = np.asarray(index, dtype=float) + ix_shift
            # Convert from z-slice to "z-pixel"
            img_loc[0] *= slice_to_pixel

            transcript_loc = np.array([z_slice * slice_to_pixel,
                                       y_pixel, x_pixel])
            # Calculate Euclidean distance between 2 points
            dist = np.linalg.norm(transcript_loc - img_loc)
            if dist < min_dist:
                min_dist = dist
                cell_id = cell
    return cell_id


def write_sparse_mtx(directory, matrix, cells, features):
    """
    Write Xenium feature-cell matrix in Seurat/Scanpy-compatible MTX.
    """
    # Create the matrix folder.
    os.mkdir(directory)

    # Convert matrix to scipy"s COO sparse matrix.
    sparse_mat = sparse.coo_matrix(matrix.values)

    # Write matrix in MTX format.
    sio.mmwrite(os.path.join(directory, "matrix.mtx"), sparse_mat)

    # Write cells as barcodes.tsv. File name is chosen to ensure
    # compatibility with Seurat/Scanpy.
    with open(os.path.join(directory, "barcodes.tsv"), "w", newline="") as t:
        writer = csv.writer(t, delimiter="\t", lineterminator="\n")
        for cell in cells:
            writer.writerow(["cell_" + str(cell)])

    # Write features as features.tsv. Write 3 columns to ensure
    # compatibility with Seurat/Scanpy.
    with open(os.path.join(directory, "features.tsv"), "w", newline="") as t:
        writer = csv.writer(t, delimiter="\t", lineterminator="\n")
        for f in features:
            feature = str(f, "utf-8")
            if feature.startswith("NegControlProbe_") or feature.startswith(
                    "antisense_"):
                writer.writerow([feature, feature, "Negative Control Probe"])
            elif feature.startswith("NegControlCodeword_"):
                writer.writerow([feature, feature,
                                 "Negative Control Codeword"])
            elif feature.startswith("BLANK_"):
                writer.writerow([feature, feature, "Blank Codeword"])
            else:
                writer.writerow([feature, feature, "Gene Expression"])
    subprocess.run("gzip -f " + os.path.join(directory, "*"), shell=True)


def extract_tiff(file_path="morphology.ome.tif", level=6, **kwargs):
    """
    Extract tiff file (adapted from 10x Genomics website).

    File path can be specified as a full file path or as the
    directory containing the file 'morphology.ome.tif'.
    """
    if level < 0 or level > 6:
        raise ValueError("`level` must be between 0 and 6.")
    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, "morphology.ome.tif")
    with tifffile.TiffFile(file_path) as tif:
        image = tif.series[0].levels[level].asarray()
    kwargs = {**dict(photometric="minisblack", dtype="uint16",
                     tile=(1024, 1024), compression="JPEG_2000_LOSSY",
                     metadata={"axes": "ZYX"}), **kwargs
              }  # defaults for any unspecified variable keyword arguments
    tifffile.imwrite("level_" + str(level) + file_path, image, **kwargs)


def describe_tiff(file_path="morphology.ome.tif"):
    """
    Extract tiff file (adapted from 10x Genomics website).

    File path can be specified as a full file path or as the
    directory containing the file 'morphology.ome.tif'.
    """

    if os.path.isdir(file_path):
        file_path = os.path.join(file_path, "morphology.ome.tif")
    with tifffile.TiffFile(file_path) as tif:
        for tag in tif.pages[0].tags.values():
            if tag.name == "ImageDescription":
                print(tag.name + ":", tag.value)


def command_cellpose(file_path="morphology.ome.tif", diameter=13.6,
                     channels=None):
    """Get CellPose command."""
    channels = [0, 0] if channels is None else channels
    file_path = os.path.abspath(file_path)
    com = str(
        f"python -m cellpose --dir {file_path} --pretrained_model "
        f"nuclei --chan {channels[0]} --chan2 {channels[1]} --img_filter "
        f"_morphology.ome --diameter {diameter} --do_3D --save_tif --verbose")
    return com

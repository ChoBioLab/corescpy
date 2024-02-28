#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-exception-caught
"""
Data manipulation utilities.

@author: E. N. Aslinger
"""

import tifffile as tf
import cv2
import os
import decoupler as dc
import scanpy as sc
import pandas as pd
import numpy as np
from corescpy.processing import get_layer_dict

layers = get_layer_dict()


def create_pseudobulk(adata, col_cell_type, col_sample_id=None,
                      layer=layers["counts"], mode="sum",
                      kws_process=True, **kwargs):
    """Get pseudo-bulk of scRNA-seq data."""
    if kws_process is True:
        kws_process = dict(target_sum=1e6, max_value=10, n_comps=10)
    if layer and layer not in adata.layers:
        raise ValueError(f"{layer} not in adata.layers. Set layer argument to"
                         " None or the name of the layer with count data.")
    pdata = dc.get_pseudobulk(
        adata, sample_col=col_sample_id, groups_col=col_cell_type,
        layer=layer, mode=mode, **kwargs)
    pdata.layers[layers["counts"]] = pdata.X.copy()
    if kws_process is True or isinstance(kws_process, dict):
        sc.pp.normalize_total(pdata, target_sum=kws_process["target_sum"])
        sc.pp.log1p(pdata)
        pdata.layers[layers["log1p"]] = pdata.X.copy()
        sc.pp.scale(pdata, max_value=kws_process["max_value"])
        pdata.layers[layers["scaled"]] = pdata.X.copy()
        sc.tl.pca(pdata, n_comps=kws_process["n_comps"])
    return pdata


def create_condition_combo(adata, col_condition, col_label_new=None, sep="_"):
    """Create a column representing combination of multiple columns."""
    if col_label_new is None:
        col_label_new = "_".join(col_condition)
    if isinstance(adata, pd.DataFrame):  # if adata is dataframe
        adata[col_label_new] = adata[col_condition[0]]  # start w/ 1st
        for x in col_condition[1:]:  # loop to add "_{condition value}"...
            adata[col_label_new] = adata[col_label_new].astype(
                "string") + sep + adata[x]  # ...to make full combination
    else:  # if adata is an AnnData object
        adata.obs = create_condition_combo(adata.obs, col_condition,
                                           col_label_new=col_label_new)
    return adata


def merge_pca_subset(adata, adata_subset,
                     key_added="X_pca", retain_cols=True):
    """Merge gene-subsetted AnnData on which PCA was run into full."""
    ann = adata.copy()
    ixs = np.array(pd.Series(ann.var.index.values).isin(list(
        adata_subset.var.index.values)))
    n_comps = adata_subset.varm["PCs"].shape[1]
    ann.uns["pca"] = adata_subset.uns["pca"]
    ann.obsm[key_added] = adata_subset.obsm[key_added]
    for i in adata_subset.varm:
        ann.varm["PCs"] = np.zeros(shape=(ann.n_vars, n_comps))
        ann.varm["PCs"][ixs] = adata_subset.varm["PCs"]
    if retain_cols is True:
        ann.obs = ann.obs.join(adata_subset.obs, lsuffix="_pre_pca_subset")
    else:
        ann.obs = ann.obs.drop(list(ann.obs.columns.intersection(
            adata_subset.obs.columns)), axis=1).join(adata_subset.obs)
    return ann


def write_ome_tif(file_path, subresolutions=7, pixelsize=0.2125):
    """Write .tif file to .ome.tif (modified from 10x functions)."""
    image = tf.imread(file_path)
    if image.shape[2] > image.shape[0]:
        image = np.transpose(image, (1, 2, 0))
    f_n = os.path.splitext(file_path)[0] + ".ome.tif"
    with tf.TiffWriter(f_n, bigtiff=True) as tif:
        metadata = {
            "SignificantBits": 8,
            "PhysicalSizeX": pixelsize,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixelsize,
            "PhysicalSizeYUnit": "µm",
            # "Channel": {"Name": ["newname1", "newname2", "newname3"]}
            # # Use this line to edit channel names for multi-channel images
        }
        kwargs = dict(
            photometric="minisblack",
            tile=(1024, 1024),
            compression="jpeg2000",
            resolutionunit="CENTIMETER"
        )
        tif.write(
            np.moveaxis(image, -1, 0),
            subifds=subresolutions,
            resolution=(1e4 / pixelsize, 1e4 / pixelsize),
            metadata=metadata,
            **kwargs
        )

        scale = 1
        for i in range(subresolutions):
            scale /= 2
            width = int(np.floor(image.shape[1] * scale))
            height = int(np.floor(image.shape[0] * scale))
            downsample = cv2.resize(image, (width, height),
                                    interpolation=cv2.INTER_AREA)
            tif.write(
                np.moveaxis(downsample, -1, 0),
                subfiletype=1,
                resolution=(1e4 / scale / pixelsize, 1e4 / scale / pixelsize),
                **kwargs
            )


RFX_CONVERT = r"""
require(Seurat)
require(zellkonverter)

convert <- function(file, file_new = NULL, assay = NULL) {
    if (is.null(assay)) {
        assay <- names(seu@assays)  # all assays if not set
    }
    seu <- readRDS(file)  # read RDS into Seurat object
    print(str(seu, 3))  # print object structure (3 levels deep)
    sce <- Seurat::as.SingleCellExperiment(seu, assay = assay)  # convert
    print(sce)
    if (!identical(file_new, FALSE)) {  # if "file_new" is False; don't write
        cat(paste0("\n\n<<< READING DATA >>>\n", file_new))
        zellkonverter::writeH5AD(sce, file_new)  # write .h5ad
    }
    return(sce)
}
"""

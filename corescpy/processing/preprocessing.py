#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-exception-caught
"""
Preprocessing single-cell data.

@author: E. N. Aslinger
"""

import os
from warnings import warn
import traceback
from copy import deepcopy
import matplotlib.pyplot as plt
# import anndata
from anndata import AnnData
import scanpy as sc
import pertpy as pt
import muon
import seaborn
import spatialdata
import pandas as pd
import numpy as np
import corescpy as cr

pd.DataFrame.iteritems = pd.DataFrame.items  # back-compatibility
# REGRESS_OUT_VARS = ["total_counts", "pct_counts_mt"]
REGRESS_OUT_VARS = None  # default variables to regress out
# pp_defaults = dict(cell_filter_pmt=[None, 30],
#                    cell_filter_ncounts=None,
#                    cell_filter_ngene=[200, None],
#                    gene_filter_ncell=[3, None],
#                    gene_filter_ncounts=None)  # can call using 1 arguments


def get_layer_dict():
    """Retrieve layer name conventions."""
    lays = {"preprocessing": "preprocessing",
            "perturbation": "X_pert",
            "unnormalized": "unnormalized",
            "norm_total_counts": "norm_total_counts",
            "log1p": "log1p",
            "unscaled": "unscaled",
            "scaled": "scaled",
            "unregressed": "unregressed",
            "counts": "counts"}
    return lays


def create_object_multi(file_path, kws_init=None, kws_pp=None, spatial=False,
                        kws_cluster=None, kws_harmony=True, **kwargs):
    """Create objects, then preprocess, cluster, & integrate them."""
    # Arguments
    ids = list(file_path.keys())
    if kwargs:
        print(f"\n\nUn-used Keyword Arguments: {kwargs}")
    [kws_init, kws_pp, kws_cluster] = [
        deepcopy(x) for x in [kws_init, kws_pp, kws_cluster]]
    [kws_init, kws_pp, kws_cluster] = [dict(zip(ids, x)) if isinstance(
        x, list) else dict(zip(ids, [x] * len(ids))) for x in [
            kws_init, kws_pp, kws_cluster]]  # dictionaries for each

    if spatial is True:
        for x in kws_init:
            if "col_sample_id" not in kws_init[x]:
                kws_init[x]["col_sample_id"] = "unique.idents"
            # kws_init[x]["spatial"] = spatial

    # Create AnnData Objects
    # selves = dict(zip(ids, [cr.pp.read_spatial(
    #     file_path[f], **kws_init[f], library_id=f
    #     ) if spatial is True else cr.Omics(file_path[f], **kws_init[f]) if (
    #         "kws_process_guide_rna" not in kws_init) else cr.Crispr(
    #             file_path[f], **kws_init[f]) for f in file_path])
    #               )  # create individual objects
    if spatial is True:
        selves = dict(zip(file_path.keys(), [cr.Spatial(
            cr.pp.read_spatial(file_path[f], **kws_init[f], library_id=f),
            **kws_init[f], library_id=f) for f in file_path]))
    else:
        selves = dict(zip(ids, [cr.Omics(file_path[f], **kws_init[f]) if (
            "kws_process_guide_rna" not in kws_init) else cr.Crispr(
                file_path[f], **kws_init[f]) for f in file_path]))

    # Preprocessing & Clustering
    for x in selves:  # preprocess & cluster each object
        if kws_pp[x] is not None:
            print(f"\n<<< PREPROCESSING {x} >>>")
            selves[x].preprocess(**kws_pp[x])
        if kws_cluster[x] is not None:
            print(f"\n<<< CLUSTERING {x} >>>")
            selves[x].cluster(**kws_cluster[x])
    print(f"\n<<< CONCATENATING OBJECTS: {', '.join(ids)} >>>")
    col_id = kws_init[list(kws_init.keys())[0]]["col_sample_id"] if (
        spatial is True) else selves[ids[0]]._columns["col_sample_id"]
    if isinstance(selves[list(selves.keys())[0]].adata,
                  spatialdata.SpatialData):
        adata = spatialdata.concatenate(
            [selves[x].adata for x in selves], region_key=col_id,
            join="outer", batch_key=col_id if col_id else "unique.idents",
            batch_categories=ids, index_unique="-", fill_value=None,
            uns_merge="unique")  # concatenate spatial
    else:
        adata = AnnData.concatenate(
            *[selves[x].adata for x in selves],
            join="outer", batch_key=col_id if col_id else "unique.idents",
            batch_categories=ids, index_unique="-", fill_value=None,
            uns_merge="same")  # concatenate adata

    # Integrate
    if kws_harmony is not None:
        print("\n<<< INTEGRATING WITH HARMONY >>>")
        # sc.tl.pca(adata)  # PCA
        sc.external.pp.harmony_integrate(
            adata, col_id, basis="X_pca",
            adjusted_basis="X_pca_harmony", **kws_harmony)  # harmony
        adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]  # assign new basis
        adata.uns["harmony"] = True
    return adata


def create_object(file, col_gene_symbols="gene_symbols", assay=None,
                  kws_process_guide_rna=None, assay_gdo=None, raw=False,
                  gex_only=False, prefix=None, spatial=False, **kwargs):
    """
    Create object from Scanpy- or Muon-compatible file(s) or object.
    """
    # Load Object (or Copy if Already AnnData or MuData)
    csid = kwargs.pop("col_sample_id", None)
    if isinstance(file, (AnnData, spatialdata.SpatialData,
                         muon.MuData)):  # if already data object
        print("\n\n<<< LOADING OBJECT >>>")
        adata = file.copy() if "copy" in dir(file) else file
        if "table" in dir(adata) and "original_ix" not in adata.table.uns:
            adata.table.uns["original_ix"] = adata.table.obs.index.values
    # Spatial Data
    elif spatial not in [None, False]:
        kwargs = {**dict(prefix=prefix, col_sample_id=csid,
                         col_gene_symbols=col_gene_symbols),
                  **kwargs}  # user-specified + variable keyword arguments
        adata = cr.pp.read_spatial(file_path=file, **kwargs)
        if "table" in dir(adata) and "original_ix" not in adata.table.uns:
            adata.table.uns["original_ix"] = adata.table.obs.index.values

    # Non-Spatial Data
    elif isinstance(file, (str, os.PathLike)) and os.path.splitext(
            file)[1] == ".h5mu":  # MuData
        print(f"\n\n<<< LOADING FILE {file} with muon.read() >>>")
        adata = muon.read(file)
    elif isinstance(file, dict):  # metadata in protospacer files
        print("\n\n<<< LOADING PROTOSPACER METADATA >>>")
        adata = cr.pp.combine_matrix_protospacer(
            **file, col_gene_symbols=col_gene_symbols, gex_only=gex_only,
            prefix=prefix, **kwargs)  # + metadata from protospacer
    elif os.path.isdir(file):  # if directory, assume 10x format
        print(f"\n\n<<< LOADING 10X FILE {file} >>>")
        adata = sc.read_10x_mtx(
            file, var_names=col_gene_symbols, cache=True,
            gex_only=gex_only, prefix=prefix, **kwargs)  # read 10x
    else:  # other, catch-all attempt
        print(f"\n\n<<< LOADING FILE {file} with sc.read() >>>")
        adata = sc.read(file)

    # For CRISPR Data with gRNAA in Separate Assay of Muon Object
    if assay_gdo:
        print(f"\n\n<<< Joining {assay_gdo} (gRNA) & {assay} assay data >>>")
        kws_pmc = dict(assay=[assay, assay_gdo])
        for x in ["col_guide_rna", "col_num_umis", "feature_split",
                  "guide_split", "keep_extra_columns"]:
            if (kws_process_guide_rna and x in kws_process_guide_rna) or (
                    x in kwargs):
                kws_pmc.update({x: kwargs[x] if (
                    x in kwargs) else kws_process_guide_rna[x]})
        adata = cr.pp.process_multimodal_crispr(adata, **kws_pmc)

    # Use Raw Data (Optional)
    if raw is True:
        if "raw" in dir(adata):
            adata = adata.raw.to_adata()
            adata.raw = None
        else:
            warn("Unable to set adata to adata.raw (attribute not in adata).")

    # Formatting & Initial Counts
    try:
        (adata.table if isinstance(adata, spatialdata.SpatialData) else adata
         ).var_names_make_unique()
    except Exception:
        print(traceback.format_exc())
        warn("\n\n\nCould not make var names unique.")
    try:
        (adata.table if isinstance(adata, spatialdata.SpatialData) else adata
         ).obs_names_make_unique()
    except Exception:
        print(traceback.format_exc())
        warn("\n\n\nCould not make obs names unique.")
    cr.tl.print_counts(adata, title="Initial")

    # Gene Symbols -> Index of .var
    rename_var_index(adata.table if isinstance(
        adata, spatialdata.SpatialData) else adata,
                     assay=assay, col_gene_symbols=col_gene_symbols)

    # Process Guide RNA
    if kws_process_guide_rna not in [None, False]:
        if assay:
            apg = cr.pp.process_guide_rna(
                adata.mod[assay].copy(), **kws_process_guide_rna)
            adata = adata[adata.obs.index.isin(apg.obs.index)]  # subset
            adata.mod[assay] = apg
        else:
            adata = cr.pp.process_guide_rna(adata, **kws_process_guide_rna)
        cct = kws_process_guide_rna["col_cell_type"] if "col_cell_type" in (
            kws_process_guide_rna) else None  # to group counts ~ cell type
        cr.tl.print_counts(adata, title="Post-gRNA Processing", group_by=cct)

    # Layers & QC
    layers = cr.pp.get_layer_dict()  # standard layer names
    rna = adata.table if isinstance(adata, spatialdata.SpatialData
                                    ) else adata[assay] if assay else adata
    if layers["counts"] not in rna.layers:
        if isinstance(adata, spatialdata.SpatialData):
            adata.table.layers[layers["counts"]] = adata.table.X.copy()
        elif assay:
            adata[assay].layers[layers["counts"]] = adata[assay].X.copy()
        else:
            adata.layers[layers["counts"]] = adata.X.copy()
    # cr.pp.perform_qc(adata.copy(), hue=col_sample_id)  # plot QC
    print("\n\n", adata)
    return adata


def process_data(adata, col_gene_symbols=None, col_cell_type=None,
                 outlier_mads=None, cell_filter_pmt=None,
                 cell_filter_ncounts=None, cell_filter_ngene=None,
                 gene_filter_ncell=None, gene_filter_ncounts=None,
                 remove_malat1=False, target_sum=1e4, kws_hvg=True,
                 kws_scale=None, regress_out=REGRESS_OUT_VARS,
                 custom_thresholds=None, figsize=None, **kwargs):
    """
    Perform various data processing steps.

    Args:
        adata (AnnData or MuData): The input data object.
        col_gene_symbols (str, optional): The name of the column or
            index in `.var` containing gene symbols. Defaults to None.
        col_cell_type (str, optional): The name of the column
            in `.obs` containing cell types. Defaults to None.
        target_sum (float, optional): Total-count normalize to
            <target_sum> reads per cell, allowing between-cell
            comparability of counts. If None, total count-normalization
            will not be performed. Defaults to 1e4.
        outlier_mads (float or int or dict): To calculate outliers
            based on MADs (see SC Best Practices). If a dictionary,
            key based on names of columns added by QC. Filtering
            will be performed based on outlier status rather than
            other arguments to this function if not None.
            Defaults to None.
        cell_filter_pmt (list, optional): The range of percentage of
            mitochondrial genes per cell allowed. Will filter out cells
            that have outside the range [minimum % mt, maximum % mt].
            If either element is None, filtering is not performed
            according to that property. (The same goes for the
            other filtering arguments below.)
            Defaults to 5.
        cell_filter_ncounts (list, optional): Retain only cells
            that have a number of reads within the
            range specified: [minimum reads, maximum].
        cell_filter_ngene (list, optional): Retain only cells that
            express a number of genes within the
            range specified:
            [minimum genes expressed within a cell, maximum].
            Specify 0 as the minimum in order still to calculate
            certain metrics but without filtering.
            Defaults to None (no filtering on this property performed).
        gene_filter_ncell (list, optional): Retain only genes that
            are expressed by a number of cells within the
            range specified: [minimum cells expressing gene, maximum].
            If either element is None, filtering is not performed
            according to that property.
            If True, filtering criteria are derived from the data
                (i.e., outlier calculation).
            Defaults to None (no filtering on this property performed).
        gene_filter_ncounts (list, optional): Retain only genes with
            counts (# molecules of a given gene)
            within the range specified:
            [minimum count, maximum (high may indicate doublets)].
            Defaults to None (no filtering on this property performed).
        remove_malat1 (bool, optional): Whether to filter out
            Malat1 gene (often a technical issue). Defaults to False.
        kws_hvg (dict or bool, optional): The keyword arguments for
            detecting variable genes or True for default arguments.
            To filter by HVGs, include an
            extra argument in this dictionary: {"filter": True}.
            Otherwise, a highly_variable genes column is created and
            can be used in clustering, but non-HVGs will be retained in
            the data for other uses. Defaults to True.
        kws_scale (int, bool, or dict, optional): Specify True to
            center on average gene expression and scale to unit
            variance, as an integer to additionally clip maximum
            standard deviation, or as a dictionary with keyword
            arguments in order to normalize with reference to some
            category in a column in `.obs`.
            If a dictionary, under the key "col_reference", specify the
            name of the columns containing reference vs. other, and
            under the key "key_reference", the label of the reference
            category within that column.
            Note that using a dictionary (z-scoring) will result in
            the `.X` attribute being reset to the
            un-log-normalized data, then scaled and set to
            the scaled data. To avoid this behavior, pass "layer" in
            the dictionary to specify a layer to set before scaling.
        regress_out (list or None, optional): The variables to regress
            out. Defaults to `pp.preprocessing.REGRESS_OUT_VARS`.
        custom_thresholds (dict or None, optional): A dictionary, keyed
            by column names on which to filter data (before all other
            steps), with lists [minimum, maximum] as each item.
        figsize (tuple or None, optional): Figure size.
        **kwargs: Additional keyword arguments.

    Returns:
        adata (AnnData): The processed data object.
        figs (dict): A dictionary of generated figures.
    """
    # Setup Object
    layers = cr.pp.get_layer_dict()  # layer names
    ann = adata.copy()  # copy so passed AnnData object not altered inplace
    if custom_thresholds:  # filter on custom thresholds
        for x in custom_thresholds:
            dff = ann.obs[x] if x in ann.obs else ann.var[x]  # .obs or .var?
            if custom_thresholds[x][0]:  # filter by minimum
                ann = ann[ann.obs[x] >= custom_thresholds[x][0]]
            if custom_thresholds[x][1]:  # filter by maximum
                ann = ann[dff <= custom_thresholds[x][1]]
    if layers["counts"] not in ann.layers:
        if ann.X.min() < 0:  # if any data < 0, can't be gene read counts
            raise ValueError(
                f"Must have counts in `adata.layers['{layers['counts']}']`.")
        ann.layers[layers["counts"]] = ann.X.copy()  # store counts in layer
    else:
        ann.X = ann.layers[layers["counts"]]  # use counts layer
    # ann.layers[layers["preprocessing"]] = ann.X.copy() # set original layer
    ann.obs["n_counts"] = ann.X.sum(1)
    ann.obs["log_counts"] = np.log(ann.obs["n_counts"])
    ann.obs["n_genes"] = (ann.X > 0).sum(1)

    # Initial Information/Arguments
    if col_gene_symbols == ann.var.index.names[0]:  # if symbols=index...
        col_gene_symbols = None  # ...so functions will refer to index name
    figs = {}
    if isinstance(kws_scale, dict):  # if extracted all sc.pp.scale arguments
        max_val, cen = [kws_scale.pop(x, None)
                        for x in ["max_value", "zero_center"]]
        if any((i for i in [max_val, cen])) and len(
                kws_scale) == 0:  # if extracted all regular-scale keywords
            kws_scale = True  # so doesn't think scaling by reference (CRISPR)
    else:
        max_val, cen = 0, True
    kws_hvg = {} if kws_hvg is True else kws_hvg
    filter_hvgs = kws_hvg.pop("filter") if "filter" in kws_hvg else False
    n_top = kwargs.pop("n_top", 10)
    sids = [np.nan if f"col_{x}" not in kwargs else np.nan if kwargs[
        f"col_{x}"] is None else kwargs.pop(f"col_{x}") for x in [
            "sample_id", "subject", "batch"]]  # batch/sample/subject
    sids = list(pd.Series(sids).dropna().unique())  # unique & not None
    if len(sids) == 0:
        sids = None
    if kwargs:
        print(f"\nUn-Used Keyword Arguments: {kwargs}\n\n")

    # Highly Expressed Genes
    try:
        figs["highly_expressed_genes"], axs = plt.subplots(figsize=figsize)
        sc.pl.highest_expr_genes(
            adata, ax=axs, n_top=n_top, gene_symbols=col_gene_symbols)
    except Exception as err:
        print(traceback.format_exc())
        warn(f"\n\n{'=' * 80}\n\nCouldn't plot highly expressed genes!")
        figs["highly_expressed_genes"] = err

    # Set Up Layer & Variables
    if outlier_mads is not None:  # if filtering based on calculating outliers
        if isinstance(outlier_mads, (int, float)):  # same MADs, all metrics
            qc_mets = ["log1p_total_counts", "log1p_n_genes_by_counts",
                       "pct_counts_in_top_20_genes"]
            outlier_mads = dict(zip(qc_mets, [outlier_mads] * len(qc_mets)))
    cr.tl.print_counts(ann, title="Initial", group_by=col_cell_type)

    # Exploration & QC Metrics
    print("\n<<< PERFORMING QUALITY CONTROL ANALYSIS>>>")
    figs["qc_metrics"] = cr.pp.perform_qc(ann, hue=sids)  # QC metrics & plots

    # Basic Filtering (DO FIRST...ALL INPLACE)
    print("\n<<< FILTERING CELLS (TOO FEW GENES) & GENES (TOO FEW CELLS) >>>")
    if cell_filter_ngene:
        sc.pp.filter_cells(ann, min_genes=cell_filter_ngene[0])
    cr.tl.print_counts(ann, title="Post-Basic Filter", group_by=col_cell_type)
    if gene_filter_ncell:
        sc.pp.filter_genes(ann, min_cells=gene_filter_ncell[0])
    cr.tl.print_counts(ann, title="Post-Basic Filter", group_by=col_cell_type)

    # Further Filtering
    print("\n<<< FURTHER CELL & GENE FILTERING >>>")
    ann = cr.pp.filter_qc(
        ann, outlier_mads=outlier_mads, cell_filter_pmt=cell_filter_pmt,
        cell_filter_ncounts=cell_filter_ncounts,
        cell_filter_ngene=cell_filter_ngene,
        gene_filter_ncell=gene_filter_ncell,
        gene_filter_ncounts=gene_filter_ncounts)  # filter based on QC metrics
    if remove_malat1 is True:  # remove MALAT1 genes (often technical issue)?
        ann = ann[:, ~ann.var_names.str.startswith('MALAT1')]  # remove MALAT1
    cr.tl.print_counts(ann, title="Post-Filter", group_by=col_cell_type)

    # Doublets
    # doublets = detect_doublets(adata)
    # if remove_doublets is True:
    #     adata
    #     # TODO: doublets

    # Normalization
    print("\n<<< NORMALIZING CELL COUNTS >>>")
    if target_sum is not None:  # total-count normalization INPLACE
        print("\n\t*** Total-count normalizing...")
        sc.pp.normalize_total(ann, target_sum=target_sum, copy=False)
    print(ann)
    print(f"\n\t*** Log-normalizing => `.X` & {layers['log1p']} layer...")
    sc.pp.log1p(ann)  # log-transformed; INPLACE
    ann.layers[layers["log1p"]] = ann.X.copy()  # also keep in layer
    ann.raw = ann.copy()  # before HVG, batch correction, etc.

    # Gene Variability (Detection, Optional Filtering)
    print("\n<<< DETECTING VARIABLE GENES >>>")
    sc.pp.highly_variable_genes(ann, **kws_hvg)
    cr.pl.plot_hvgs(ann, palette=["red", "black"], figsize=figsize)
    if filter_hvgs is True:
        print("\n<<< FILTERING BY HIGHLY VARIABLE GENES >>>")
        ann = ann[:, ann.var.highly_variable]  # filter by HVGs
        cr.tl.print_counts(ann, title="HVGs", group_by=col_cell_type)

    # Set .raw if Needed
    if regress_out or max_val:
        adata.raw = adata.copy()

    # Regress Out Confounds
    if regress_out:
        print(f"\n<<< REGRESSING OUT CONFOUNDS >>>\n\n\t{regress_out}")
        ann.layers[layers["unregressed"]] = ann.X.copy()
        sc.pp.regress_out(ann, regress_out, copy=False)
        warn("Confound regression doesn't yet properly use layers.")

    # Gene Expression Normalization
    if kws_scale is not None:  # scale by overall or reference
        print("\n<<< NORMALIZING RAW GENE EXPRESSION >>>")
        normalize_genes(ann, kws_reference=kws_scale if isinstance(
            kws_scale, dict) else None, max_value=max_val, zero_center=cen)
        print(f"\n\t*** Scaling => `.X` & {layers['scaled']} layer...")
        ann.layers[layers["scaled"]] = ann.X.copy()  # also keep in layer

    # Final Data Examination
    cr.tl.print_counts(ann, title="Post-Processing", group_by=col_cell_type)
    # figs["qc_metrics_post"] = cr.pp.perform_qc(ann.copy(), hue=sids)
    return ann, figs


def normalize_genes(adata, zero_center=True, max_value=None,
                    kws_reference=None, **kwargs):
    """Normalize gene expression relative to overall or reference."""
    if kws_reference is not None:  # with reference to some control
        print("\n\t*** Resetting to raw counts before scaling...")
        if any((x not in kws_reference for x in [
                "col_reference", "key_reference"])):
            err = "'col_reference' & 'key_reference' must be in normalization"
            raise ValueError(err + " argument if method = 'z'.")
        print("\n\t*** Z-scoring (relative to "
              f"{kws_reference['key_reference']})")
        adata = z_normalize_by_reference(adata, **kws_reference)
    else:  # with reference to whole sample
        print("\n\t*** Scaling gene expression...")
        if max_value:
            print(f"\n\t*** Clipping maximum GEX SD to {max_value}...")
            kwargs.update({"max_value": max_value})
        sc.pp.scale(adata, copy=False, zero_center=zero_center,
                    **kwargs)  # center/standardize


def rename_var_index(adata, assay=None, col_gene_symbols="gene_symbols"):
    """Make sure `.var` index name is same as `gene_symbols`."""
    ixn = (adata if assay is None else adata.mod[assay] if "mod" in dir(
        adata) else adata[assay]).var.index.names[0]  # var index name
    if "mod" in dir(adata) and ixn != col_gene_symbols:
        for x in adata.mod.keys():
            adata.mod[x].var = adata.mod[x].var.reset_index().set_index(
                col_gene_symbols) if (col_gene_symbols in adata.mod[
                    x].var.columns) else adata.mod[x].var.rename_axis(
                        col_gene_symbols)
    elif assay is not None and ixn != col_gene_symbols:
        adata[assay].var = adata[assay].var.reset_index().set_index(
            col_gene_symbols) if (col_gene_symbols in adata[
                assay].var.columns) else adata[assay].var.rename_axis(
                    col_gene_symbols)
    elif ixn != col_gene_symbols:
        adata.var = adata.var.reset_index().set_index(col_gene_symbols) if (
            col_gene_symbols in adata.var.columns) else adata.var.rename_axis(
                col_gene_symbols)
    else:
        pass  # index already = col_gene_symbols


def check_normalization(adata, n_genes=1000):
    """Check if an AnnData object is log-normalized."""
    return (adata.X[:1000].min() >= 0) or (adata.X[:n_genes].max() <= 9.22)


def z_normalize_by_reference(adata, col_reference="Perturbation",
                             key_reference="Control",
                             retain_zero_variance=True, layer=None, **kwargs):
    """
    Mean-center & standardize by reference condition
    (within-batch option).
    If `retain_zero_variance` is True, genes with
    zero variance are retained.
    """
    if kwargs:
        print(f"\nUn-Used Keyword Arguments: {kwargs}\n\n")
    if layer is None:
        layer = cr.pp.get_layer_dict()["counts"]  # raw counts layer
    if layer is not None:
        adata.X = adata.layers[layer].copy()  # reset to raw counts
    if layer:
        adata.X = adata.layers[layer].copy()
    gex = adata.X.copy()  # gene expression matrix (full)
    gex_ctrl = adata[adata.obs[col_reference] == key_reference
                     ].X.A.copy()  # reference condition
    gex, gex_ctrl = [q.A if "A" in dir(q) else q
                     for q in [gex, gex_ctrl]]  # sparse -> dense matrix
    mus, sds = np.nanmean(gex_ctrl, axis=0), np.nanstd(
        gex_ctrl, axis=0)  # means & SDs of reference condition genes
    if retain_zero_variance is True:  # retain zero-variance genes?
        sds[sds == 0] = 1   # to retain, set 0 variance to unit variance
    adata.X = (gex - mus) / sds  # z-score gene expression
    return adata


def perform_qc(adata, log1p=True, hue=None, patterns=None, layer=None):
    """Calculate & plot quality control metrics."""
    figs = {}
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    if patterns is None:
        patterns = [("MT-", "mt-"), ("RPS", "RPL", "rps", "rpl"), (
            "^HB[^(P)]", "^hb[^(p)]")]  # pattern matching for gene symbols
        patterns = dict(zip(["mt", "ribo", "hb"], patterns))  # dictionary
    names = dict(zip(["mt", "ribo", "hb"],
                     ["Mitochondrial", "Ribosomal", "Hemoglobin"]))
    p_names = [names[k] if k in names else k for k in patterns]  # "pretty"
    patterns_names = dict(zip(patterns, p_names))  # map abbreviated to pretty
    rename_perc = dict(zip([f"pct_counts_{p}" for p in names], [
        names[p] + " " + "%" + " of Counts" for p in names]))

    # Calculate QC Metrics
    qc_vars = []  # to hold mt, rb, hb, etc. if present in data
    print(f"\n\t*** Detecting {', '.join(p_names)} genes...")
    for k in patterns:  # calculate MT, RB, HB counts
        gvars = adata.var_names.str.startswith(patterns[k])
        if any(gvars):
            qc_vars += [k]
        adata.var[k] = gvars
    print("\n\t*** Calculating & plotting QC metrics...\n\n")
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, log1p=log1p,
                               percent_top=None, inplace=True)  # QC metrics

    # Determine Available QC Metrics & Color-Coding (e.g., by Subject)
    pct_n = [f"pct_counts_{k}" for k in qc_vars]  # "% counts" variables
    for x in pct_n:  # replace NaN % (in case no mt, rb, hb) wth 0
        adata.obs.loc[adata.obs[x].isnull(), x] = 0
    patterns_names = dict(zip(qc_vars, [patterns_names[k] for k in qc_vars]))
    hhh = [hue] if isinstance(hue, str) or hue is None else hue  # hues
    if len(hhh) > 1:  # only include sample ID as hue if others only 1 value
        hht = list(np.array(hhh)[np.where([
            len(adata.obs[h].unique()) > 1 for h in hhh])[0]])
        hhh = list(pd.unique(hht if len(hht) > 0 else [
            hhh[0]]))  # if no non-unique hue values, still color by subject

    # % Counts (MT, RB, HB) versus Counts (Scatter Plots)
    scatter_vars = pct_n + ["n_genes_by_counts", "cell_area", "nucleus_area"]
    scatter_vars = list(set(scatter_vars).intersection(adata.obs.columns))
    rrs, ccs = len(pd.unique(hhh)), len(scatter_vars)
    fff, axs = plt.subplots(rrs, ccs, figsize=(
        5 * ccs, 5 * rrs), sharex=False, sharey=False)  # subplot grid
    for i, h in enumerate(pd.unique(hhh)):
        for j, v in enumerate(scatter_vars):
            aij = axs if not isinstance(axs, np.ndarray) else axs[
                i, j] if len(axs.shape) > 1 else axs[j] if ccs > 1 else axs[i]
            try:  # % mt, etc. vs. counts
                sc.pl.scatter(adata, x="total_counts", y=v, ax=aij,
                              color=h, frameon=False, show=False)  # scatter
                if aij.legend_ is not None and v != "n_genes_by_counts":
                    aij.legend_.remove()  # legend only on last column
                aij.set_title(f"{v} (by {h})" if h else v)  # title
            except Exception as err:
                print(traceback.format_exc())
                warn(f"\n\n{'=' * 80}\n\nCouldn't plot {h} vs. {v}: {err}")
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()

    # All QC Variables versus Each Other (Joint Plots, Scatter & KDE)
    for h in pd.unique(hhh):
        figs[f"qc_scatter_by_{h}" if h else "qc_scatter"] = fff
        try:  # pairplot of all QC variables (hue=grouping variable, if any)
            ctm = list(set(
                ["n_genes_by_counts", "total_counts", "log1p_total_counts"]
                ).intersection(adata.obs.columns))
            vam = pct_n + ctm + list([h] if h else [])  # QC variable names
            mets_df = adata.obs[vam].rename_axis("Metric", axis=1).rename(
                {"total_counts": "Total Counts in Cell",
                 "cell_area": "Cell Area", "nucleus_area": "Nucleus Area",
                 "log1p_total_counts": "Log-Normalized Total Counts",
                 **rename_perc, "n_genes_by_counts": "Genes Detected in Cell",
                 **patterns_names}, axis=1)  # rename
            fff = seaborn.pairplot(
                mets_df, hue=h if h else None, height=3, diag_kind="hist",
                #  diag_kind="kde", diag_kws=dict(fill=True, cut=0),
                plot_kws=dict(marker=".", linewidth=0.05))  # pair
        except Exception as err:
            fff = err
            print(traceback.format_exc())
        figs[f"pairplot_by_{h}" if h else "pairplot"] = fff

    # % Counts (MT, RB, HB) Distribution (KDE) Plots
    if len(pct_n) > 0:  # if any QC vars (e.g., MT RNA) present...
        try:  # KDE of % of counts ~ QC variable
            mets_df = adata.obs[pct_n].rename_axis("Metric", axis=1).rename(
                patterns_names, axis=1).stack().rename_axis(["bc", "Metric"])
            figs["pct_counts_kde"] = seaborn.displot(
                mets_df.to_frame("Percent Counts"), x="Percent Counts",
                hue="Metric", kind="kde", cut=0, fill=True)  # KDE plot
        except Exception as err:
            print(traceback.format_exc())
            figs["pct_counts_kde"] = err
    else:
        figs["pct_counts_kde"] = "No percent counts variables"
    try:  # log1p genes by counts (x) vs. total counts (y)
        figs["qc_log"] = seaborn.jointplot(
            data=adata.obs, x="log1p_total_counts",
            y="log1p_n_genes_by_counts", kind="hex")  # jointplot
    except Exception as err:
        print(traceback.format_exc())
        figs["qc_log"] = err
    return figs


def filter_qc(adata, outlier_mads=None, cell_filter_pmt=None,
              cell_filter_ncounts=None, cell_filter_ngene=None,
              gene_filter_ncell=None, gene_filter_ncounts=None, nmads=2):
    """Filter low-quality/outlier cells & genes."""
    ann = adata.copy()
    if isinstance(cell_filter_pmt, (int, float)):  # if just 1 # for MT %...
        cell_filter_pmt = [0, cell_filter_pmt]  # ...assume it's for maximum %
    if cell_filter_pmt is None:  # None = no MT filter but calculates metrics
        cell_filter_pmt = [0,
                           100]
    min_mt, max_mt = cell_filter_pmt if cell_filter_pmt else None, None
    if outlier_mads is not None:  # automatic filtering using outlier stats
        outliers = ann.obs[outlier_mads.keys()]
        print(f"\n<<< DETECTING OUTLIERS {outliers.columns} >>>")
        for x in outlier_mads:
            outliers.loc[:, f"outlier_{x}"] = cr.tl.is_outlier(
                ann.obs, outlier_mads[x], nmads)  # metric outlier column x
        cols_outlier = list(set(
            outliers.columns.difference(ann.obs.columns)))
        outliers.loc[:, "outlier"] = outliers[cols_outlier].any()  # binary
        print(f"\n<<< FILTERING OUTLIERS ({cols_outlier}) >>>")
        ann.obs = ann.obs.join(outliers[["outlier"]])  # + outlier column
        print(f"Total Cell Count: {ann.n_obs}")
        ann = ann[(~ann.obs.outlier) & (~ann.obs.mt_outlier)].copy()  # drop
        print(f"Cell Count (Outliers Dropped): {ann.n_obs}")
    else:  # manual filtering
        print("\n<<< PERFORING THRESHOLD-BASED FILTERING >>>")
        print(f"\nTotal Cell Count: {ann.n_obs}")
        print("\n\t*** Filtering cells by mitochondrial gene percentage...")
        print(f"\n\tMinimum={min_mt}\n\tMaximum={max_mt}")
        if min_mt and max_mt:
            ann = ann[(ann.obs.pct_counts_mt < max_mt) * (
                ann.obs.pct_counts_mt >= min_mt)]  # filter by MT %
        print(f"\tNew Count: {ann.n_obs}")
        cell_filter_ngene, cell_filter_ncounts, gene_filter_ncell, \
            gene_filter_ncounts = [x if x else [None, None] for x in [
                cell_filter_ngene, cell_filter_ncounts,
                gene_filter_ncell, gene_filter_ncounts]]  # iterable if None

        # Filter Cells
        print("\n\t*** Filtering genes based on # of genes expressed...")
        for i, x in zip(["min_genes", "max_genes"], cell_filter_ngene):
            if x is not None:
                sc.pp.filter_cells(ann, **{i: x})
                print(f"\n\t{i}={x}\tCount: {ann.n_obs}")
            else:
                print(f"\n\tNo {i}")
        print("\n\t*** Filtering cells based on # of reads...")
        for i, x in zip(["min_counts", "max_counts"], cell_filter_ncounts):
            if x is not None:
                sc.pp.filter_cells(ann, **{i: x})
                print(f"\n\t{i}={x}\tCount: {ann.n_obs}")
            else:
                print(f"\n\tNo {i}")

        # Filter Genes
        print("\n\t*** Filtering genes based on # cells expressing them...")
        for i, x in zip(["min_cells", "max_cells"], gene_filter_ncell):
            if x is not None:
                sc.pp.filter_genes(ann, **{i: x})
                print(f"\n\t{i}={x}\tCount: {ann.n_vars}")
            else:
                print(f"\n\tNo {i}")
        print("\n\t*** Filtering genes based on counts...")
        for i, x in zip(["min_counts", "max_counts"], gene_filter_ncounts):
            if x is not None:
                sc.pp.filter_genes(ann, **{i: x})
                print(f"\n\t{i}={x}\tCount: {ann.n_vars}")
            else:
                print(f"\n\tNo {i}")
        print(f"\nPost-Filtering Cell #: {ann.n_obs}, Gene #: {ann.n_vars}\n")
    return ann


def remove_batch_effects(adata, col_cell_type="leiden",
                         col_sample_id="orig.ident", plot=True, **kws_train):
    """Remove batch effects (IN PROGRESS)."""
    if not kws_train:
        kws_train = dict(max_epochs=100, batch_size=32,
                         early_stopping=True, early_stopping_patience=25)
    train = adata.copy()
    train.obs["cell_type"] = train.obs[col_cell_type].tolist()
    train.obs["batch"] = train.obs[col_sample_id].tolist()
    if plot is True:
        sc.pl.umap(train, color=[col_sample_id, col_cell_type],
                   wspace=.5, frameon=False)
    print("\n<<< PREPARING DATA >>>")
    pt.tl.SCGEN.setup_anndata(
        train, batch_key="batch", labels_key="cell_type")  # prepare AnnData
    model = pt.tl.SCGEN(train)
    print("\n<<< TRAINING >>>")
    model.train(**kws_train)  # training
    print("\n<<< CORRECTING FOR BATCH EFFECTS >>>")
    corr = model.batch_removal()  # batch-corrected adata
    if plot is True:
        sc.pp.neighbors(corr)
        sc.tl.umap(corr)
        sc.pl.umap(corr, color=[col_sample_id, col_cell_type], wspace=0.4)
    return corr

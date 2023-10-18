#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import os
import re
import scanpy as sc
import pertpy as pt
import muon
from  warnings import warn
# import scipy
import seaborn
import matplotlib.pyplot as plt
import collections      
import scipy.sparse as sp_sparse
import h5py
import copy
from anndata import AnnData
import crispr as cr
import pandas as pd
import numpy as np
# from crispr.defaults import names_layers

regress_out_vars = ["total_counts", "pct_counts_mt"]


def get_layer_dict():
    """Retrieve layer name conventions."""
    lay =  {"preprocessing": "preprocessing", 
            "perturbation": "X_pert",
            "unnormalized": "unnormalized",
            "norm_log1p": "norm_log1p",
            "norm_z": "norm_z",
            "unscaled": "unscaled", 
            "unregressed": "unregressed",
            "counts": "counts"}
    return lay


def create_object(file, col_gene_symbols="gene_symbols", assay=None,
                  col_barcode=None, col_sample_id=None, 
                  kws_process_guide_rna=None, 
                  kws_concat=None, plot=True, **kwargs):
    """
    Create object from Scanpy- or Muon-compatible file(s).
    
    Provide as a dictionary (keyed by your desired 
    subject/sample names) consisting of whatever objects you would pass 
    to `create_object()`'s `file` argument if you want to concatenate
    multiple datasets. You must also specify col_sample (a string
    with the desired name of the sample/subject ID column). The 
    other arguments of this function can be specified as normal
    if they are common across samples; otherwise, specify them as 
    lists in the same order as the `file` dictionary.
    """
    if kws_concat is None:
        kws_concat = {}
    layers = cr.pp.get_layer_dict()  # standard layer names
    if col_sample_id is not None:  # concatenate multiple datasets
        print(f"\n<<< LOADING MULTIPLE FILEs {file} >>>")
        adatas = [None] * len(file)
        batch_categories = list(file.keys())  # keys = sample IDs
        file = [file[f] for f in file]  # turn to list 
        for f in range(len(file)):
            print(f"\n\n\t*** Creating object {f + 1} of {len(file)}")
            kpr = kws_process_guide_rna[f] if isinstance(
                kws_process_guide_rna, list) else kws_process_guide_rna
            asy = assay if isinstance(
                assay, str) or assay is None else assay[f]  # GEX/RNA assay 
            cgs = col_gene_symbols if isinstance(col_gene_symbols, str) or (
                col_gene_symbols is None) else col_gene_symbols[f]  # symbols
            adatas[f] = cr.pp.create_object(
                file[f], assay=asy, col_gene_symbols=cgs, col_sample_id=None,
                kws_process_guide_rna=kpr, plot=False, **kwargs)  # AnnData
        print(f"\n<<< CONCATENATING FILES {file} >>>")
        adata = AnnData.concatenate(
            *adatas, join="outer", batch_key=col_sample_id,
            batch_categories=batch_categories, **{
                **dict(uns_merge="same", index_unique="-", fill_value=None), 
                **kws_concat})  # concatenate AnnData objects
        kws_process_guide_rna = None  # don't perform again on concatenated
    elif isinstance(file, (str, os.PathLike)) and os.path.splitext(
        file)[1] == ".h5mu":  # MuData
        print(f"\n<<< LOADING FILE {file} with muon.read() >>>")
        adata = muon.read(file)
    elif isinstance(file, dict):  # metadata in protospacer files
        print(f"\n<<< LOADING PROTOSPACER METADATA >>>")
        adata = combine_matrix_protospacer(
            **file, col_gene_symbols=col_gene_symbols, 
            col_barcode=col_barcode, **kwargs)  # + metadata from protospacer
    elif not isinstance(file, (str, os.PathLike)):  # if already AnnData
        print(f"\n<<< LOADING OBJECT >>>")
        adata = file.copy()
    elif os.path.isdir(file):  # if directory, assume 10x format
        print(f"\n<<< LOADING 10X FILE {file} >>>")
        adata = sc.read_10x_mtx(
            file, var_names=col_gene_symbols, cache=True, **kwargs)
    elif os.path.splitext(file)[1] == ".h5":  # .h5 file
        print(f"\n<<< LOADING 10X .h5 FILE {file} >>>")
        print(f"H5 File Format ({file})\n\n")
        explore_h5_file(file, "\n\n\n")
        adata = sc.read_10x_h5(file, **kwargs)
    else:
        print(f"\n<<< LOADING FILE {file} with sc.read() >>>")
        adata = sc.read(file)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    cr.pp.print_counts(adata, title="Raw")
    if col_gene_symbols not in adata.var.columns:
        if assay: 
            adata[assay] = adata[assay].var.rename_axis(col_gene_symbols) 
        else:
            adata.var = adata.var.rename_axis(col_gene_symbols)
    if kws_process_guide_rna:  # process guide RNA
        if assay:
            adata[assay]  = cr.pp.process_guide_rna(
                adata[assay], **kws_process_guide_rna)
        else:
            adata = cr.pp.process_guide_rna(adata, **kws_process_guide_rna)
        cct = kws_process_guide_rna["col_cell_type"] if "col_cell_type" in (
            kws_process_guide_rna) else None
        cr.pp.print_counts(adata[assay] if assay else adata, 
                           title="Post-Guide RNA Processing", group_by=cct)
        # TODO: FIGURE
    if assay: 
        adata[assay].layers[layers["counts"]] = adata[assay].X.copy()
    else:
        adata.layers[layers["counts"]] = adata.X.copy()
    if plot is True:
        cr.pp.perform_qc(adata, hue=col_sample_id)  # calculate & plot QC
    print("\n\n", adata)
    return adata


def combine_matrix_protospacer(
    directory="", subdirectory_mtx="filtered_feature_bc_matrix", 
    col_gene_symbols="gene_symbols", 
    file_protospacer="crispr_analysis/protospacer_calls_per_cell.csv", 
    col_barcode="cell_barcode", 
    **kwargs):
    """
    Combine CellRanger directory-derived AnnData `.obs` & perturbation data.
    
    Example
    -------
    >>> data_dir = "/home/asline01/projects/crispr/examples/data"
    >>> adata = combine_matrix_protospacer(
    ... f"{data_dir}/crispr-screening/HH03", 
    ... "filtered_feature_bc_matrix", col_gene_symbols="gene_symbols", 
    ... file_protospacer="crispr_analysis/protospacer_calls_per_cell.csv", 
    ... col_barcode="cell_barcode")
    
    Or using create_object(), with directory/file-related arguments in 
    a dictionary passed to the "file" argument:
    
    >>> data_dir = "/home/asline01/projects/crispr/examples/data"
    >>> adata = create_object(
    ... dict(directory=f"{data_dir}/crispr-screening/HH03", 
    ... subdirectory_mtx="filtered_feature_bc_matrix", 
    ... file_protospacer="crispr_analysis/protospacer_calls_per_cell.csv"),
    ... col_barcode="cell_barcode", col_gene_symbols="gene_symbols")
    
    """
    adata = sc.read_10x_mtx(
        os.path.join(directory, subdirectory_mtx), 
        var_names=col_gene_symbols, **kwargs)  # 10x matrix, barcodes, features
    dff = pd.read_csv(os.path.join(directory, file_protospacer), 
                      index_col=col_barcode)  # perturbation information
    if col_barcode is None:
        dff, col_barcode = dff.set_index(dff.columns[0]), dff.columns[0]
    adata.obs = adata.obs.join(dff.rename_axis(adata.obs.index.names[0]))
    return adata

        
def print_counts(adata, group_by=None, title="Total", **kwargs):   
    if kwargs:
        pass
    print(f"\n\n{'=' * 80}\nCell Counts: {title}\n{'=' * 80}\n")
    if group_by is not None and group_by in adata.obs:
        print(adata.n_obs)
        for x in adata.layers:
            print(f"{x}: {adata.layers[x].shape}")
        if group_by is not None and group_by in adata.obs:
            print(adata.obs[group_by].value_counts())
        print("\n")
    print(f"\n\n{'=' * 80}\nGene Counts: {title}\n{'=' * 80}\n")
    
    
def process_data(adata, 
                 col_gene_symbols=None,
                 col_cell_type=None,
                 # remove_doublets=True,
                 outlier_mads=None,
                 cell_filter_pmt=5,
                 cell_filter_ncounts=None, 
                 cell_filter_ngene=None,
                 gene_filter_ncell=None,
                 target_sum=1e4,
                 normalization="log",
                 kws_hvg=True,
                 scale=10, kws_scale=None,
                 regress_out=regress_out_vars, 
                 **kwargs):
    """
    Perform various data processing steps.

    Args:
        adata (AnnData or MuData): The input data object.
        col_gene_symbols (str, optional): The name of the column or 
            index in `.var` containing gene symbols. Defaults to None.
        col_cell_type (str, optional): The name of the column 
            in `.obs` containing cell types. Defaults to None.
        normalization (str or dict, optional): If "log,"
            perform conventional normalization steps 
            (log-transform the data after total-count normalization). 
            If "z", perform z-normalization. If a dictionary,
            include the method ("log" or "z") under the key "method,"
            If not None and `method="z"`, should contain 
            the name of the column in `.obs` containing batch IDs 
            (e.g., orig.ident) under the key "col_batch" (which should 
            be None if z-normalization is just to be applied 
            uniformly without respect to batch).
            If `method="z"`, should contain keys for
            "col_reference" and "key_reference" containing the 
            column name where perturbed versus unperturbed labels are
            stored and the label in that column that corresponds to
            the control condition, respectively. Defaults to "log".
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
        kws_hvg (dict or bool, optional): The keyword arguments for 
            detecting variable genes or True for default arguments. 
            To filter by HVGs, include an 
            extra argument in this dictionary: {"filter": True}.
            Otherwise, a highly_variable genes column is created and
            can be used in clustering, but non-HVGs will be retained in
            the data for other uses. Defaults to True.
        scale (int or bool, optional): The scaling factor or True 
            for no clipping. Defaults to 10.
        kws_scale (dict, optional): The keyword arguments for 
            scaling. Defaults to None.
        regress_out (list or None, optional): The variables to 
            regress out. Defaults to regress_out_vars.
        **kwargs: Additional keyword arguments.

    Returns:
        adata (AnnData): The processed data object.
        figs (dict): A dictionary of generated figures.
    """
    # Setup Object
    if kwargs:
        print(f"\nUn-Used Keyword Arguments: {kwargs}\n\n")
    layers = cr.pp.get_layer_dict()  # layer names
    ann = adata.copy()  # copy so passed AnnData object not altered inplace
    ann.raw = ann.copy()  # original in `.raw`
    ann.layers[layers["preprocessing"]] = ann.X.copy() # set original in layer
    ann.obs["n_counts"] = ann.X.sum(1)
    ann.obs["log_counts"] = np.log(ann.obs["n_counts"])
    ann.obs["n_genes"] = (ann.X > 0).sum(1)
    print(ann)
    
    # Initial Information/Arguments
    if col_gene_symbols == ann.var.index.names[0]:  # if symbols=index...
        col_gene_symbols = None  # ...so functions will refer to index name
    figs = {}
    if kws_scale is None:
        kws_scale = {}
    kws_scale, kws_hvg = [x if x else {} for x in [kws_scale, kws_hvg]]
    filter_hvgs = kws_hvg.pop("filter") if "filter" in kws_hvg else False
    n_top = kwargs.pop("n_top") if "n_top" in kwargs else 4000
    if isinstance(normalization, str):  # if provided as string...
        normalization = dict(method=normalization)  # ...to method argument
    sid = normalization["col_batch"] if "col_batch" in normalization else None
        
    # Set Up Layer & Variables
    if outlier_mads is not None:  # if filtering based on calculating outliers 
        if isinstance(outlier_mads, (int, float)):  # same MADs, all metrics
            qc_mets = ["log1p_total_counts", "log1p_n_genes_by_counts",
                       "pct_counts_in_top_20_genes"]
            outlier_mads = dict(zip(qc_mets, [outlier_mads] * len(qc_mets)))
    cr.pp.print_counts(ann, title="Initial", group_by=col_cell_type)
    print(col_gene_symbols, "\n\n", n_top, "\n\n", ann.var.describe(), "\n\n")

    # Basic Filtering (DO FIRST...ALL INPLACE)
    print("\n<<< FILTERING CELLS (TOO FEW GENES) & GENES (TOO FEW CELLS) >>>") 
    sc.pp.filter_cells(ann, min_genes=cell_filter_ngene[0])
    sc.pp.filter_genes(ann, min_cells=gene_filter_ncell[0])
    cr.pp.print_counts(ann, title="Post-Basic Filter", group_by=col_cell_type)
    
    # Exploration & QC Metrics
    print("\n<<< PERFORMING QUALITY CONTROL ANALYSIS>>>")
    figs["qc_metrics"] = cr.pp.perform_qc(
        ann, n_top=n_top, col_gene_symbols=col_gene_symbols,
        hue=sid)  # QC metric calculation & plottomg
    
    # Further Filtering
    print("\n<<< FURTHER CELL & GENE FILTERING >>>")
    ann = cr.pp.filter_qc(ann, outlier_mads=outlier_mads,
                          cell_filter_pmt=cell_filter_pmt,
                          cell_filter_ncounts=cell_filter_ncounts,
                          cell_filter_ngene=cell_filter_ngene, 
                          gene_filter_ncell=gene_filter_ncell)
    cr.pp.print_counts(ann, title="Post-Filter", group_by=col_cell_type)
    
    # Doublets
    # doublets = detect_doublets(adata)
    # if remove_doublets is True:
    #     adata
    #     # TODO: doublets
    
    # Normalization
    ann = cr.pp.normalize(ann, target_sum=target_sum, **normalization)
        
    # Gene Variability (Detection, Optional Filtering)
    print("\n<<< DETECTING VARIABLE GENES >>>")
    sc.pp.highly_variable_genes(ann, layer=layers["norm_log1p"], **kws_hvg)
    if filter_hvgs is True:
        print("\n<<< FILTERING BY HIGHLY VARIABLE GENES >>>")
        ann = ann[:, ann.var.highly_variable]  # filter by HVGs
        cr.pp.print_counts(ann, title="HVGs", group_by=col_cell_type)
    
    # Regress Out Confounds
    if regress_out: 
        print("\n<<< REGRESSING OUT CONFOUNDS >>>")
        ann.layers[layers["unregressed"]] = ann.X.copy()
        sc.pp.regress_out(ann, regress_out, copy=False)
        warn("Confound regression doesn't yet properly use layers.")
    
    # Scale Gene Expression
    if scale is not None:
        print("\n<<< SCALING >>>")
        ann.layers[layers["unscaled"]] = ann.X.copy()
        if scale is not True:  # if scale = int; also clip values/"scale" SDs
            kws_scale.update(dict(max_value=scale))
        sc.pp.scale(ann, copy=False, layer=layers["norm_log1p"], 
                    **kws_scale)  # scale GEX INPLACE log1p layer
    
    # Store Final Object
    ann.X = ann.layers[layers[norm]].copy()  # norm -> .X
    ann.raw = ann.copy()
            
    # Final Data Examination
    cr.pp.print_counts(ann, title="Post-Processing", group_by=col_cell_type)
    figs["qc_metrics_post"] = cr.pp.perform_qc(
        ann, n_top=n_top, col_gene_symbols=col_gene_symbols,
        hue=sid)  # QC metric calculation & plottomg
    return ann, figs


def normalize(adata, method="log", target_sum=1e4, kws_z=None):
    """Create normalization in adata layers."""
    layers = cr.pp.get_layer_dict()
    adata.layers[layers["unnormalized"]] = adata.X.copy()
    if target_sum is not None:  # total-count normalization INPLACE
        print("\n<<< TOTAL-COUNT-NORMALIZING >>>")
        adata.layers[layers["norm_total_counts"]] = sc.pp.normalize_total(
            adata, target_sum=target_sum)  # total-count normalize
    else:
        print("\n<<< ***NOT*** TOTAL-COUNT-NORMALIZING >>>")
    print(f"\n<<< LOG-NORMALIZING => layer {layers['norm_log1p']} >>>")
    adata.layers[layers["norm_log1p"]] = sc.pp.log1p(
        adata, copy=True)  # log-transformed for later use; NOT INPLACE
    if kws_z:  # if want to perform z-normalization
        if ~all((x in kws_z for x in ["col_reference", "key_reference"])):
            raise ValueError(
                "'col_reference' and 'key_reference' must be "
                "in `normalization` argument if method = 'z'.")
        adata.layers[layers["norm_z"]] = cr.pp.z_normalize_by_reference(
            adata, **kws_z)  # z-normalize to controls
    return adata


def z_normalize_by_reference(adata, col_reference, key_reference="Control", 
                             col_batch=None, retain_zero_variance=True, 
                             layer=None, **kwargs):
    """
    Mean-center & standardize by 
    reference condition, optionally within-batches.
    
    If `retain_zero_variance` is True, then genes with zero variance 
    are retained.
    """
    if kwargs:
        print(f"\nUn-Used Keyword Arguments: {kwargs}\n\n")
    adata = adata.copy()
    if layer:
        adata.X = adata.layer[layer].copy()
    if col_batch is None:
        col_batch, col_batch_origin = "batch", None
        if "batch" == col_reference:
            raise ValueError(
                f"col_reference cannot be {col_batch} when col_batch=None.")
        adata.obs.loc[:, col_batch] = "1"  # so can loop even if only 1 batch
    else:
        col_batch_origin = col_batch
    batches_adata, batch_labs = [], adata.obs[col_batch].unique()
    for s in batch_labs:  # loop over batches
        batch_adata = adata[adata.obs[col_batch] == s].copy()
        gex = batch_adata.X.copy()  #  gene expression matrix (full)
        gex_ctrl = batch_adata[batch_adata.obs[
            col_reference] == key_reference].X.A.copy()  # reference condition
        gex, gex_ctrl = [q.A if "A" in dir(q) else q 
                         for q in [gex, gex_ctrl]]  # sparse -> dense matrix
        mus, sds = np.nanmean(gex_ctrl, axis=0), np.nanstd(
            gex_ctrl, axis=0)  # means & SDs of reference condition genes
        if retain_zero_variance is True:
            sds[sds == 0] = 1   # retain zero-variance genes at unit variance
        batch_adata.X = (gex - mus) / sds  # z-score gene expression
        batches_adata += [batch_adata]  # concatenate batch adata
    if col_batch_origin is None:  # in case just 1 batch (no concatenation)
        adata = batches_adata[0]
        adata.obs.drop(col_batch, axis=1, inplace=True)
    else:  # concatenate batches
        adata = AnnData.concatenate(
            *batches_adata, join="outer", batch_key=col_batch, 
            uns_merge="same", index_unique="-", 
            batch_categories=batch_labs, fill_value=None)  # concatenate
    return adata


def perform_qc(adata, n_top=20, col_gene_symbols=None, hue=None):
    """Calculate & plot quality control metrics."""
    figs = {}
    figs["highly_expressed_genes"] = sc.pl.highest_expr_genes(
        adata, n_top=n_top, gene_symbols=col_gene_symbols)  # high GEX genes
    patterns = dict(zip(["mt", "ribo", "hb"], 
                        [("MT-", "mt-"), ("RPS", "RPL"), ("^HB[^(P)]")]))
    patterns_names = dict(zip(patterns, [
        "Mitochondrial", "Ribosomal", "Hemoglobin"]))
    p_names = [patterns_names[k] for k in patterns_names]
    print(f"\n\t*** Detecting {', '.join(p_names)} genes...") 
    for k in patterns:
        try:
            adata.var[k] = adata.var_names.str.startswith(patterns[k])
        except Exception as err:
            warn(f"\n\n{'=' * 80}\n\nCouldn't assign {k}: {err}")
    qc_vars = list(set(patterns.keys()).intersection(
        adata.var.keys()))  # available QC metrics 
    pct_ns = [f"pct_counts_{k}" for k in qc_vars]
    print("\n\t*** Calculating & plotting QC metrics...\n\n") 
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, percent_top=None, 
                               log1p=True, inplace=True)  # QC metrics
    rrs, ccs = cr.pl.square_grid(len(pct_ns + ["n_genes_by_counts"]))  # dims
    fff, axs = plt.subplots(rrs, ccs, figsize=(5 * rrs, 5 * ccs))  # subplots
    for a, v in zip(axs.flat, pct_ns + ["n_genes_by_counts"]):
        try:  # unravel axes to get coordinates, then scatterplot facet
            sc.pl.scatter(adata, x="total_counts", y=v, 
                          ax=a, color=hue, show=False)  # scatterplot
        except Exception as err:
            print(err)
    plt.show()
    figs[f"qc_{v}_scatter"] = fff
    try:
        varm = pct_ns if hue else pct_ns
        varm += ["n_genes_by_counts"]
        figs["pairplot"] = seaborn.pairplot(
            adata.obs[varm].rename_axis("Metric", axis=1).rename({
                "total_counts": "Total Counts", **patterns_names}, axis=1), 
            diag_kind="kde", diag_kws=dict(fill=True, cut=0))  # pairplot
    except Exception as err:
        figs["pairplot"] = err
        print(err)
    try:
        vark = pct_ns + [hue] if hue is None else pct_ns
        figs["pct_counts_kde"] = seaborn.displot(
            adata.obs[vark].rename_axis("Metric", axis=1).rename(
                patterns_names, axis=1).stack().to_frame("Percent Counts"), 
            x="Percent Counts", col="Metric", 
            kind="kde", hue=hue, cut=0, fill=True)  # KDE of pct_counts
    except Exception as err:
        figs["pct_counts_kde"] = err
        print(err)
    try:
        figs["metrics_violin"] = sc.pl.violin(
            adata, ["n_genes_by_counts", "total_counts"] + pct_ns,
            jitter=0.4, multi_panel=True)  # violin of counts, genes
    except Exception as err:
        figs["qc_metrics_violin"] = err
        print(err)
    try:
        figs["qc_log"] = seaborn.jointplot(
            data=adata.obs, x="log1p_total_counts", 
            y="log1p_n_genes_by_counts", kind="hex")  # jointplot
    except Exception as err:
        figs["qc_log"] = err
        print(err)
    print(adata.var.describe())
    return figs


def filter_qc(adata, outlier_mads=None, 
              cell_filter_pmt=None,
              cell_filter_ncounts=None,
              cell_filter_ngene=None, 
              gene_filter_ncell=None):
    """Filter low-quality/outlier cells & genes."""
    adata = adata.copy()
    if cell_filter_pmt is None:
        cell_filter_pmt = [0, 
                           100]  # so doesn't filter MT but calculates metrics
    min_pct_mt, max_pct_mt = cell_filter_pmt
    if outlier_mads is not None:  # automatic filtering using outlier stats
        outliers = adata.obs[outlier_mads.keys()]
        print(f"\n<<< DETECTING OUTLIERS {outliers.columns} >>>") 
        for x in outlier_mads:
            outliers.loc[:, f"outlier_{x}"] = cr.tl.is_outlier(
                adata.obs, outlier_mads[x])  # separate metric outlier columns
        cols_outlier = list(set(
            outliers.columns.difference(adata.obs.columns)))
        outliers.loc[:, "outlier"] = outliers[cols_outlier].any()  # binary
        print(f"\n<<< FILTERING OUTLIERS ({cols_outlier}) >>>") 
        adata.obs = adata.obs.join(outliers[["outlier"]])  # + outlier column
        print(f"Total Cell Count: {adata.n_obs}")
        adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)
                      ].copy()  # drop outliers
        print(f"Post-Filtering Cell Count: {adata.n_obs}")
    else:  # manual filtering
        print("\n<<< PERFORING THRESHOLD-BASED FILTERING >>>") 
        print(f"\nTotal Cell Count: {adata.n_obs}")
        print("\n\t*** Filtering cells by mitochondrial gene percentage...") 
        print(f"\n\tMinimum: {min_pct_mt}\n\tMaximum: {max_pct_mt}")
        adata = adata[(adata.obs.pct_counts_mt <= max_pct_mt) * (
            adata.obs.pct_counts_mt >= min_pct_mt)]  # filter by MT %
        print(f"\tNew Count: {adata.n_obs}")
        print("\n\t*** Filtering genes based on # of genes expressed...")
        if cell_filter_ngene[0] is not None:
            print(f"\n\tMinimum: {cell_filter_ngene[0]}...")
            sc.pp.filter_cells(adata, min_genes=cell_filter_ngene[0])
            print(f"\tNew Count: {adata.n_obs}")
        else:
            print("\n\tNo minimum")
        if cell_filter_ngene[1] is not None:
            print(f"\tMaximum: {cell_filter_ngene[1]}")
            sc.pp.filter_cells(adata, max_genes=cell_filter_ngene[1])
            print(f"\tNew Count: {adata.n_obs}")
        print("\n\t*** Filtering cells based on # of reads...")
        if cell_filter_ncounts[0] is not None:
            print(f"\n\tMinimum: {cell_filter_ncounts[0]}")
            sc.pp.filter_cells(adata, min_counts=cell_filter_ncounts[0])
            print(f"\tNew Count: {adata.n_obs}")
        else:
            print("\n\tNo minimum")
        if cell_filter_ncounts[1] is not None:
            print(f"\n\tMaximum: {cell_filter_ncounts[1]}")
            sc.pp.filter_cells(adata,
                # min_genes=None, max_genes=None, min_counts=None,
                max_counts=cell_filter_ncounts[1])
            print(f"\tNew Count: {adata.n_obs}")
        else:
            print("\n\tNo maximum")
        print("\n\t*** Filtering genes based on # of cells in which they "
              "are expressed...")
        if gene_filter_ncell[0] is not None:
            print(f"\n\tMinimum: {gene_filter_ncell[0]}")
            sc.pp.filter_genes(adata, min_cells=gene_filter_ncell[0])
            print(f"\tNew Count: {adata.n_obs}")
        else:
            print("\n\tNo minimum")
        if gene_filter_ncell[1] is not None:
            print(f"\n\tMaximum: {gene_filter_ncell[1]}")
            sc.pp.filter_genes(adata, max_cells=gene_filter_ncell[1])
            print(f"\tNew Count: {adata.n_obs}")
        else:
            print("\n\tNo maximum")
        print(f"\nPost-Filtering Cell Count: {adata.n_obs}")
    return adata


def explore_h5_file(file):
    """Explore an H5 file's format (thanks to ChatGPT)."""
    with h5py.File(file, "r") as h5_file:
        top_level_groups = list(h5_file.keys())
        for group_name in top_level_groups:
            print(f"Group: {group_name}")
            for g in h5_file[group_name]:
                print(f"  Dataset: {g}")


def remove_batch_effects(adata, col_cell_type="leiden", 
                         col_batch="orig.ident", plot=True, **kws_train):
    """Remove batch effects (IN PROGRESS)."""
    if not kws_train:
        kws_train = dict(max_epochs=100, batch_size=32, 
                         early_stopping=True, early_stopping_patience=25)
    train = adata.copy()
    train.obs["cell_type"] = train.obs[col_cell_type].tolist()
    train.obs["batch"] = train.obs[col_batch].tolist()
    if plot is True:
        sc.pl.umap(train, color=[col_batch, col_cell_type], 
                   wspace=.5, frameon=False)
    print(f"\n<<< PREPARING DATA >>>") 
    pt.tl.SCGEN.setup_anndata(train, batch_key="batch", 
                              labels_key="cell_type")  # prepare AnnData
    model = pt.tl.SCGEN(train)
    print(f"\n<<< TRAINING >>>") 
    model.train(**kws_train)  # training
    print(f"\n<<< CORRECTING FOR BATCH EFFECTS >>>") 
    corrected_adata = model.batch_removal()  # batch correction
    if plot is True:
        sc.pp.neighbors(corrected_adata)
        sc.tl.umap(corrected_adata)
        sc.pl.umap(corrected_adata, color=[col_batch, col_cell_type], 
                   wspace=0.4, frameon=False)
    return corrected_adata


def remove_guide_counts_from_gex_matrix(adata, col_target_genes, 
                                        key_ignore=None):
    """Remove guide RNA counts from gene expression matrix."""
    guides = list(adata.obs[col_target_genes].dropna())  # guide names
    if key_ignore is not None:
        guides = list(set(guides).difference(
            [key_ignore] if isinstance(key_ignore, str) else key_ignore))
    guides_in_varnames = list(set(adata.var_names).intersection(set(guides)))
    if len(guides_in_varnames) > 0:
        print(f"\n\t*** Removing {', '.join(guides)} guides "
            "from gene expression matrix...")
        adata._inplace_subset_var(list(set(adata.var_names).difference(
            set(guides_in_varnames))))  # remove guide RNA counts
    return adata


def detect_guide_targets(col_guide_rna_series,
                         feature_split="|", guide_split="-",
                         key_control_patterns=None,
                         key_control="Control", **kwargs):
    """Detect guide gene targets (see `filter_by_guide_counts`)."""
    if kwargs:
        print(f"\nUn-Used Keyword Arguments: {kwargs}\n\n")
    if key_control_patterns is None:
        key_control_patterns = [
            key_control]  # if already converted, pattern=key itself
    if isinstance(key_control_patterns, str):
        key_control_patterns = [key_control_patterns]
    targets = col_guide_rna_series.str.strip(" ").replace("", np.nan)
    if key_control_patterns and pd.Series(
        key_control_patterns).isnull().any():  # if NAs = control sgRNAs
        targets = targets.replace(np.nan, key_control)  # NaNs -> control key
        key_control_patterns = list(pd.Series(key_control_patterns).dropna())
    else:  # if NaNs mean unperturbed cells
        if any(pd.isnull(targets)):
            warn(f"Dropping rows with NaNs in `col_guide_rna`.")
        targets = targets.dropna()
    if feature_split is not None or guide_split is not None:
        targets, nums = [targets.apply(
            lambda x: [re.sub(p, ["", r"\1"][j], str(i)) if re.search(
                p, str(i)) else [i, ""][j] for i in list(
                    x.split(feature_split) if feature_split  # if multi
                    else [x])]  # if single gRNA
            if p not in ["", None] else  # ^ if need to remove guide suffixes
            list(x.split(feature_split) if feature_split else [x] if p else p
                )  # ^ no suffixes: split x or [x] (j=0), "" for suffix (j=1)
            ) for j, p in enumerate(list(
                [f"{guide_split}.*", rf'^.*?{re.escape(guide_split)}(.*)$']
            if guide_split else [None, ""]))
                            ]  # each entry -> list of target genes
    if key_control_patterns:  # if need to search for control key patterns
        targets = targets.apply(
            lambda x: [i if i == key_control else key_control if any(
                    (k in i for k in key_control_patterns)) else i 
                for i in x])  # find control keys among targets
    # targets = targets.apply(
    #     lambda x: [[x[0]] if len(x) == 2 and x[1] == "" else x
    #     for i in x])  # in case all single-transfected
    grnas = targets.to_frame("t").join(nums.to_frame("n")).apply(
        lambda x: [i + str(guide_split if guide_split else "") + "_".join(
            np.array(x["n"])[np.where(np.array(x["t"]) == i)[0]]) 
                   for i in pd.unique(x["t"])],  # sum gRNA counts/gene target 
        axis=1).apply(lambda x: feature_split.join(x)).to_frame(
            "ID")  # e.g., STAT1-1|STAT1-2|NT-1-2 => STAT1-1_2 counts
    # DO NOT change the name of grnas["ID"]
    return targets, grnas


def filter_by_guide_counts(adata, col_guide_rna, col_num_umis, 
                           max_pct_control_drop=75,
                           min_n_target_control_drop=100,
                           min_pct_avg_n=40,
                           min_pct_dominant=80,
                           drop_multi_control=True,
                           feature_split="|", guide_split="-",
                           key_control_patterns=None,
                           key_control="Control", **kwargs):
    """
    Process sgRNA names (e.g., multi-probe names).

    Args:
        adata (AnnData): AnnData object (RNA assay, 
            so if multi-modal, subset before passing to this argument).
        col_guide_rna (str): _description_
        col_num_umis (str): Column with the UMI counts.
        max_pct_control_drop (int, optional): If control 
            UMI counts are less than or equal to this percentage of the 
            total counts for that cell, and if a non-control sgRNA is 
            also present and meets other filtering criteria, then 
            consider that cell pseudo-single-transfected 
            (non-control gene). Note that controls in 
            multiply-transfected cells will also be ultimately dropped
            if `drop_multi_control` is True.
            Dropping with this criterion means cells with only control
            guides will be completely dropped if not meeting criteria.
            Set to 0 to ignore this filtering. Defaults to 75.
        min_n_target_control_drop (int, optional): If UMI counts
            across target (non-control) guides are above this number,
            notwithstanding whether control guide percent exceeds
            `max_percent_umis_control_drop`, drop control from that 
            cell. For instance, if 100 and 
            `max_percent_umis_control_drop=75`, even if a cell's 
            UMIs are 75% control guides, if there are at least 100 
            non-control guides, consider the cell only transfected for 
            the non-control guides. Set to None to ignore this 
            filtering criterion. Note that controls in 
            multiply-transfected cells will also be ultimately dropped
            if `drop_multi_control` is True.
            Dropping with this criterion means cells with only control
            guides will be completely dropped if not meeting criteria.
            Defaults to 100.
        drop_multi_control (bool, optional): If True, drop control
            guides from cells that, after all other filtering,
            are multiply-transfected. Defaults to True.
        min_pct_avg_n (int, optional): sgRNAs with counts below this 
            percentage of the average UMI count will be considered 
            noise and dropped from the list of genes for which
            that cell is considered transfected. Defaults to 40.
        min_pct_dominant (int, optional): sgRNAs with counts at or 
            above this percentage of the cell total UMI count will be 
            considered dominant, and all other guides will be dropped 
            from the list of genes for whichmthat cell is considered 
            transfected. Defaults to 80.
        feature_split (str, optional): For designs with multiple 
            guides, the character that splits guide names in 
            `col_guide_rna`. For instance, "|" for 
            `STAT1-1|CNTRL-1|CDKN1A`. Defaults to "|".
            If only single guides, set to None.
        guide_split (str, optional): The character that separates 
            guide (rather than gene target)-specific IDs within gene. 
            For instance, guides targeting STAT1 may include 
            STAT1-1, STAT1-2, etc.; the argument would be "-" 
            so the function can identify all of those as 
            targeting STAT1. Defaults to "-".
        key_control_patterns (list, optional): List (or single string) 
            of patterns in guide RNA column entries that correspond to 
            a control. For instance, if control entries in the original 
            `col_guide_rna` column include `NEGCNTRL` and
            `Control.D`, you should specify ['Control', 'CNTRL'] 
            (assuming no non-control sgRNA names contain 
            those patterns). If blank entries should be interpreted as 
            control guides, then include np.nan/numpy.nan in this list.
            Defaults to None -> [np.nan].
        key_control (str, optional): The name you want the control 
            entries to be categorized as under the new `col_guide_rna`. 
            for instance, `CNTRL-1`, `NEGCNTRL`, etc. would all be 
            replaced by "Control" if that's what you specify here. 
            Defaults to "Control".

    Returns:
        pandas.DataFrame: A dataframe (a) with sgRNA names replaced 
            under their target gene categories (or control) and
            (b) with `col_guide_rna` and `col_num_umis` column entries
            (strings) grouped into lists 
            (new columns with suffix "list_all"). 
            Note that the UMI counts are summed across sgRNAs 
            targeting the same gene within a cell. Also versions of the 
            columns (and the corresponding string versions) 
            filtered by the specified  
            criteria (with suffixes "_filtered" and "_list_filtered" 
            for list versions).
            
    Notes:
        FUTURE DEVELOPERS: The Crispr class object initialization 
        depends on names of the columns created in this function. 
        If they are changed (which should be avoided), be sure to 
        change throughout the package.
    """
    # Extract Guide RNA Information
    ann = adata.copy()
    if guide_split is None:
        guide_split = "$"
    if key_control_patterns is None:
        key_control_patterns = [np.nan]
    guides = ann.obs[col_guide_rna].copy()  # guide names
    
    # If `guide_split` in Any Gene Names, Temporarily Substitute
    grs = None
    if guide_split is not None:
        split_char = [guide_split in g for g in ann.var_names]
        if any(split_char):
            grs = "==="
            bad_symb = np.array(ann.var_names)[np.where(split_char)[0]]
            if grs in guide_split:
                raise ValueError(f"{grs} is a reserved name and cannot be "
                                 "contained within `guide_split`.")
            warn(f"`guide_split` ({guide_split}) found in at least "
                 f"one gene name ({', '.join(bad_symb)}). Using {grs}. "
                 "as temporary substitute. Will attempt to replace later, "
                 "but note that there are risks in having a `guide_split` "
                 "as a character also found in gene names.")
            guides = guides.apply(lambda x: re.sub(bad_symb[np.where(
                [i in str(x) for i in bad_symb])[0][0]], re.sub(
                guide_split, grs, bad_symb[np.where(
                    [i in str(x) for i in bad_symb])[0][0]]), 
                str(x)) if any((i in str(x) for i in bad_symb)) else x)
    
    # Find Gene Targets & Counts of Guides
    targets, grnas = cr.pp.detect_guide_targets(
        guides, feature_split=feature_split, guide_split=guide_split,
        key_control_patterns=key_control_patterns, 
        key_control=key_control, **kwargs)  # target genes
    if grs is not None:  # if guide_split was in any gene name
        targets = targets.apply(lambda x: [
            re.sub(grs, guide_split, i) for i in x])  # replace grs in list
        grnas.loc[:, "ID"] = grnas["ID"].apply(
            lambda x: re.sub(grs, guide_split, str(x)))  # e.g., back to HLA-B
    tg_info = grnas["ID"].to_frame(
        col_guide_rna + "_flat_ix").join(
            targets.to_frame(col_guide_rna + "_list"))
    if col_num_umis is not None:
        tg_info = tg_info.join(ann.obs[[col_num_umis]].apply(
            lambda x: [float(i) for i in list(
                str(x[col_num_umis]).split(feature_split)
                if feature_split else [float(x[col_num_umis])])], 
            axis=1).to_frame(col_num_umis + "_list"))
        tg_info = tg_info.join(tg_info[col_num_umis + "_list"].dropna().apply(
            sum).to_frame(col_num_umis + "_total"))  # total UMIs/cell
    tg_info = ann.obs[col_guide_rna].to_frame(col_guide_rna).join(tg_info)
    if tg_info[col_guide_rna].isnull().any() and (~any(
        [pd.isnull(x) for x in key_control_patterns])):
        warn(f"NaNs present in guide RNA column ({col_guide_rna}). "
             f"Dropping {tg_info[col_guide_rna].isnull().sum()} "
             f"out of {tg_info.shape[0]} rows.")
        tg_info = tg_info[~tg_info[col_guide_rna].isnull()]

    # Sum Up gRNA UMIs
    cols = [col_guide_rna + "_list", col_num_umis + "_list"]
    feats_n = tg_info[cols].dropna().apply(lambda x: pd.Series(
        dict(zip(pd.unique(x[cols[0]]), [sum(np.array(x[cols[1]])[
            np.where(np.array(x[cols[0]]) == i)[0]]) for i in pd.unique(
                x[cols[0]])]))), axis=1).stack().rename_axis(["bc", "g"])
    feats_n = feats_n.to_frame("n").join(feats_n.groupby(
        "bc").sum().to_frame("t"))  # sum w/i-cell # gRNAs w/ same target gene 
    feats_n = feats_n.assign(p=feats_n.n / feats_n.t * 100)  # to %age
    
    # Other Variables
    feats_n = feats_n.join(feats_n.reset_index("g").groupby("bc").apply(
            lambda x: 0 if all(x["g"] == key_control) else x[
                x["g"] != key_control]["n"].sum()
            ).to_frame("n_non_ctrl"))  # overridden by dominant guide?
    feats_n = feats_n.join(feats_n.groupby(["bc", "g"]).apply(
        lambda x: "retain" if (x.name[1] != key_control) 
        else ("low_control" if float(x["p"]) <= max_pct_control_drop 
              else "high_noncontrol" if min_n_target_control_drop and float(
                  x["n_non_ctrl"]) >= min_n_target_control_drop
              else "retain")).to_frame("drop_control"))  # control drop labels
    feats_n = feats_n.join(feats_n.n.groupby("bc").mean().to_frame(
        "n_cell_avg"))  # average UMI count within-cell
    feats_n = feats_n.reset_index("g")
    feats_n = feats_n.assign(control=feats_n.g == key_control
                             )  # control guide dummy-coded column
    feats_n = feats_n.assign(target=feats_n.g != key_control).set_index(
        "g", append=True)  # target guide dummy-coded column
    if min_pct_dominant is not None:
        feats_n = feats_n.assign(dominant=feats_n.p >= min_pct_dominant)
        feats_n = feats_n.assign(
            dominant=feats_n.dominant & feats_n.target
            )  # only non-control guides considered dominant
    else:
        feats_n = feats_n.assign(dominant=False)  # no filtering based on this
    # CHECK: 
    # feats_n[feats_n.p >= min_pct_dominant].dominant.all()
    feats_n = feats_n.assign(
        low_umi=feats_n.n < feats_n.n_cell_avg * min_pct_avg_n / 100 
        if min_pct_avg_n is not None else False
        )  # low % of mean UMI count (if filtering based on that)?
    feats_n = feats_n.join(feats_n.dominant.groupby("bc").apply(
            lambda x: pd.Series(
                [True if (x.any()) and (i is False) else False for i in x], 
                index=x.reset_index("bc", drop=True).index)
            ).to_frame("drop_nondominant"))  # overridden by dominant guide?
    # CHECK: 
    # feats_n.loc[feats_n[(feats_n.p < min_pct_dominant) & (
    #     feats_n.drop_nondominant)].reset_index().bc.unique()].groupby(
    #         "bc").apply(lambda x: x.dominant.any()).all()
    # feats_n[(feats_n.p >= 80)].drop_nondominant.sum() == 0
    filt = feats_n.copy()  # start with full feats_n

    # Filtering Phase I (If Dominant Guide, Drop Others; Drop Low Controls)
    filt = filt[filt.drop_control.isin(["retain"])
                ]  # low control or high non-control = not control-transfected 
    filt = filt[~filt.drop_nondominant]  # drop if another guide dominates
    
    # Filtering Phase II (Filter Low Targeting Guides in Multiply-Transfected)
    filt = filt.join(filt.reset_index("g").g.groupby("bc").apply(
        lambda x: len(x[x != key_control].unique()) > 1).to_frame(
            "multi_noncontrol"))  # multi-transfected with non-control guides?
    filt = filt.assign(low_umi_multi=filt.low_umi & filt.multi_noncontrol)
    filt = filt[~filt.low_umi_multi]  # drop low guides, multi-NC-transfected
    
    # Filtering Phase III (Remove Control from Multi-Transfected)
    filt = filt.join(filt.reset_index("g").g.groupby("bc").apply(
        lambda x: "multi" if len(x.unique()) > 1 else "single").to_frame(
            "transfection"))  # after filtering, single or multi-guide?
    if drop_multi_control is True:  # drop control (multi-transfected cells)
        filt = filt.assign(multi=filt.transfection == "multi")
        filt = filt.assign(multi_control=filt.control & filt.multi)
        filt = filt[~filt.multi_control]  # drop
        filt = filt.drop(["multi", "multi_control"], axis=1)
    filt = filt.drop("transfection", axis=1)
    filt = filt.join(filt.reset_index("g").g.groupby("bc").apply(
        lambda x: "multi" if len(x.unique()) > 1 else "single").to_frame(
            "transfection"))  # after filtering, single or multi-guide?
    
    # Join Counts/%s/Filtering Categories w/ AnnData-Indexed Guide Information
    filt = filt.n.to_frame("u").groupby("bc").apply(
        lambda x: pd.Series({cols[0]: list(x.reset_index("g")["g"]), 
                            cols[1]: list(x.reset_index("g")["u"])}))
    tg_info = tg_info.join(filt, lsuffix="_all", rsuffix="_filtered")  # join
    tg_info = tg_info.dropna().loc[ann.obs.index.intersection(
        tg_info.dropna().index)]  # re-order according to adata index
    
    # Re-Make String Versions of New Columns with List Entries
    for q in [col_guide_rna, col_num_umis]:  # string versions of list entries 
        tg_info.loc[:, q + "_filtered"] = tg_info[q + "_list_filtered"].apply( 
            lambda x: x if not isinstance(x, list) else feature_split.join(
                str(i) for i in x))  # join processed names by `feature_split`
        
    # DON'T CHANGE THESE!
    rnd = {"g": "Gene", "t": "Total Guides in Cell", 
           "p": "Percent of Cell Guides", "n": "Number in Cell"}
    # Crispr.get_guide_counts() depends on the names in "rnd"
    
    feats_n = feats_n.reset_index().rename(rnd, axis=1).set_index(
        [feats_n.index.names[0], rnd["g"]])
    tg_info = tg_info.assign(feature_split=feature_split)
    return tg_info, feats_n

def process_guide_rna(adata, col_guide_rna="guide_id", 
                      col_guide_rna_new="condition", 
                      col_num_umis="UMI count",
                      key_control="NT", 
                      conserve_memory=False,
                      remove_multi_transfected=False,
                      **kws_process_guide_rna):
    """Process guide RNA & add results to `.uns` (NOT INPLACE)."""
    print("\n\n<<<PERFORMING gRNA PROCESSING AND FILTERING>>>\n")
    ann = copy.deepcopy(adata)
    kws_pga = copy.deepcopy(kws_process_guide_rna)
    
    # Filter by Guide Counts
    tg_info, feats_n = cr.pp.filter_by_guide_counts(
        ann, col_guide_rna, col_num_umis=col_num_umis,
        key_control=key_control, **kws_pga
        )  # process (e.g., multi-probe names) & filter by # gRNA
    
    # Add Results to AnnData
    if "feature_split" not in kws_pga:
        kws_pga["feature_split"] = None
    try:
        tg_info = tg_info.loc[ann.obs.index]
    except Exception as err:
        warn(f"{err}\n\nCouldn't re-order tg_info to mirror adata index!")
    tg_info_all = None  # fill later if needed
    if remove_multi_transfected is True:  # remove multi-transfected
        tg_info_all = tg_info.copy() if conserve_memory is False else None
        tg_info = tg_info.dropna(subset=[f"{col_guide_rna}_list_filtered"])
        tg_info = tg_info.join(tg_info[
            f"{col_guide_rna}_list_filtered"].apply(
                lambda x: np.nan if not isinstance(x, list) and pd.isnull(
                    x) else "multi" if len(x) > 1 else "single").to_frame(
                        "multiple"))  # multiple- or single-transfected
    for x in [col_num_umis, col_guide_rna, col_guide_rna_new]:
        if f"{x}_original" in ann.obs:
            warn(f"'{x}_original' already in adata. Dropping.")
            print(ann.obs[[f"{x}_original"]])
            ann.obs = ann.obs.drop(f"{x}_original", axis=1)
    ann.obs = ann.obs.join(tg_info[
        f"{col_guide_rna}_list_all"].apply(
            lambda x: kws_pga["feature_split"].join(x) if isinstance(
                x, (np.ndarray, list, set, tuple)) else x).to_frame(
                    col_guide_rna), lsuffix="_original"
                )  # processed full gRNA string without guide_split...
    ann.obs = ann.obs.join(tg_info[col_num_umis + "_filtered"].to_frame(
        col_num_umis), lsuffix="_original")  # filtered UMI (summed~gene)
    ann.obs = ann.obs.join(tg_info[col_guide_rna + "_filtered"].to_frame(
        col_guide_rna_new), lsuffix="_original")  # filtered gRNA summed~gene
    # ann.obs = anns[f].obs.join(
    #     tg_info[col_guide_rna].to_frame(col_condition), 
    #     lsuffix="_original")  # condition column=filtered target genes
    nobs = ann.n_obs
    
    # Remove Multiply-Transfected Cells (optionally)
    if remove_multi_transfected is True:
        print(ann.obs)
        print("\n\n\t*** Removing multiply-transfected cells...")
        ann.obs = ann.obs.join(tg_info[["multiple"]], lsuffix="_original")
        ann = ann[ann.obs.multiple != "multi"]
        print(f"Dropped {nobs - ann.n_obs} out of {nobs} observations "
              f"({round(100 * (nobs - ann.n_obs) / nobs, 2)}" + "%).")
        print(ann.obs)
        
    # Remove Filtered-Out Cells
    print(f"\n\n\t*** Removing filtered-out cells...")
    ann = ann[~ann.obs[col_guide_rna_new].isnull()]
    print(f"Dropped {nobs - ann.n_obs} out of {nobs} observations "
            f"({round(100 * (nobs - ann.n_obs) / nobs, 2)}" + "%).")
    
    # Remove Guide RNA Counts from Gene Expression Matrix
    key_ignore = [key_control]
    if remove_multi_transfected is False:
        key_ignore += list(pd.Series([
            x if kws_pga["feature_split"] in x else np.nan 
            for x in ann.obs[col_guide_rna].unique()]).dropna()
                           )  # ignore multi-transfected during removal
    ann = cr.pp.remove_guide_counts_from_gex_matrix(
        ann, col_guide_rna, key_ignore=key_ignore)
    print(ann.obs)
    ann.uns["guide_rna_keywords"] = kws_pga
    ann.uns["guide_rna_feats_n"] = feats_n
    ann.uns["guide_rna_tg_info"] = tg_info
    ann.uns["guide_rna_tg_info_all"] = tg_info_all
    return ann
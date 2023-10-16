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
import warnings
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

def create_object(file, col_gene_symbols="gene_symbols", assay=None,
                  col_barcode=None, col_sample_id=None, 
                  kws_process_guide_rna=None, **kwargs):
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
    if col_sample_id is not None:  # concatenate multiple datasets
        adatas = [None] * len(file)
        batch_categories = list(file.keys())  # keys = sample IDs
        file = [file[f] for f in file]  # turn to list 
        for f in range(len(file)):
            print(f"\t*** Creating object {f + 1} of {len(file)}")
            asy = assay if isinstance(
                assay, str) or assay is None else assay[f]  # GEX/RNA assay 
            cgs = col_gene_symbols if isinstance(col_gene_symbols, str) or (
                col_gene_symbols is None) else col_gene_symbols[f]  # symbols
            adatas[f] = cr.pp.create_object(
                file[f], assay=asy, col_gene_symbols=cgs, 
                kws_process_guide_rna=None, **kwargs)  # create AnnData object
        adata = AnnData.concatenate(
            *adatas, join="inner", batch_key=col_sample_id, uns_merge="same", 
            index_unique="-",  batch_categories=batch_categories,
            fill_value=None)  # concatenate adata objects
    elif isinstance(file, (str, os.PathLike)) and os.path.splitext(
        file)[1] == ".h5mu":  # MuData
        print(f"\n<<< LOADING FILE {file} with muon.read()>>>")
        adata = muon.read(file)
    elif isinstance(file, dict):
        adata = combine_matrix_protospacer(
            **file, col_gene_symbols=col_gene_symbols, col_barcode=col_barcode,
            **kwargs)  # when perturbation info not in mtx
    elif not isinstance(file, (str, os.PathLike)):
        print(f"\n<<< LOADING OBJECT>>>")
        adata = file.copy()
    elif os.path.isdir(file):  # if directory, assume 10x format
        print(f"\n<<< LOADING 10X FILE {file}>>>")
        adata = sc.read_10x_mtx(file, var_names=col_gene_symbols, cache=True,
                                **kwargs)  # 10x matrix, barcodes, features
    elif os.path.splitext(file)[1] == ".h5":
        print(f"\n<<< LOADING 10X .h5 FILE {file}>>>")
        print(f"H5 File Format ({file})\n\n")
        explore_h5_file(file, "\n\n\n")
        adata = sc.read_10x_h5(file, **kwargs)
    else:
        print(f"\n<<< LOADING FILE {file} with sc.read()>>>")
        adata = sc.read(file)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    if col_gene_symbols not in adata.var.columns:
        adata.var = adata.var.rename_axis(col_gene_symbols)
    if kws_process_guide_rna is not None:  # guide RNA processing
        kpr = kws_process_guide_rna[f] if isinstance(
            kws_process_guide_rna, list) else kws_process_guide_rna
        if assay:
            adatas[f][asy]  = cr.pp.process_guide_rna(
                    adatas[f][asy].copy(), **kpr)  # process gRNA
        else:
            adatas[f]  = cr.pp.process_guide_rna(
                adatas[f].copy(), **kpr)  # process gRNA
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
    

def process_data(adata, assay=None, assay_protein=None,
                 layer_original="original",
                 col_gene_symbols=None,
                 col_cell_type=None,
                 # remove_doublets=True,
                 outlier_mads=None,
                 cell_filter_pmt=5,
                 cell_filter_ncounts=None, 
                 cell_filter_ngene=None,
                 gene_filter_ncell=None,
                 target_sum=1e4,
                 logarithmize=True,
                 kws_hvg=True,
                 scale=10, kws_scale=None,
                 regress_out=regress_out_vars, 
                 **kwargs):
    """
    Perform various data processing steps.

    Args:
        adata (AnnData or MuData): The input data object.
        assay (str, optional): The name of the gene expression assay 
            (for multi-modal data only). Defaults to None.
        assay_protein (str, optional): The name of the protein assay. 
            (for multi-modal data only). Defaults to None.
        layer_original (str, optional): The name of the AnnData layer 
            to store the original data. Defaults to "raw".
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
        logarithmize (bool, optional): Whether to log-transform 
            the data after total-count normalization. Defaults to True.
        kws_hvg (dict or bool, optional): The keyword arguments for 
            detecting variable genes or True for default arguments. 
            To calculate HVGs without filtering by them, include an 
            extra argument in this dictionary: {"filter": False}.
            That way, the highly_variable column is created and can
            be used in clustering, but non-HVGs will be retained in 
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
    # Initial Information
    print(adata)
    if assay:
        adata[assay].layers[layer_original] = adata[assay].X
    else:
        adata.layers[layer_original] = adata.X
    if outlier_mads is not None:  # if filtering based on calculating outliers 
        if isinstance(outlier_mads, (int, float)):  # same MADs, all metrics
            qc_mets = ["log1p_total_counts", "log1p_n_genes_by_counts",
                       "pct_counts_in_top_20_genes"]
            outlier_mads = dict(zip(qc_mets, [outlier_mads] * len(qc_mets)))
    if col_gene_symbols == adata.var.index.names[0]:
        col_gene_symbols = None
    if col_cell_type is not None and col_cell_type in adata.obs:
        print(f"\n\n{'=' * 80}\nCell Counts\n{'=' * 80}\n\n")
        print(adata.obs[col_cell_type].value_counts())
    figs = {}
    n_top = kwargs.pop("n_top") if "n_top" in kwargs else 20
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    print(col_gene_symbols, assay, n_top)
    print((adata[assay] if assay else adata).var.describe())

    # Basic Filtering (DO FIRST)
    print("\n<<< FILTERING CELLS (TOO FEW GENES) & GENES (TOO FEW CELLS) >>>") 
    sc.pp.filter_cells(adata[assay] if assay else adata, 
                       min_genes=cell_filter_ngene[0])
    sc.pp.filter_genes(adata[assay] if assay else adata, 
                       min_cells=gene_filter_ncell[0])
        
    
    # Highly-Expressed Genes
    figs["highly_expressed_genes"] = sc.pl.highest_expr_genes(
        adata[assay] if assay else adata, n_top=n_top,
        gene_symbols=col_gene_symbols)
    if kws_scale is None:
        kws_scale = {}
    
    # QC Metrics
    print("\n<<< PERFORMING QUALITY CONTROL ANALYSIS>>>")
    figs["qc_metrics"] = perform_qc(
        adata[assay] if assay else adata)  # calculate & plot QC
    
    # Filtering
    filter_qc(adata[assay] if assay else adata, outlier_mads=outlier_mads, 
              cell_filter_pmt=cell_filter_pmt,
              cell_filter_ncounts=cell_filter_ncounts,
              cell_filter_ngene=cell_filter_ngene, 
              gene_filter_ncell=gene_filter_ncell)
    
    # Doublets
    # doublets = detect_doublets(adata[assay] if assay else adata)
    # if remove_doublets is True:
    #     adata
    #     # TODO: doublets
    
    # Plot Post-Filtering Metrics + Shifted Logarithm
    # scales_counts = sc.pp.normalize_total(
    #     adata[assay] if assay else adata, target_sum=target_sum, 
    #     inplace=False)  # count-normalize (not in-place yet)
    # if assay:
    #     adata[assay].layers["log1p_norm"] = sc.pp.log1p(
    #         scales_counts["X"], copy=True)  # set as layer
    # else:
    #     adata.layers["log1p_norm"] = sc.pp.log1p(
    #         scales_counts["X"], copy=True)  # set as layer
    # fff, axes = plt.subplots(1, 2, figsize=(10, 5))
    # _ = seaborn.histplot((adata[assay] if assay else adata).obs[
    #     "total_counts"], bins=100, kde=False, ax=axes[0])
    # axes[0].set_title("Total Counts (Post-Filtering)")
    # _ = seaborn.histplot(adata.layers["log1p_norm"].sum(1), bins=100, 
    #                      kde=False, ax=axes[1])
    # axes[1].set_title("Shifted Logarithm")
    # plt.show()
    # figs["normalization"] = fff
    
    # Normalize (Actually Modify Object Now)
    if target_sum is not None:
        print("\n<<< TOTAL-COUNT-NORMALIZING >>>")
        sc.pp.normalize_total(
            adata[assay] if assay else adata, target_sum=target_sum, 
            inplace=True)  # count-normalize (not in-place yet)
    else:
        print("\n<<< ***NOT*** TOTAL-COUNT-NORMALIZING >>>")
    if logarithmize is True:
        print("\n<<< LOG-NORMALIZING >>>")
        sc.pp.log1p(adata[assay] if assay else adata)
    else:
        print("\n<<< ***NOT*** LOG-NORMALIZING >>>")
    if assay_protein is not None:  # if includes protein assay
        muon.prot.pp.clr(adata[assay_protein])
        
    # Freeze Normalized, Filtered data
    if assay:
        adata[assay].raw = adata[assay].copy()
    else:
        adata.raw = adata.copy()
        
    # Filter by Gene Variability 
    if kws_hvg is not None:
        print("\n<<< DETECTING VARIABLE GENES >>>")
        if kws_hvg is True:
            kws_hvg = {}
        filter_hvgs = kws_hvg.pop("filter") if "filter" in kws_hvg else True
        sc.pp.highly_variable_genes(adata[assay] if assay else adata, 
                                    **kws_hvg)  # highly variable genes 
        try:
            if filter_hvgs is True:
                if assay is None:
                    adata = adata[:, adata.var.highly_variable
                                  ]  # filter by HVGs
                else:
                    adata[assay]= adata[:, adata[
                        assay].var.highly_variable]  # filter by HVGs
        except (TypeError, IndexError) as err_h:
                warnings.warn(f"""\n\n{'=' * 80}\n\n Could not subset 
                              by highly variable genes: {err_h}""")
    
    # Regress Confounds
    if regress_out: 
        print("\n<<< REGRESSING OUT CONFOUNDS >>>")
        sc.pp.regress_out(adata[assay] if assay else adata, regress_out)
    
    # Scaling Genes
    if scale is not None:
        print("\n<<< SCALING >>>")
        if scale is True:  # if True, just scale to unit variance
            sc.pp.scale(adata[assay] if assay else adata, 
                        **kws_scale)  # scale
        else:  # if scale provided as an integer...
            sc.pp.scale(adata[assay] if assay else adata, max_value=scale,
                        **kws_scale)  # ...also clip values > "scale" SDs
            
    # Cell Counts (Post-Processing)
    if col_cell_type is not None and col_cell_type in adata.obs:
        print(f"\n\n{'=' * 80}\nCell Counts (Post-Processing)"
              "\n{'=' * 80}\n\n")
        print(adata.obs[col_cell_type].value_counts())
        
    # Gene Expression Heatmap
    # if assay:
    #     labels = adata[assay].var.reset_index()[col_gene_symbols]  # genes
    #     if labels[0] != adata[assay].var.index.values[0]:
    #         labels = dict(zip(list(adata[assay].var.index.values), list(
    #             adata[assay].var.reset_index()[col_gene_symbols])))
    # else:
    #     labels = adata.var.reset_index()[col_gene_symbols]  # gene names
    #     if labels[0] != adata.var.index.values[0]:
    #         labels = dict(zip(list(adata.var.index.values), list(
    #             adata.var.reset_index()[col_gene_symbols])))
    # figs["gene_expression"] = sc.pl.heatmap(
    #     adata[assay] if assay else adata, labels,
    #     col_cell_type, show_gene_labels=True, dendrogram=True)
    # if layer is not None:
    #     labels = adata.var.reset_index()[col_gene_symbols]  # gene names 
    #     figs[f"gene_expression_{layer}"] = sc.pl.heatmap(
    #         adata[assay] if assay else adata, labels,
    #         layer=layer, show_gene_labels=True, 
    #         dendrogram=True)
            
    print("\n\n")
    return adata, figs


def perform_qc(adata):
    """Calculate & plot quality control metrics."""
    figs = {}
    patterns = dict(zip(["mt", "ribo", "hb"], 
                        [("MT-", "mt-"), ("RPS", "RPL"), ("^HB[^(P)]")]))
    patterns_names = dict(zip(patterns, ["Mitochondrial", "Ribosomal", 
                                         "Hemoglobin"]))
    p_names = [patterns_names[k] for k in patterns_names]
    print(f"\n\t*** Detecting {', '.join(p_names)} genes...") 
    for k in patterns:
        try:
            adata.var[k] = adata.var_names.str.startswith(patterns[k])
        except Exception as err:
            warnings.warn(f"\n\n{'=' * 80}\n\nCouldn't assign {k}: {err}")
    qc_vars = list(set(patterns.keys()).intersection(
        adata.var.keys()))  # available QC metrics 
    pct_ns = [f"pct_counts_{k}" for k in qc_vars]
    # pct_counts_vars = dict(zip(qc_vars, pct_ns))
    print("\n\t*** Calculating & plotting QC metrics...\n\n") 
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, percent_top=None, 
                               log1p=True, inplace=True)  # QC metrics
    rrs, ccs = cr.pl.square_grid(len(pct_ns + ["n_genes_by_counts"]))  # dims
    fff, axs = plt.subplots(rrs, ccs, figsize=(5 * rrs, 5 * ccs))  # subplots
    for a, v in zip(axs.flat, pct_ns + ["n_genes_by_counts"]):
        try:  # unravel axes to get coordinates, then scatterplot facet
            # aaa = np.unravel_index(a.get_subplotspec().num1, (rrs, ccs))
            # a_x = fff.add_subplot(a)
            sc.pl.scatter(adata, x="total_counts", y=v, ax=a, show=False)
        except Exception as err:
            print(err)
    plt.show()
    figs[f"qc_{v}_scatter"] = fff
    try:
        varm = pct_ns + ["n_genes_by_counts"]
        figs["pairplot"] = seaborn.pairplot(
            adata.obs[varm].rename_axis("Metric", axis=1).rename({
                "total_counts": "Total Counts", **patterns_names}, axis=1), 
            diag_kind="kde", diag_kws=dict(fill=True, cut=0))
    except Exception as err:
        figs["pairplot"] = err
        print(err)
    try:
        figs["pct_counts_kde"] = seaborn.displot(
            adata.obs[pct_ns].rename_axis("Metric", axis=1).rename(
                patterns_names, axis=1).stack().to_frame("Percent Counts"), 
            x="Percent Counts", col="Metric", kind="kde", cut=0, fill=True)
    except Exception as err:
        figs["pct_counts_kde"] = err
        print(err)
    try:
        figs["metrics_violin"] = sc.pl.violin(
            adata, ["n_genes_by_counts", "total_counts"] + pct_ns,
            jitter=0.4, multi_panel=True)
    except Exception as err:
        figs["qc_metrics_violin"] = err
        print(err)
    try:
        figs["qc_log"] = seaborn.jointplot(
            data=adata.obs, x="log1p_total_counts", 
            y="log1p_n_genes_by_counts", kind="hex")
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
            sc.pp.filter_cells(adata, 
                # min_genes=None, min_counts=None, max_counts=None,
                max_genes=cell_filter_ngene[1])
            print(f"\tNew Count: {adata.n_obs}")
        print("\n\t*** Filtering cells based on # of reads...")
        if cell_filter_ncounts[0] is not None:
            print(f"\n\tMinimum: {cell_filter_ncounts[0]}")
            sc.pp.filter_cells(adata, 
                # min_genes=None, max_genes=None, max_counts=None,
                min_counts=cell_filter_ncounts[0])
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


def assign_guide_rna(adata, col_num_umis="num_umis", 
                     assignment_threshold=5, method="max"):
    """Assign guide RNAs to cells (based on pertpy tutorial notebook)."""
    layer, out_layer = "guide_counts", "assigned_guides"
    gdo = adata.copy()
    gdo.obs = gdo.obs[[col_num_umis]].rename(
        {col_num_umis: "nCount_RNA"}, axis=1)
    gdo.layers[layer] = gdo.X.copy()
    sc.pp.log1p(gdo)
    pt.pl.guide.heatmap(gdo, key_to_save_order="plot_order")
    g_a = pt.pp.GuideAssignment()
    if method.lower() == "max":
        g_a.assign_by_threshold(gdo, assignment_threshold=assignment_threshold, 
                                layer=layer, output_layer=out_layer)
    else:
        g_a.assign_by_threshold(gdo, assignment_threshold=assignment_threshold, 
                                layer=layer, output_layer=out_layer)
    pt.pl.guide.heatmap(gdo, layer=out_layer, order_by="plot_order")
    return gdo


def remove_batch_effects(adata, col_cell_type="leiden", 
                         col_batch="batch", plot=True):
    """Remove batch effects (IN PROGRESS)."""
    if plot is True:
        sc.pl.umap(adata, color=[col_batch, col_cell_type], 
                   wspace=.5, frameon=False)
    pt.tl.SCGEN.setup_anndata(adata, batch_key=col_batch, 
                              labels_key=col_cell_type)
    model = pt.tl.SCGEN(adata)
    model.train(max_epochs=100, batch_size=32, 
                early_stopping=True, early_stopping_patience=25)
    corrected_adata = model.batch_removal()
    if plot is True:
        sc.pp.neighbors(corrected_adata)
        sc.tl.umap(corrected_adata)
        sc.pl.umap(corrected_adata, color=[col_batch, col_cell_type], 
                   wspace=0.4, frameon=False)
    return corrected_adata


def detect_guide_targets(col_guide_rna_series,
                         feature_split="|", guide_split="-",
                         key_control_patterns=None,
                         key_control="Control", **kwargs):
    """Detect guide gene targets (see `filter_by_guide_counts` docstring)."""
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
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
            warnings.warn(
                f"Dropping rows with NaNs in `col_guide_rna`.")
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
                           max_percent_umis_control_drop=75,
                           min_percent_umis=40,
                           feature_split="|", guide_split="-",
                           key_control_patterns=None,
                           key_control="Control", **kwargs):
    """
    Process sgRNA names (e.g., multi-probe names).

    Args:
        adata (AnnData): AnnData object (RNA assay, so if multi-modal, subset).
        col_guide_rna (str): _description_
        col_num_umis (str): Column with the UMI counts.
        max_percent_umis_control_drop (int, optional): If control UMI counts 
            are less than or equal to this percentage of the total counts for 
            that cell, and if a non-control sgRNA is also present and 
            meets other filtering criteria, then consider that cell 
            pseudo-single-transfected (non-control gene). Defaults to 75.
        min_percent_umis (int, optional): sgRNAs with counts below this 
            percentage will be considered noise for that guide. Defaults to 40.
        feature_split (str, optional): For designs with multiple guides,
            the character that splits guide names in `col_guide_rna`. 
            For instance, "|" for `STAT1-1|CNTRL-1|CDKN1A`. Defaults to "|".
            If only single guides, set to None.
        guide_split (str, optional): The character that separates 
            guide (rather than gene target)-specific IDs within gene. 
            For instance, guides targeting STAT1 may include 
            STAT1-1, STAT1-2, etc.; the argument would be "-" so the function  
            can identify all of those as targeting STAT1. Defaults to "-".
        key_control_patterns (list, optional): List (or single string) 
            of patterns in guide RNA column entries that correspond to a 
            control. For instance, if control entries in the original 
            `col_guide_rna` column include `NEGCNTRL` and
            `Control.D`, you should specify ['Control', 'CNTRL'] 
            (assuming no non-control sgRNA names contain those patterns). 
            If blank entries should be interpreted as control guides, 
            then include np.nan/numpy.nan in this list.
            Defaults to None -> [np.nan].
        key_control (str, optional): The name you want the control 
            entries to be categorized as under the new `col_guide_rna`. 
            for instance, `CNTRL-1`, `NEGCNTRL`, etc. would all be replaced by 
            "Control" if that's what you specify here. Defaults to "Control".

    Returns:
        pandas.DataFrame: A dataframe (a) with sgRNA names replaced under 
            their target gene categories (or control) and
            (b) with `col_guide_rna` and `col_num_umis` column entries
            (strings) grouped into lists (new columns with suffix "list_all"). 
            Note that the UMI counts are summed across sgRNAs 
            targeting the same gene within a cell. Also versions of the columns
            (and the corresponding string versions) filtered by the specified  
            criteria (with suffixes "_filtered" and "_list_filtered" 
            for list versions).
            
    Notes:
        FUTURE DEVELOPERS: The Crispr class object initialization depends on
        names of the columns created in this function. If they are changed
        (which should be avoided), be sure to change throughout the package.
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
            bad_gene_symb = np.array(adata.var_names)[np.where(split_char)[0]]
            if grs in guide_split:
                raise ValueError(f"{grs} is a reserved name and cannot be "
                                 "contained within `guide_split`.")
            warnings.warn(f"`guide_split` ({guide_split}) found in at least "
                          f"one gene name ({', '.join(bad_gene_symb)}). "
                          f"Temporarily substituting {grs}. "
                          "Will attempt to replace later, but keep in "
                          "mind that there are big risks in having "
                          "`guide_split` be a character that is allowed to "
                          "be included in gene names.")
            guides = guides.apply(lambda x: re.sub(bad_gene_symb[np.where(
                [i in str(x) for i in bad_gene_symb])[0][0]], re.sub(
                guide_split, grs, bad_gene_symb[np.where(
                    [i in str(x) for i in bad_gene_symb])[0][0]]), 
                str(x)) if any((i in str(x) for i in bad_gene_symb)) else x)
    
    # Find Gene Targets & Counts of Guides
    targets, grnas = detect_guide_targets(
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
        warnings.warn(f"NaNs present in {col_guide_rna} column. "
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

    # Filtering
    feats_n = feats_n.join(
    feats_n.p.groupby(["bc", "g"]).apply(
        lambda x: "control" if (x.name[1] == key_control and 
                                float(x) <= max_percent_umis_control_drop) 
        else "low_umi" if float(x) < min_percent_umis
        else np.nan).to_frame("umi_category"))  # filter
    filt = feats_n[~feats_n.umi_category.isnull()]
    filt = filt.n.to_frame("u").groupby("bc").apply(
        lambda x: pd.Series({cols[0]: list(x.reset_index("g")["g"]), 
                            cols[1]: list(x.reset_index("g")["u"])}))
    tg_info = tg_info.join(filt, lsuffix="_all", rsuffix="_filtered")  # join
    tg_info = tg_info.dropna().loc[ann.obs.index.intersection(
        tg_info.dropna().index)]  # re-order according to adata index
    
    # # Determine Exclusion Criteria Met per gRNA
    # tg_info = tg_info.join(feats_n.to_frame("u").groupby("bc").apply(
    #     lambda x: pd.Series({cols[0]: list(x.reset_index("g")["g"]), 
    #                          cols[1]: list(x.reset_index("g")["u"])})), 
    #                        rsuffix="_unique", lsuffix="_all")
    # ecol, grc, nuc = [f"{col_guide_rna}_exclusions", f"{col_guide_rna}_list_unique", 
    #                   f"{col_num_umis}_percent"]  # column names
    # tg_info.loc[:, ecol] = tg_info[grc].apply(lambda x: [""] * len(x))  # placehold
    # tg_info.loc[:, ecol] = tg_info.apply(
    #     lambda x: [x[ecol][i] + " " + "low_umis" if x[
    #         nuc][i] < min_percent_umis else x[ecol][i] 
    #                for i, q in enumerate(x[grc])], axis=1)  # low UMI count
    # tg_info.loc[:, ecol] = tg_info.apply(
    #     lambda x: [x[ecol][i] + " " + "tolerable_control_umis" if (
    #         q == key_control and any(np.array(x[grc]) != key_control) and x[
    #             nuc][i] <= max_percent_umis_control_drop) else x[ecol][i] 
    #                for i, q in enumerate(x[grc])], axis=1
    #     )  # control guides w/i tolerable % => drop control guide
    # tg_info.loc[:, ecol] = tg_info[ecol].apply(lambda x: [
    #     np.nan if i == "" else i for i in x])
    
    # Re-Make String Versions of New Columns with List Entries
    for q in [col_guide_rna, col_num_umis]:  # string versions of list entries 
        tg_info.loc[:, q + "_filtered"] = tg_info[q + "_list_filtered"].apply( 
            lambda x: feature_split.join(str(i) for i in x)
            )  # join names of processed/filtered gRNAs by `feature_split`
        
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
    tg_info, feats_n = cr.pp.filter_by_guide_counts(
        ann, col_guide_rna, col_num_umis=col_num_umis,
        key_control=key_control, **kws_pga
        )  # process (e.g., multi-probe names) & filter by # gRNA
    
    # Add Results to AnnData
    ann.uns["guide_rna"] = {}
    ann.uns["guide_rna"]["keywords"] = kws_pga
    ann.uns["guide_rna"]["counts_unfiltered"] = feats_n
    try:
        tg_info = tg_info.loc[ann.obs.index]
    except Exception as err:
        warnings.warn(f"{err}\n\nCouldn't re-order tg_info "
                      "in process_guide_rna() to mirror adata index!")
    tg_info_all = None  # fill later if needed
    if remove_multi_transfected is True:  # remove multi-transfected
        ann.uns["guide_rna"]["counts_single_multi"] = tg_info.copy()
        tg_info_all = tg_info.copy() if conserve_memory is False else None
        tg_info = tg_info.loc[tg_info[
            f"{col_guide_rna}_list_filtered"].apply(
                lambda x: np.nan if len(x) > 1 else x).dropna().index] 
    ann.uns["guide_rna"]["counts"] = tg_info.copy()
    for x in [col_num_umis, col_guide_rna, col_guide_rna_new]:
        if f"{x}_original" in ann.obs:
            warnings.warn(f"'{x}_original' already in adata. Dropping.")
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
    ann.obs = ann.obs.join(tg_info[col_guide_rna].to_frame(
        col_guide_rna_new), lsuffix="_original")  # filtered gRNA summed~gene
    # ann.obs = anns[f].obs.join(
    #     tg_info[col_guide_rna].to_frame(col_condition), 
    #     lsuffix="_original")  # condition column=filtered target genes
    return ann, tg_info, feats_n, tg_info_all
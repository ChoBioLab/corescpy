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
# import anndata
import crispr as cr
import pandas as pd
import numpy as np
# from crispr.defaults import names_layers

regress_out_vars = ["total_counts", "pct_counts_mt"]

def create_object(file, col_gene_symbols="gene_symbols", assay=None,
                  col_barcode=None, **kwargs):
    """Create object from Scanpy- or Muon-compatible file."""
    # extension = os.path.splitext(file)[1]
    if isinstance(file, (str, os.PathLike)) and os.path.splitext(
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
    # TODO: Flesh this out, generalize, test, etc.
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    if col_gene_symbols not in adata.var.columns:
        adata.var = adata.var.rename_axis(col_gene_symbols)
    if assay is not None:
        adata = adata[assay]  # subset by assay if desired 
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
                 col_gene_symbols=None,
                 col_cell_type=None,
                 # remove_doublets=True,
                 target_sum=1e4,
                 outlier_mads=None,
                 cell_filter_pmt=5,
                 cell_filter_ncounts=None, 
                 cell_filter_ngene=None,
                 gene_filter_ncell=None,
                 gene_count_range=None,
                 scale=10,  # or scale=True for no clipping
                 regress_out=regress_out_vars, 
                 kws_hvg=None, kws_scale=None,
                 **kwargs):
    """
    Perform various data processing steps.

    Args:
        adata (AnnData or MuData): The input data object.
        assay (str, optional): The name of the gene expression assay 
            (for multi-modal data only). Defaults to None.
        assay_protein (str, optional): The name of the protein assay. 
            (for multi-modal data only). Defaults to None.
        col_gene_symbols (str, optional): The name of the column or 
            index in `.var` containing gene symbols. Defaults to None.
        col_cell_type (str, optional): The name of the column 
            in `.obs` containing cell types. Defaults to None.
        target_sum (float, optional): Total-count normalize to
            <target_sum> reads per cell, allowing between-cell 
            comparability of counts. Defaults to 1e4.
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
        scale (int or bool, optional): The scaling factor or True for no clipping. Defaults to 10.
        regress_out (list or None, optional): The variables to regress out. Defaults to regress_out_vars.
        kws_hvg (dict or bool, optional): The keyword arguments for detecting variable genes or True for default arguments. Defaults to None.
        kws_scale (dict, optional): The keyword arguments for scaling. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        adata (AnnData): The processed data object.
        figs (dict): A dictionary of generated figures.
    """
    # Initial Information
    print(adata)
    if assay:
        adata[assay].layers["raw_original"] = adata[assay].X 
    else:
        adata.layers["raw_original"] = adata.X
    if outlier_mads is not None:  # if filtering based on calculating outliers 
        if isinstance(outlier_mads, (int, float)):  # same MADs, all metrics
            qc_mets = ["log1p_total_counts", "log1p_n_genes_by_counts",
                       "pct_counts_in_top_20_genes"]
            outlier_mads = dict(zip(qc_mets, [outlier_mads] * len(qc_mets)))
    if col_gene_symbols == adata.var.index.names[0]:
        col_gene_symbols = None
    cell_gene_count_range, cell_count_range = [x if x else None for x in [
        cell_gene_count_range, cell_count_range]]
    gene_cell_count_range, gene_count_range = [x if x else None for x in [
        gene_cell_count_range, gene_count_range]]
    if col_cell_type is not None and col_cell_type in adata.obs:
        print(f"\n\n{'=' * 80}\nCell Counts\n{'=' * 80}\n\n")
        print(adata.obs[col_cell_type].value_counts())
    figs = {}
    n_top = kwargs.pop("n_top") if "n_top" in kwargs else 20
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    min_pct_mt, max_pct_mt = cell_filter_pmt
    print(col_gene_symbols, assay, n_top)

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
    print("\n\t\t* Detecting mitochondrial, ribosomal, & hemoglobin genes...") 
    figs["qc_metrics"] = calculate_qc_metrics(
        adata[assay] if assay else adata)  # calculate & plot QC
    
    # Filtering
    if outlier_mads is not None:  # automatic filtering using outlier stats
        outliers = adata[assay].obs[outlier_mads.keys()]
        for x in outlier_mads:
            outliers.loc[:, f"outlier_{x}"] = cr.tl.is_outlier(
                adata[assay].obs, outlier_mads[x])
        if assay:
            cols_outlier = list(set(
                outliers.columns.difference(adata[assay].obs.columns)))
        else:
            cols_outlier = list(set(outliers.columns.difference(
                adata.obs.columns)))
        outliers.loc[:, "outlier"] = outliers[cols_outlier].any()
        if assay:
            adata[assay].obs = adata[assay].obs.join(outliers[["outlier"]])
        else:
            adata.obs = adata.obs.join(outliers[["outlier"]])
        print(f"Total Cell Count: {(adata[assay] if assay else adata).n_obs}")
        if assay:
            adata[assay] = adata[assay][~adata[assay].obs.outlier].copy()
        else:
            adata = adata[(~adata.obs.outlier) & (
                ~adata.obs.mt_outlier)].copy()
        print("Post-Filtering Cell Count: "
              f"{(adata[assay] if assay else adata).n_obs}")
    else:  # manual filtering
        print(f"Total Cell Count: {(adata[assay] if assay else adata).n_obs}")
        qc_mets = ["pct_counts_mt", "n_genes_by_counts"]
        if assay:
            adata[assay] = adata[assay][(adata[
                assay].obs.pct_counts_mt < max_pct_mt) * (
                    adata[assay].obs.pct_counts_mt > min_pct_mt)
                ]  # filter based on MT %
        else:
            adata = adata[(adata.obs.pct_counts_mt < max_pct_mt) * (
                adata.obs.pct_counts_mt > min_pct_mt)]  # filter based on MT %
        sc.pp.filter_cells(adata[assay] if assay else adata, 
                           min_genes=cell_filter_ngene[0])
        sc.pp.filter_cells(adata[assay] if assay else adata, 
                           max_genes=cell_filter_ngene[1])
        # if cell_filter_ngene[0] is not None:
        sc.pp.filter_cells(adata[assay] if assay else adata, 
                           min_counts=cell_filter_ncounts[0])
        # if cell_filter_ngene[1] is not None:
        sc.pp.filter_cells(adata[assay] if assay else adata, 
                           max_counts=cell_filter_ncounts[1])
        # if gene_filter_ncell[0] is not None:
        sc.pp.filter_genes(adata[assay] if assay else adata, 
                           min_cells=gene_filter_ncell[0])
        # if gene_filter_ncell[1] is not None:
        sc.pp.filter_genes(adata[assay] if assay else adata, 
                           max_cells=gene_filter_ncell[1])
        print("Post-Filtering Cell Count: "
              f"{(adata[assay] if assay else adata).n_obs}")
    
    # Filtering
        
    # Doublets
    # doublets = detect_doublets(adata[assay] if assay else adata)
    # if remove_doublets is True:
    #     adata
    #     # TODO: doublets
    
    # Normalize
    print("\n<<< NORMALIZING >>>")
    scales_counts = sc.pp.normalize_total(
        adata[assay] if assay else adata, 
        target_sum=target_sum, inplace=False)  # count-normalize
    if assay:
        adata[assay].layers["log1p_norm"] = sc.pp.log1p(
            scales_counts["X"], copy=True)
    # adata.X = adata.layers["log1p_norm"]
    if assay:
        adata[assay] = sc.pp.log1p(adata[assay].layers[
            "log1p_norm"])  # log-normalize
    else:
        adata = sc.pp.log1p(adata.layers["log1p_norm"])  # log-normalize
    adata.raw = adata  # freeze normalized & filtered adata
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    p_1 = seaborn.histplot((adata[assay] if assay else adata).obs[
        "total_counts"], bins=100, kde=False, ax=axes[0])
    axes[0].set_title("Total Counts")
    p_2 = seaborn.histplot(adata.layers["log1p_norm"].sum(1), bins=100, 
                           kde=False, ax=axes[1])
    axes[1].set_title("Shifted Logarithm")
    plt.show()
    figs["normalization"] = fig
    if assay_protein is not None:  # if includes protein assay
        muon.prot.pp.clr(adata[assay_protein])
        
    # Freeze Normalized, Filtered data
    if assay:
        adata[assay].raw = adata[assay].copy()
    else:
        adata.raw = adata.copy()
        
    # Filter by Gene Count & Variability 
    if kws_hvg is not None:
        print("\n<<< DETECTING VARIABLE GENES >>>")
        if kws_hvg is True:
            kws_hvg = {}
        figs["highly_variable_genes"] = sc.pp.highly_variable_genes(
            adata[assay] if assay else adata, 
            **kws_hvg)  # highly variable genes 
        try:
            if assay is None:
                adata= adata[:, adata.var.highly_variable]  # filter by HVGs
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
        print(f"\n\n{'=' * 80}\nCell Counts (Post-Processing)\n{'=' * 80}\n\n")
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


def calculate_qc_metrics(adata, assay=None):
    """Calculate & plot quality control metrics."""
    figs = {}
    if patterns is None:
        patterns = dict(zip(["mt", "ribo", "hb"], 
                            [("MT-", "mt-"), ("RPS", "RPL"), ("^HB[^(P)]")]))
    for k in patterns:
        try:
            adata.var[k] = adata.var_names.str.startswith(patterns[k])
        except Exception as err:
            warnings.warn(f"\n\n{'=' * 80}\n\nCould not assign {k}: {err}")
    else:
        try:
            for k in patterns:
                adata[assay].var[k] = adata[assay].var_names.str.startswith(
                    patterns[k])
        except TypeError as err_mt:
            warnings.warn(f"\n\n{'=' * 80}\n\nCould not assign {k}: {err_mt}") 
    qc_vars = list(set(patterns.keys()).intersection((
        adata[assay] if assay else adata).var.keys()))  # available QC metrics 
    pct_counts_vars = dict(zip(qc_vars, [f"pct_counts_{k}" for k in qc_vars]))
    # p1 = sns.displot(adata.obs["total_counts"], bins=100, kde=False)
    # sc.pl.violin(adata, 'total_counts')
    # p2 = sc.pl.violin(adata, "pct_counts_mt")
    # p3 = sc.pl.scatter(adata, "total_counts", "n_genes_by_counts", 
    #                    color="pct_counts_mt")
    print("\n\t\t* Calculating & plotting QC metrics...\n\n") 
    sc.pp.calculate_qc_metrics(adata[assay] if assay else adata, 
                               qc_vars=qc_vars, percent_top=None, log1p=True,
                               inplace=True)  # calculate QC metrics
    for k in qc_vars:
        try:
            figs[f"qc_pct_counts_{k}_hist"] = seaborn.histplot(
                (adata[assay] if assay else adata).obs[pct_counts_vars[k]])
        except Exception as err:
            figs[f"qc_pct_counts_{k}_hist"] = err
            print(err)
        try:
            figs["qc_metrics_violin"] = sc.pl.violin(
                adata[assay] if assay else adata, 
                ["n_genes_by_counts", "total_counts"] + pct_counts_vars,
                jitter=0.4, multi_panel=True)
        except Exception as err:
            figs["qc_metrics_violin"] = err
            print(err)
        for v in pct_counts_vars + ["n_genes_by_counts"]:
            try:
                figs[f"qc_{v}_scatter"] = sc.pl.scatter(
                    adata[assay] if assay else adata, x="total_counts", y=v)
            except Exception as err:
                figs[f"qc_{v}_scatter"] = err
                print(err)
    try:
        figs["qc_log"] = seaborn.jointplot(
            data=adata[assay].obs if assay else adata.obs,
            x="log1p_total_counts", y="log1p_n_genes_by_counts", kind="hex")
    except Exception as err:
        figs["qc_log"] = err
        print(err)
    return figs


def explore_h5_file(file):
    """Explore an H5 file's format (thanks to ChatGPT)."""
    with h5py.File(file, "r") as h5_file:
        # List all top-level groups in the HDF5 file
        top_level_groups = list(h5_file.keys())
        # Explore the groups and datasets within each group
        for group_name in top_level_groups:
            group = h5_file[group_name]
            print(f"Group: {group_name}")
            # List datasets within the group
            datasets = list(group.keys())
            for dataset_name in datasets:
                print(f"  Dataset: {dataset_name}")


def get_matrix_from_h5(file, gex_genes_return=None):
    """Get matrix from 10X h5 file (modified from 10x code)."""
    FeatureBCMatrix = collections.namedtuple("FeatureBCMatrix", [
        "feature_ids", "feature_names", "barcodes", "matrix"])
    with h5py.File(file) as f:
        if u"version" in f.attrs:
            version = f.attrs["version"]
            if version > 2:
                print(f"Version = {version}")
                raise ValueError(f"HDF5 format version version too new.")
        else:
            raise ValueError(f"HDF5 format version ({version}) too old.")
        feature_ids = [x.decode("ascii", "ignore") 
                       for x in f["matrix"]["features"]["id"]]
        feature_names = [x.decode("ascii", "ignore") 
                         for x in f["matrix"]["features"]["name"]]        
        barcodes = list(f["matrix"]["barcodes"][:])
        matrix = sp_sparse.csr_matrix((f["matrix"]["data"], 
                                       f["matrix"]["indices"], 
                                       f["matrix"]["indptr"]), 
                                      shape=f["matrix"]["shape"])
        fbm = FeatureBCMatrix(feature_ids, feature_names, barcodes, matrix)
        if gex_genes_return is not None:
            gex = {}
            for g in gex_genes_return:
                try:
                    gene_index = fbm.feature_names.index(g)
                except ValueError:
                    raise Exception(f"{g} not found in list of gene names.")
                gex.update({g: fbm.matrix[gene_index, :].toarray(
                    ).squeeze()})  # gene expression
        else:
            gex = None
        barcodes = [x.tostring().decode() for x in fbm.barcodes]
        genes = pd.Series(fbm.feature_names).to_frame("gene").join(
            pd.Series(fbm.feature_ids).to_frame("gene_ids"))
    return fbm, gex, barcodes, genes


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


def detect_guide_targets(adata, col_guide_rna="guide_ID", 
                         feature_split="|", guide_split="-",
                         key_control_patterns=None,
                         key_control="Control", **kwargs):
    """Detect guide gene targets (see `filter_by_guide_counts` docstring)."""
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    if key_control_patterns is None:
        key_control_patterns = [
            key_control]  # if already converted, pattern=key itself
    ann = adata.copy()
    if feature_split is None:
        feature_split = "|"
        if ann.obs[col_guide_rna].apply(lambda x: feature_split in x).any():
            raise ValueError(
                f"""For single-guide designs, the character {feature_split}
                cannot be found in any of the guide names ({col_guide_rna})""")
    if isinstance(key_control_patterns, str) or pd.isnull(
        key_control_patterns) and not isinstance(key_control_patterns, list):
        key_control_patterns = [key_control_patterns]
    targets = ann.obs[col_guide_rna].str.strip(" ").replace("", np.nan)
    if np.nan in key_control_patterns:  # if NAs mean control sgRNAs
        key_control_patterns = list(pd.Series(key_control_patterns).dropna())
        if len(key_control_patterns) == 0:
            key_control_patterns = [key_control]
        targets = targets.replace(
            np.nan, key_control)  # NaNs replaced w/ control key
    else:  # if NAs mean unperturbed cells
        if any(pd.isnull(targets)):
            warnings.warn(
                f"Dropping rows with NaNs in `col_guide_rna`.")
        targets = targets.dropna()
    keys_leave = [key_control]  # entries to leave alone
    targets, nums = [targets.apply(
        lambda x: [re.sub(p, ["", r"\1"][j], str(i)) for i in x.split(
            feature_split)]) for j, p in enumerate([
                f"{guide_split}.*", rf'^.*?{re.escape(guide_split)}(.*)$'])
                        ]  # each entry -> list of target genes
    # print(key_control_patterns)
    targets = targets.apply(
        lambda x: [i if i in keys_leave else key_control if any(
                (k in i for k in key_control_patterns)) else i 
            for i in x])  # find control keys among targets
    # targets = targets.apply(
    #     lambda x: [[x[0]] if len(x) == 2 and x[1] == "" else x
    #     for i in x])  # in case all single-transfected
    grnas = targets.to_frame("t").join(nums.to_frame("n")).apply(
        lambda x: [i + guide_split + "_".join(np.array(
            x["n"])[np.where(np.array(x["t"]) == i)[0]]) 
                    for i in pd.unique(x["t"])], 
        axis=1).apply(lambda x: feature_split.join(x)).to_frame(
            col_guide_rna
            )  # e.g., STAT1-1|STAT1-2|NT-1-2 => STAT1-1_2
    return targets, grnas


def find_guide_info(adata, col_guide_rna, col_num_umis=None, 
                    feature_split="|", guide_split="-",
                    key_control_patterns=None,
                    key_control="Control", **kwargs):
    """Process guide names/counts (see `filter_by_guide_counts` docstring).""" 
    if isinstance(key_control_patterns, str):
        key_control_patterns = [key_control_patterns]
    ann = adata.copy()
    if feature_split is None:
        feature_split = "|"
        if ann.obs[col_guide_rna].apply(lambda x: feature_split in x).any():
            raise ValueError(
                f"""For single-guide designs, the character {feature_split}
                cannot be found in any of the guide names ({col_guide_rna})""")
    targets, grnas = detect_guide_targets(
        ann, col_guide_rna=col_guide_rna,
        feature_split=feature_split, guide_split=guide_split, 
        key_control_patterns=key_control_patterns, 
        key_control=key_control, **kwargs)  # target genes
    tg_info = grnas.iloc[:, 0].to_frame(
        col_guide_rna + "_flat_ix").join(
            targets.to_frame(col_guide_rna + "_list"))
    if col_num_umis is not None:
        tg_info = tg_info.join(
            ann.obs[[col_num_umis]].apply(
                lambda x: [float(i) for i in str(x[col_num_umis]).split(
                    feature_split)], axis=1).to_frame(
                        col_num_umis + "_list"))
    tg_info = ann.obs[col_guide_rna].to_frame(col_guide_rna).join(tg_info)
    tg_info = tg_info.join(tg_info[col_num_umis + "_list"].dropna().apply(
        sum).to_frame(col_num_umis + "_total"))  # total UMIs/cell
    return tg_info


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
    if feature_split is None:
        feature_split = "|"
        if ann.obs[col_guide_rna].apply(lambda x: feature_split in x).any():
            raise ValueError(
                f"""For single-guide designs, the character {feature_split}
                cannot be found in any of the guide names ({col_guide_rna})""")
        # ann.obs.loc[:, col_guide_rna] = ann.obs[col_guide_rna].apply(
        #     lambda x: np.nan if pd.isnull(x) else str(x) + feature_split
        # )  # add dummy feature_split to make single-guide case work
    tg_info = find_guide_info(
        ann.copy(), col_guide_rna, col_num_umis=col_num_umis, 
        feature_split=feature_split, guide_split=guide_split, 
        key_control_patterns=key_control_patterns, 
        key_control=key_control, **kwargs)

    # Sum Up gRNA UMIs
    cols = [col_guide_rna + "_list", col_num_umis + "_list"]
    feats_n = tg_info[cols].dropna().apply(lambda x: pd.Series(
        dict(zip(pd.unique(x[cols[0]]), [sum(np.array(x[cols[1]])[
            np.where(np.array(x[cols[0]]) == i)[0]]) for i in pd.unique(
                x[cols[0]])]))), axis=1).stack().rename_axis(["bc", "g"])
    feats_n = feats_n.to_frame("n").join(feats_n.groupby(
        "bc").sum().to_frame("t"))  # sum w/i-cell # gRNAs w/ same target gene 
    feats_n = feats_n.assign(p=feats_n.n / feats_n.t * 100)  # to %age

    # Actual Filtering
    filt = feats_n.groupby(["bc", "g"]).apply(
        lambda x: np.nan if x.name[1] == key_control and float(
            x["p"]) <= max_percent_umis_control_drop else np.nan if float(
                x["p"]) < min_percent_umis else x.stack())  # filter
    filt = pd.concat(list(filt.dropna())).unstack(2)
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
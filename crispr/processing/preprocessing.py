#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import os
import scanpy as sc
import pertpy as pt
import muon
import warnings
import scipy
import seaborn
import collections      
import scipy.sparse as sp_sparse
import h5py
import anndata
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
    >>> adata = combine_matrix_protospacer(
    ... "/home/asline01/projects/crispr/examples/data/crispr-screening/HH03, 
    ... "filtered_feature_bc_matrix", col_gene_symbols="gene_symbols", 
    ... file_protospacer="crispr_analysis/protospacer_calls_per_cell.csv", 
    ... col_barcode="cell_barcode")
    
    Or using create_object(), with directory/file-related arguments in 
    a dictionary passed to the "file" argument:
    
    >>> adata = create_object(
    ... dict(directory="/home/asline01/projects/crispr/examples/data/crispr-screening/HH03, 
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
                 remove_doublets=True,
                 target_sum=1e4,  max_genes_by_counts=2500, max_pct_mt=5,
                 min_genes=200, min_cells=3, 
                 scale=10,  # or scale=True for no clipping
                 regress_out=regress_out_vars, 
                 kws_hvg=None, kws_scale=None, kws_crispr=None,
                 **kwargs):
    """
    Preprocess data (plus CRISPR-specific steps if kws_crispr is not None).
    """
    
    # Initial Information
    print(adata)
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
        
    # Doublets
    # doublets = detect_doublets(adata[assay] if assay else adata)
    # if remove_doublets is True:
    #     adata
    #     # TODO: doublets
        
    
    # Highly-Expressed Genes
    figs["highly_expressed_genes"] = sc.pl.highest_expr_genes(
        adata[assay] if assay else adata, n_top=n_top,
        gene_symbols=col_gene_symbols)
    if kws_scale is None:
        kws_scale = {}

    # Filtering
    print("\n<<< FILTERING >>>")
    sc.pp.filter_cells(adata[assay] if assay else adata, min_genes=min_genes)
    sc.pp.filter_genes(adata[assay] if assay else adata, min_cells=min_cells)
    
    # Mitochondrial Count QC
    print("\n<<< DETECTING MITOCHONDRIAL GENES >>>")
    no_mt = False
    if assay is None:
        adata.var["mt"] = adata.var_names.str.startswith(
            "MT-")  # annotate mitochondrial genes
    else:
        try:
            adata[assay].var["mt"] = adata[assay].var_names.str.startswith(
                "MT-")  # annotate mitochondrial genes
        except TypeError as err_mt:
            warnings.warn(f"\n\n{'=' * 80}\n\nCould not assign MT: {err_mt}")
            no_mt = True
    
    # Quality Control
    print("\n<<< PERFORMING QUALITY CONTROL >>>")
    if no_mt is False:
        sc.pp.calculate_qc_metrics(adata[assay] if assay else adata, 
                                   qc_vars=["mt"], percent_top=None, 
                                   log1p=True, inplace=True)
        try:
            figs["qc_pct_counts_mt_hist"] = seaborn.histplot(
                adata[assay].obs["pct_counts_mt"] if assay else adata.obs[
                    "pct_counts_mt"])
        except Exception as err:
            print(err)
        try:
            figs["qc_metrics_violin"] = sc.pl.violin(
                adata[assay] if assay else adata, 
                ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                jitter=0.4, multi_panel=True)
        except Exception as err:
            print(err)
        for v in ["pct_counts_mt", "n_genes_by_counts"]:
            try:
                figs[f"qc_{v}_scatter"] = sc.pl.scatter(
                    adata[assay] if assay else adata, x="total_counts", y=v)
            except Exception as err:
                print(err)
    figs["qc_log"] = seaborn.jointplot(
        data=adata[assay].obs if assay else adata.obs,
        x="log1p_total_counts", y="log1p_n_genes_by_counts", kind="hex")
        
    # More Filtering
    try:
        if assay is None:
            adata = adata[adata.obs.n_genes_by_counts < max_genes_by_counts, :]
            adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]
            adata.raw = adata  # freeze normalized & filtered adata
        else:
            adata[assay] = adata[assay][
                adata[assay].obs.n_genes_by_counts < max_genes_by_counts, :]
            adata[assay] = adata[assay][
                adata[assay].obs.pct_counts_mt < max_pct_mt, :]  # MT counts
            adata[assay].raw = adata[assay]  # freeze normalized, filtered data
    except TypeError as err_f:
            warnings.warn(f"\n\n{'=' * 80}\n\nCould not filter: {err_f}")
    
    # Normalize
    print("\n<<< NORMALIZING >>>")
    sc.pp.normalize_total(adata[assay] if assay else adata, 
                          target_sum=target_sum)  # count-normalize
    sc.pp.log1p(adata[assay] if assay else adata)  # log-normalize
    if assay_protein is not None:  # if includes protein assay
        muon.prot.pp.clr(adata[assay_protein])
        
    # Variable Genes
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


def assign_guide_rna(adata, assignment_threshold=5, layer="counts",
                     plot=False, **kwargs):
    """Assign guide RNAs to cells (based on pertpy tutorial notebook)."""
    gdo = adata.mod["gdo"]
    gdo.layers["counts"] = gdo.X.copy()  # save original counts
    sc.pp.log1p(gdo)  # log-transform data
    if plot is True:
        pt.pl.guide.heatmap(gdo, key_to_save_order="plot_order", 
                            **kwargs)  # heatmap
    g_a = pt.pp.GuideAssignment()  # guide assignment
    g_a.assign_by_threshold(gdo, assignment_threshold=assignment_threshold, 
                            layer=layer)  # assignment thresholding
    g_a.assign_to_max_guide(gdo, assignment_threshold=assignment_threshold, 
                            layer=layer)  # assignment thresholding
    print(gdo.obs["assigned_guide"])  # creates layer "assigned_guides"
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
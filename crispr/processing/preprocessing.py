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
import pandas as pd
import numpy as np

regress_out_vars = ["total_counts", "pct_counts_mt"]

def create_object(file, col_gene_symbols="gene_symbols", assay=None, **kwargs):
    """Create object from Scanpy-compatible file."""
    # extension = os.path.splitext(file)[1]
    if os.path.isdir(file):  # if directory, assume 10x format
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
    if assay is not None:
        adata = adata[assay]  # subset by assay if desired 
    print("\n\n", adata)
    return adata


def process_data(adata, assay=None, assay_protein=None,
                 col_gene_symbols=None,
                 target_sum=1e4,  max_genes_by_counts=2500, max_pct_mt=5,
                 min_genes=200, min_cells=3, 
                 scale=10,  # or scale=True for no clipping
                 regress_out=regress_out_vars, kws_hvg=None, **kwargs):
    """Preprocess adata."""
    
    # Initial Information
    print(adata)
    figs = {}
    n_top = kwargs["n_top"] if "n_top" in kwargs else 20
    print(f"Un-used Keyword Arguments: {kwargs}")
    figs["highly_expressed_genes"] = sc.pl.highest_expr_genes(
        adata[assay] if assay else adata, n_top=n_top,
        gene_symbols=col_gene_symbols)

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
        figs["qc_pct_counts_mt_hist"] = seaborn.histplot(
            adata.obs["pct_counts_mt"])
        figs["qc_metrics_violin"] = sc.pl.violin(adata, [
            "n_genes_by_counts", "total_counts", "pct_counts_mt"],
             jitter=0.4, multi_panel=True)
        for v in ["pct_counts_mt", ""]:
            figs[f"qc_{v}_scatter"] = sc.pl.scatter(
                adata[assay] if assay else adata, x="total_counts", y=v)
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
    # sc.pp.highly_variable_genes(adata[assay] if assay else adata, 
    #                             subset=True)  # highly variable genes
    if assay_protein is not None:  # if includes protein assay
        muon.prot.pp.clr(adata[assay_protein])
        
    # Variable Genes
    if kws_hvg is not None:
        print("\n<<< DETECTING VARIABLE GENES >>>")
        if kws_hvg is True:
            kws_hvg = {}
        sc.pp.highly_variable_genes(adata[assay] if assay else adata, 
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
            sc.pp.scale(adata[assay] if assay else adata)  # scale
        else:  # if scale provided as an integer...
            sc.pp.scale(adata[assay] if assay else adata, 
                        max_value=scale)  # ...also clip values > "scale" SDs
            
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


def remove_batch_effects(col_cell_type="leiden", col_batch="batch"):
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
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
import pandas as pd
import numpy as np

regress_out_vars = ['total_counts', 'pct_counts_mt']

def create_object(file, col_gene_symbols="gene_symbols", assay=None, **kwargs):
    """Create object from Scanpy-compatible file."""
    # extension = os.path.splitext(file)[1]
    if os.path.isdir(file):  # if directory, assume 10x format
        print(f"\n<<< LOADING 10X FILE {file}>>>")
        adata = sc.read_10x_mtx(file, var_names=col_gene_symbols, cache=True,
                                **kwargs)  # 10x matrix, barcodes, features
    elif os.path.splitext(file)[1] == ".h5":
        print(f"\n<<< LOADING 10X .h5 FILE {file}>>>")
        adata = sc.read_10x_h5(file, **kwargs)
    else:
        print(f"\n<<< LOADING FILE {file} with sc.read()>>>")
        adata = sc.read(file)
    # TODO: Flesh this out, generalize, test, etc.
    adata.var_names_make_unique()
    if assay is not None:
        adata = adata[assay]  # subset by assay if desired
    return adata


def process_data(adata, assay=None, assay_protein=None,
                 target_sum=1e4,  max_genes_by_counts=2500, max_pct_mt=5,
                 min_genes=200, min_cells=3, 
                 scale=10,  # or scale=True for no clipping
                 regress_out=regress_out_vars, hvg_kws=None):
    """Preprocess adata."""
    
    # Normalize
    print("\n<<< NORMALIZING >>>")
    sc.pp.normalize_total(adata[assay] if assay else adata, 
                          target_sum=target_sum)  # count-normalize
    sc.pp.log1p(adata[assay] if assay else adata)  # log-normalize
    sc.pp.highly_variable_genes(adata[assay] if assay else adata, 
                                subset=True)  # highly variable genes
    if assay_protein is not None:  # if includes protein assay
        muon.prot.pp.clr(adata[assay_protein])

    
    # Filtering
    print("\n<<< FILTERING >>>")
    sc.pp.filter_cells(adata[assay] if assay else adata, min_genes=min_genes)
    sc.pp.filter_genes(adata[assay] if assay else adata, min_cells=min_cells)
    
    # QC
    no_mt = False
    if assay is None:
        adata.var['mt'] = adata.var_names.str.startswith(
            'MT-')  # annotate mitochondrial genes
    else:
        try:
            adata[assay].var['mt'] = adata[assay].var_names.str.startswith(
                'MT-')  # annotate mitochondrial genes
        except TypeError as err_mt:
            warnings.warn(f"\n\n{'=' * 80}\n\nCould not assign MT: {err_mt}")
            no_mt = True
    if no_mt is False:
        sc.pp.calculate_qc_metrics(adata[assay] if assay else adata, 
                                qc_vars=['mt'], percent_top=None, 
                                log1p=False, inplace=True)
    
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
        
    # Variable Genes
    if hvg_kws is not None:
        print("\n<<< DETECTING VARIABLE GENES >>>")
        sc.pp.highly_variable_genes(adata[assay] if assay else adata, 
                                    **hvg_kws)  # highly variable genes 
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
    return adata


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
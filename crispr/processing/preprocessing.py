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
import pandas as pd
import numpy as np

OPTS_OBJECT_TYPE = ["augur"]
regress_out_vars = ['total_counts', 'pct_counts_mt']

# hvg_kws = dict(min_mean=0.0125, max_mean=3, min_disp=0.5)
# target_sum=1e4
# max_genes_by_counts=2500
# max_pct_mt=5
# min_genes=200
# min_cells=3
# scale=10
# assay=None
# regress_out=regress_out_vars


def create_object(adata, object_type="augur"):
    """Create object(s) from adata."""
    
    # Check validity of object_type argument & modify as needed
    if isinstance(object_type, str):
        object_type = [object_type]
    if isinstance(object_type, (list, tuple, np.ndarray, set)):
        for i, j in enumerate(object_type):
            if not isinstance(j, str):
                raise ValueError(f"""object_type elements must be strings.""")
    else:
        raise ValueError("object_type must be a string or list of strings.")
    object_type = [t.lower() for t in object_type]  # convert to lowercase
    
    # Convert/create objects
    objects = [np.nan] * len(object_type)  # initialize empty list for output
    for i, t in enumerate(object_type):  # iterate object_types
        if t in OPTS_OBJECT_TYPE:  # if valid object type option, convert
            obj = adata
            objects[i] = obj
        else:  # if invalid object type option, warn & leave as nan in list
            raise Warning(f"""object_type {t} invalid or not yet implemented. 
                          Options: {OPTS_OBJECT_TYPE}.""")
    return objects


def create_object_scanpy(file, assay=None, target_sum=1e4, 
                         max_genes_by_counts=2500, max_pct_mt=5,
                         min_genes=200, min_cells=3, 
                         scale=10,  # or scale=True for no clipping
                         regress_out=regress_out_vars, hvg_kws=None):
    """Create object from scanpy."""
    
    # Load
    # extension = os.path.splitext(file)[1]
    if os.path.isdir(file):  # if directory, assume 10x format
        adata = sc.read_10x_mtx(file, var_names='gene_symbols', cache=True)
    else:
        adata = sc.read(file)
    # TODO: Flesh this out, generalize, test, etc.
    adata.var_names_make_unique() 
    
    # Normalize
    sc.pp.normalize_total(adata[assay] if assay else adata, 
                          target_sum=target_sum)  # count-normalize
    sc.pp.log1p(adata[assay] if assay else adata)  # log-normalize
    sc.pp.highly_variable_genes(adata[assay] if assay else adata, 
                                subset=True)  # highly variable genes

    
    # Filtering
    sc.pp.filter_cells(adata[assay] if assay else adata, min_genes=min_genes)
    sc.pp.filter_genes(adata[assay] if assay else adata, min_cells=min_cells)
    
    # QC
    adata.var['mt'] = adata.var_names.str.startswith(
        'MT-')  # annotate mitochondrial genes
    sc.pp.calculate_qc_metrics(adata[assay] if assay else adata, 
                               qc_vars=['mt'], percent_top=None, 
                               log1p=False, inplace=True)
    
    # More Filtering
    if assay is None:
        adata = adata[adata.obs.n_genes_by_counts < max_genes_by_counts, :]
        adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]
    else:
        adata[assay] = adata[assay][
            adata[assay].obs.n_genes_by_counts < max_genes_by_counts, :]
        adata[assay] = adata[assay][adata[assay].obs.pct_counts_mt < max_pct_mt, :]
    adata.raw = adata  # freeze normalized & filtered adata
        
    # Variable Genes
    if hvg_kws is not None:
        sc.pp.highly_variable_genes(adata, **hvg_kws)  # highly variable genes 
        adata.raw = adata  # freeze normalized & filtered adata
        adata = adata[:, adata.var.highly_variable]  # filter by HVGs
    
    # Regress Confounds
    if regress_out: 
        sc.pp.regress_out(adata[assay] if assay else adata, regress_out)
    
    # Scaling Genes
    if scale is not None:
        if scale is True:  # if True, just scale to unit variance
            sc.pp.scale(adata[assay] if assay else adata)  # scale
        else:  # if scale provided as an integer...
            sc.pp.scale(adata[assay] if assay else adata, 
                        max_value=scale)  # ...also clip values above "scale" SDs
    
    return adata
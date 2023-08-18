#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import pertpy as pt
import muon as mu
import pandas as pd
import scanpy as sc


def cluster(adata, assay=None, neighbor_metric="cosine"):
    """Perform PCA, UMAP, etc."""
    sc.pp.pca(adata[assay] if assay else adata)
    sc.pp.neighbors(adata[assay] if assay else adata, metric=neighbor_metric)
    sc.tl.umap(adata[assay] if assay else adata)
    sc.pl.umap(adata[assay] if assay else adata)
    try:
        adata_pert = adata[assay] if assay else adata.copy()
        adata_pert.X = adata_pert.layers['X_pert']
        sc.pl.umap(adata[assay] if assay else adata, 
                color=["replicate", "Phase", "perturbation"])
    except Exception:
        pass


def calculate_targeting_efficiency(adata, assay=None, guide_rna_column="NT"):
    """_summary_

    Args:
        adata (_type_): _description_
        assay (_type_, optional): _description_. Defaults to None.
        guide_rna_column (str, optional): _description_. Defaults to "NT".

    Returns:
        _type_: _description_
    """
    fig =  pt.pl.ms.barplot(adata[assay] if assay else adata, 
                            guide_rna_column=guide_rna_column)
    return fig


def calculate_perturbations():
    """Calculate perturbation scores."""
    pt.pl.ms.perturbscore(adata = mdata['rna'], labels='gene_target', target_gene='IFNGR2', color = 'orange')
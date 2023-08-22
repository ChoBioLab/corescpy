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


def cluster(adata, assay=None, neighbor_metric="cosine", plot=False):
    """Perform PCA, UMAP, etc."""
    figs = {}  # for figures
    sc.pp.pca(adata[assay] if assay else adata)
    sc.pp.neighbors(adata[assay] if assay else adata, metric=neighbor_metric)
    sc.tl.umap(adata[assay] if assay else adata)
    try:
        adata_pert = adata[assay] if assay else adata.copy()
        adata_pert.X = adata_pert.layers['X_pert']
            figs.update({"UMAP": sc.pl.umap(adata[assay] if assay else adata,
                    color=["replicate", "Phase", "perturbation"])})
    except Exception:
        pass

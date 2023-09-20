#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import pertpy as pt
import muon as mu
import warnings
import pandas as pd
import scanpy as sc


def cluster(adata, assay=None, plot=True, colors=None,
            paga=False,  # if issues with disconnected clusters, etc.
            method_cluster="leiden",
            kws_pca=None, kws_neighbors=None, kws_umap=None, kws_cluster=None):
    """Perform PCA, UMAP, etc."""
    figs = {}  # for figures
    kws_pca, kws_neighbors, kws_umap, kws_cluster = [
        {} if x is None else x for x in [
            kws_pca, kws_neighbors, kws_umap, kws_cluster]]
    if plot is True:
        try:
            figs["highest_counts_per_cell"] = sc.pl.highest_expr_genes(
                adata, n_top=20)
        except Exception as err:
            warnings.warn(f"Error plotting highest counts per cell: {err}")
    print("\n\n<<< PERFORMING PCA >>>")
    if len(kws_pca) > 0:
        print(kws_pca)
    sc.pp.pca(adata[assay] if assay else adata)
    print("\n\n<<< COMPUTING NEIGHBORHOOD GRAPH >>>")
    if len(kws_neighbors) > 0:
        print(kws_neighbors)
    sc.pp.neighbors(adata[assay] if assay else adata, 
                    **kws_neighbors)
    print(f"\n\n<<< EMBEDDING WITH UMAP >>>")
    if len(kws_umap) > 0:
        print(kws_umap)
    if paga is True:
        sc.tl.paga(adata)
        sc.pl.paga(adata, plot=False)  # plot=True for coarse-grained graph
        sc.tl.umap(adata, init_pos='paga', **kws_umap)
    else:
        sc.tl.umap(adata[assay] if assay else adata, **kws_umap)
    print(f"\n\n<<< CLUSTERING WITH {method_cluster.upper()} METHOD >>>")
    if str(method_cluster).lower() == "leiden":
        sc.tl.leiden(adata[assay] if assay else adata, **kws_cluster)  # leiden
    elif str(method_cluster).lower() == "leiden":  # louvain
        sc.tl.louvain(adata[assay] if assay else adata, **kws_cluster)
    else:
        raise ValueError("method_cluster must be 'leiden' or 'louvain'")
    print(f"\n\n<<< CREATING UMAP PLOTS >>>")
    if plot is True:
        try:
            figs["pca_variance_ratio"] = sc.pl.pca_variance_ratio(
                adata, log=True)  # scree-like plot for PCA components
        except Exception as err:
            warnings.warn(f"Failed to plot PCA variance ratio: {err}")
        try:
            figs["umap"] =  sc.pl.umap(adata[assay] if assay else adata, 
                                       color=method_cluster, 
                                       legend_loc='on data', title='', 
                                       frameon=False, save='.pdf')
        except Exception as err:
            warnings.warn(f"Failed to plot UMAP: {err}")
        if colors is not None:  # plot UMAP + extra color coding subplots
            try:
                figs["umap_extra"] = sc.pl.umap(adata[assay] if assay else adata, color=list(
                    pd.unique([method_cluster] + list(colors))))
            except Exception as err:
                warnings.warn(f"Failed to plot UMAP with extra colors: {err}")
        return figs
    return figs


def find_markers(adata, assay=None, plot=True, n_genes=25, method="wilcoxon"):
    """Find cluster gene markers."""
    figs = {}
    sc.tl.rank_genes_groups(adata, 'leiden', method=method)
    if plot is True:
        figs["marker_rankings"] = sc.pl.rank_genes_groups(
            adata, n_genes=n_genes, sharey=False)
    return adata.uns['rank_genes_groups'], figs
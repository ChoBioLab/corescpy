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
import celltypist
import pandas as pd
import scanpy as sc


def cluster(adata, assay=None, plot=True, colors=None,
            paga=False,  # if issues with disconnected clusters, etc.
            method_cluster="leiden",
            kws_pca=None, kws_neighbors=None, 
            kws_umap=None, kws_cluster=None, **kwargs):
    """
    Perform clustering and visualize results.
    
    Returns
    -------
    figs : dictionary of figures visualizing results
    
    adata: AnnData (adds fields to data)
        `.obsm['X_pca']`: Data as represented by PCA
        `.varm['PCs']`: Principal components loadings
        `.uns['pca']['variance_ratio']`: 
            Proportion of variance explained by each PCA component
        `.uns['pca']['variance']`: 
            Eigenvalues of covariance matrix 
                (variance expalined by PCA components).
    
    """
    figs = {}  # for figures
    # if "col_gene_symbols" in kwargs:
    #     col_gene_symbols = kwargs.pop("col_gene_symbols")
    # else:
    #     col_gene_symbols = None
    if kwargs:
        print(f"Un-used Keyword Arguments: {kwargs}")
    kws_pca, kws_neighbors, kws_umap, kws_cluster = [
        {} if x is None else x for x in [
            kws_pca, kws_neighbors, kws_umap, kws_cluster]]
    print("\n\n<<< PERFORMING PCA >>>")
    if len(kws_pca) > 0:
        print("\n", kws_pca)
    if "use_highly_variable" in kws_pca and "highly_variable" not in adata.var:
        warnings.warn("""use_highly_variable set to True, 
                      but 'highly_variable' not found in `adata.var`""")
        kws_pca["use_highly_variable"] = False
    sc.pp.pca(adata[assay] if assay else adata, **kws_pca)
    print("\n\n<<< COMPUTING NEIGHBORHOOD GRAPH >>>")
    if len(kws_neighbors) > 0:
        print("\n", kws_neighbors)
    sc.pp.neighbors(adata[assay] if assay else adata, 
                    **kws_neighbors)
    print(f"\n\n<<< EMBEDDING: UMAP >>>")
    if len(kws_umap) > 0:
        print("\n", kws_umap)
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
                adata[assay] if assay else adata, 
                log=True)  # scree-like plot for PCA components
        except Exception as err:
            warnings.warn(f"Failed to plot PCA variance ratio: {err}")
        try:
            figs["umap"] =  sc.pl.umap(
                adata[assay] if assay else adata, color=method_cluster, 
                legend_loc="on data", title="", frameon=False)  # UMAP plot
        except Exception as err:
            warnings.warn(f"Failed to plot UMAP: {err}")
        if colors is not None:  # plot UMAP + extra color coding subplots
            try:
                figs["umap_extra"] = sc.pl.umap(
                    adata[assay] if assay else adata, color=list(
                        pd.unique([method_cluster] + list(colors))))
            except Exception as err:
                warnings.warn(f"Failed to plot UMAP with extra colors: {err}")
        return figs
    return figs


def find_markers(adata, assay=None, col_cell_type="leiden", layer="scaled",
                 key_reference="rest", n_genes=25, method="wilcoxon", 
                 plot=True, **kwargs):
    """Find cluster gene markers."""
    figs = {}
    sc.tl.rank_genes_groups(adata, col_cell_type, method=method, 
                            reference=key_reference, 
                            key_added="rank_genes_groups", 
                            **kwargs)
    if plot is True:
        figs["marker_rankings"] = sc.pl.rank_genes_groups(
            adata, n_genes=n_genes, sharey=False)
    ranks = sc.get.rank_genes_groups_df(
        adata, None, key='rank_genes_groups', pval_cutoff=None, 
        log2fc_min=None, log2fc_max=None, gene_symbols=None)
    ranks = ranks.rename({"group": col_cell_type}, axis=1).set_index(
        [col_cell_type, "names"])
    return ranks, figs


def perform_celltypist(adata, model, col_cell_type=None, 
                       majority_voting=False, **kwargs):
    """Annotate cell types using CellTypist."""
    try:
        mod = celltypist.models.Model.load(
            model=model if ".pkl" in model else model + ".pkl")  # load model
    except Exception as err:
        print(f"{err}\n\nFailed to load CellTypist model {model}. Try:\n\n")
        print(celltypist.models.models_description())
    preds = celltypist.annotate(
        adata, model=model, majority_voting=majority_voting, **kwargs)  # run
    if col_cell_type is not None:  # compare to a different cell type label
        celltypist.dotplot(preds, use_as_reference=col_cell_type, 
                           use_as_prediction="predicted_labels")
    return preds


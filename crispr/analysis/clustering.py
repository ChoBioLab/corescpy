#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import warnings
import celltypist
from anndata import AnnData
import scanpy as sc
import os
import pandas as pd


def cluster(adata, layer=None,
            plot=True, colors=None,
            kws_celltypist=None,
            paga=False,  # if issues with disconnected clusters, etc.
            method_cluster="leiden", 
            resolution=1,
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
    ann = adata.copy()
    if layer:
        ann.X = adata.layers[layer].copy()  # set layer
    if kwargs:
        print(f"Un-used Keyword Arguments: {kwargs}")
    kws_pca, kws_neighbors, kws_umap, kws_cluster = [
        {} if x is None else x for x in [
            kws_pca, kws_neighbors, kws_umap, kws_cluster]]
    if kws_pca is not False:  # unless indicated not to run PCA
        if "use_highly_variable" not in kws_pca:  # default = use HVGs
            kws_pca["use_highly_variable"] = True
        print("\n\n<<< PERFORMING PCA >>>")
        if len(kws_pca) > 0:
            print("\n", kws_pca)
        if "use_highly_variable" in kws_pca and (
            "highly_variable" not in ann.var):
            warnings.warn("""use_highly_variable set to True, 
                        but 'highly_variable' not found in `adata.var`""")
            kws_pca["use_highly_variable"] = False
        sc.pp.pca(ann, **kws_pca)  # dimensionality reduction (PCA)
        print("\n\n<<< COMPUTING NEIGHBORHOOD GRAPH >>>\n"
            f"{kws_neighbors if kws_neighbors else ''}")
    sc.pp.neighbors(ann, **kws_neighbors)  # neighborhood
    print(f"\n\n<<< EMBEDDING: UMAP >>>")
    if kws_umap:
        print("\nUMAP Keywords:\n\n", kws_umap)
    if paga is True:
        sc.tl.paga(ann)
        sc.pl.paga(ann, plot=False)  # plot=True for coarse-grained graph
        sc.tl.umap(ann, init_pos="paga", **kws_umap)
    else:
        sc.tl.umap(ann, **kws_umap)
    print(f"\n\n<<< CLUSTERING WITH {method_cluster.upper()} METHOD >>>")
    if str(method_cluster).lower() == "leiden":
        sc.tl.leiden(ann, resolution=resolution, 
                     **kws_cluster)  # leiden clustering
    elif str(method_cluster).lower() == "louvain":
        sc.tl.louvain(ann, resolution=resolution, 
                      **kws_cluster)  # louvain clustering
    else:
        raise ValueError("method_cluster must be 'leiden' or 'louvain'")
    print(f"\n\n<<< CREATING UMAP PLOTS >>>")
    if plot is True:
        try:  # scree-like plot for PCA components
            figs["pca_var_ratio"] = sc.pl.pca_variance_ratio(ann, log=True)
        except Exception as err:
            warnings.warn(f"Failed to plot PCA variance ratio: {err}")
        try:  # plot UMAP by clusters
            figs["umap"] =  sc.pl.umap(
                ann, color=method_cluster, legend_loc="on data", 
                title="", frameon=False)  # UMAP plot
        except Exception as err:
            warnings.warn(f"Failed to plot UMAP: {err}")
        if colors is not None:  # plot UMAP + extra color coding subplots
            try:
                figs["umap_extra"] = sc.pl.umap(ann, color=list(pd.unique(
                    [method_cluster] + list(colors))))  # UMAP extra panels
            except Exception as err:
                warnings.warn(f"Failed to plot UMAP with extra colors: {err}")
    if kws_celltypist is not None:
        ann.uns["celltypist"], figs["celltypist"] = perform_celltypist(
            ann, **kws_celltypist)  # celltypist annotations
        ann.obs = ann.obs.join(ann.uns["celltypist"].predicted_labels, 
                               lsuffix="_last")  # to data
    return ann, figs


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
                       mode="best match", p_threshold=0.5, 
                       over_clustering=True, min_proportion=0, 
                       majority_voting=False, plot_markers=False,
                       kws_train=None, space=None, out_dir=None, **kwargs):
    """
    Annotate cell types using CellTypist.
    Provide string corresponding to CellTypist or, to train a custom 
    model based on other data, provide an AnnData object with training
    data (and, to provide further keyword arguments, including 
    `labels` if `col_cell_type` is not the same as for the new data, 
    also specify `kws_train`).
    """
    figs, kws_train = {}, kws_train if kws_train else {}
    jobs = kwargs.pop("n_jobs") if "n_jobs" in kwargs else os.cpu_count() - 1
    if isinstance(model, AnnData):  # anndata provided; train custom model
        if "n_jobs" not in kws_train:  # use cpus - 1 if # jobs unspecified
            kws_train["n_jobs"] = jobs
        if "col_cell_type" in kws_train:  # rename cell type argument if need
            kws_train["labels"] = kws_train.pop("col_cell_type")
        if "labels" not in kws_train and col_cell_type:
            kws_train["labels"] = col_cell_type
        print(f"\n\n<<< TRAINING CUSTOM CELLTYPIST MODEL >>>")
        if (mod.X[:1000].min() < 0) or (mod.X[:1000].max() > 9.22):
            print(f"*** Total-count & log-normalizing training data...")
            mod = mod.copy()
            sc.pp.normalize_total(mod, target_sum=1e4)
            sc.pp.log1p(mod)
        model = celltypist.train(model, **kws_train)  # custom model
    elif isinstance(model, str):  # model name provided
        try:
            model = celltypist.models.Model.load(
                model=model if ".pkl" in model else model + ".pkl")  # model
        except Exception as err:
            print(f"{err}\n\nNo CellTypist model: {model}. Try:\n\n")
            print(celltypist.models.models_description())
    else:  # CellTypist model object provided
            print(f"CellTypist model provided: {model}.")
    res = celltypist.annotate(
        adata, model=model, majority_voting=majority_voting, 
        p_thres=p_threshold, mode=mode, over_clustering=over_clustering, 
        min_prop=min_proportion, **kwargs)  # run
    if col_cell_type is not None:  # predicted-existing membership overlap
        for x in ["majority_voting", "predicted_labels"]:
            if x == "predicted_labels" or majority_voting is True:
                figs[f"label_transfer_{x}"] = celltypist.dotplot(
                    res, use_as_reference=col_cell_type, 
                    use_as_prediction=x, 
                    title=f"Label Transfer: {col_cell_type} vs. {x}")
    if out_dir:
        res.to_plots(out_file=out_dir, plot_probability=True)
        res.to_table(out_file=out_dir, plot_probability=True)
    # ann = res.to_adata(insert_labels=True, insert_prob=True)
    ann = res.to_adata(insert_labels=True)
    if col_cell_type is not None and plot_markers is True:  # markers
        figs["markers"] = {}
        for y in ["predicted_labels", "majority_voting"]:  # plot markers
            figs["markers"][y] = {}
            for x in ann.obs[y].unique():
                try:
                    markers = model.extract_top_markers(x, 3)
                    figs["markers"][y][f"markers_{x}"] = sc.pl.violin(
                        ann, markers, groupby=col_cell_type, rotation = 90)
                except Exception as err:
                    warnings.warn(f"{err}\n\n\nError in {y}={x} marker plot!")
                    figs["markers"][y][f"markers_{x}"] = err
    figs["label_transfer_mv_pl"] = celltypist.dotplot(
        res, use_as_reference="predicted_labels", 
        use_as_prediction="majority_voting", 
        title="Majority Voting versus Predicted Labels")  # mv vs. pl
    ctc = ["predicted_labels", "majority_voting"]  # celltypist columns
    ccts = set(pd.unique(ctc + list(col_cell_type if col_cell_type else []))
               ).intersection(ann.obs.columns)  # celltypist & original column
    if space is None:  # space b/t celltypist & cell type plot facets
        space = 0.75 if max([len(ann.obs[x].unique()) 
                             for x in ccts]) > 30 else 0.5
    figs["all"] = sc.pl.umap(ann, return_fig=True, legend_fontsize=6, 
                             color=list(ccts), wspace=space)  # all 1 plot
    return ann, res, figs


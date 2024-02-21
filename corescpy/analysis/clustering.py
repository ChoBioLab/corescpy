#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-exception-caught
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import warnings
import os
import re
from copy import deepcopy
import celltypist
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
# import rapids_singlecell as rsc
import pandas as pd
import numpy as np
import corescpy as cr


def cluster(adata, layer=None, method_cluster="leiden",
            paga=False,  # if issues with disconnected clusters, etc.
            resolution=1, n_comps=None, use_highly_variable=True,
            kws_pca=None, kws_neighbors=None, kws_umap=None, kws_cluster=None,
            genes_subset=None, seed=0, use_gpu=False,  kws_celltypist=None,
            plot=True, colors=None, **kwargs):
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
    ann = adata.copy()
    if layer:
        print(f"\n\n*** Using layer: {layer}.\n\n")
        ann.X = adata.layers[layer].copy()  # set layer
    if ann.var.index.values[0] not in ann.var_names:
        raise ValueError("`adata.var_names` must be index of `.var`.")
    if isinstance(kws_pca, dict):
        for k, x in zip(["n_comps", "use_highly_variable"],
                        [n_comps, use_highly_variable]):
            if k in kws_pca and x != kws_pca[k]:
                raise ValueError(f"Can't use `{k}` & `kws_pca['{k}']`.")
        kws_pca.update(dict(use_highly_variable=use_highly_variable,
                            n_comps=n_comps))
    elif kws_pca is not False:
        kws_pca = dict(use_highly_variable=use_highly_variable,
                       n_comps=n_comps)
    else:
        print(f"\n\n`kws_pca`=False. Using existing if present:\n\n{ann}\n\n")
    if kws_pca is not False:
        kws_pca["random_state"] = seed  # seed to PCA arguments
        if use_highly_variable is True and "highly_variable" not in ann.var:
            warnings.warn("`use_highly_variable`=True & 'highly_variable'"
                          " not in `.var`. Setting to False.")
            kws_pca["use_highly_variable"] = False
    kws_neighbors, kws_umap, kws_cluster = [cr.tl.merge({
        "random_state": seed}, x) for x in [
            kws_neighbors, kws_umap, kws_cluster]]  # seed->arguments; None={}
    kws_neighbors["n_pcs"] = n_comps  # components for neighbors = for PCA
    if kwargs:
        print(f"\n\nUn-used Keyword Arguments: {kwargs}")

    # Dimensionality Reduction (PCA)
    if kws_pca is not False:  # unless indicated not to run PCA
        print(f"\n\n<<< PERFORMING PCA >>>\n{kws_pca}\n")
        ann_use = ann[:, ann.var_names.isin(genes_subset)] if (
            genes_subset not in [None, False]) else ann  # data for PCA
        sc.pp.pca(ann_use, **kws_pca)  # run PCA
        if genes_subset not in [None, False]:  # subsetted genes data -> full
            ann = cr.tl.merge_pca_subset(ann, ann_use, retain_cols=False)
        else:  # if used full gene set
            ann = ann_use

    # Neighborhood Graph
    print(f"\n\n<<< COMPUTING NEIGHBORHOOD GRAPH >>>\n{kws_neighbors}\n")
    sc.pp.neighbors(ann, **kws_neighbors)  # compute neighborhood graph

    # UMAP Embedding
    print(f"\n\n<<< EMBEDDING UMAP >>>\n{kws_umap}\n")
    if use_gpu is True:
        raise NotImplementedError("GPU-accelerated UMAP not yet implemented.")
        # rsc.tl.umap(ann, **kws_umap)  # UMAP with rapids (GPU-accelerated)
    else:
        sc.tl.umap(ann, **kws_umap)  # vanilla Scanpy UMAP

    # Clustering with Leiden or Louvain
    print(f"\n\n<<< CLUSTERING WITH {method_cluster.upper()} METHOD >>>")
    if method_cluster == "leiden":  # Leiden clustering
        if use_gpu is True:
            raise NotImplementedError("GPU-acceleration not yet supported.")
            # rsc.tl.leiden(ann, resolution=resolution, **kws_cluster)
        else:
            sc.tl.leiden(ann, resolution=resolution, **kws_cluster)
    elif method_cluster == "louvain":  # Louvain clustering
        if use_gpu is True:
            raise NotImplementedError("GPU-acceleration not yet supported.")
            # rsc.tl.louvain(ann, resolution=resolution, **kws_cluster)
        else:
            sc.tl.louvain(ann, resolution=resolution, **kws_cluster)
    else:
        raise ValueError("method_cluster must be 'leiden' or 'louvain'")

    # PAGA Correction (Optional)
    if paga is True:  # recompute with PAGA (optional)
        print("\n\n<<< PERFORMING PAGA >>>")
        sc.tl.paga(ann, groups=method_cluster)
        if plot is True:
            sc.pl.paga(ann, plot=False)  # plot=True for coarse-grained graph
        if use_gpu is True:  # GPU-accelerated UMAP
            raise NotImplementedError("Not yet implemented: GPU UMAP.")
            # rsc.tl.umap(ann, **kws_umap, init_pos="paga")  # UMAP with PAGA
        else:  # Vanilla UMAP
            sc.tl.umap(ann, **kws_umap, init_pos="paga")  # UMAP with PAGA

    # Plotting
    print("\n\n<<< CREATING PLOTS >>>")
    if plot is True:
        try:  # scree-like plot for PCA components
            sc.pl.pca_variance_ratio(ann, log=True)
            figs["pca_var_ratio"] = plt.gcf()
        except Exception as err:
            warnings.warn(f"Failed to plot PCA variance ratio: {err}")
        try:  # plot UMAP by clusters
            figs["umap"] = sc.pl.umap(ann, color=method_cluster, title="",
                                      frameon=False, legend_loc="on data")
        except Exception as err:
            warnings.warn(f"Failed to plot UMAP: {err}")
        if colors is not None:  # plot UMAP + extra color coding subplots
            try:
                figs["umap_extra"] = sc.pl.umap(ann, color=list(pd.unique(
                    [method_cluster] + list(colors))), wspace=(
                        len(colors) + 1) * 0.075)  # UMAP extra panels
            except Exception as err:
                warnings.warn(f"Failed to plot UMAP with extra colors: {err}")
    return ann, figs


def find_marker_genes(adata, assay=None, col_cell_type="leiden", n_genes=5,
                      key_reference="rest", layer="log1p", p_threshold=None,
                      col_gene_symbols=None, method="wilcoxon", kws_plot=True,
                      use_raw=False, key_added="rank_genes_groups", **kwargs):
    """Find cluster gene markers."""
    figs = {}
    if kws_plot is True:
        kws_plot = {}
    if assay:
        adata = adata[assay]
    if layer:
        adata.X = adata.layers[layer].copy()  # change anndata layer if need
    sc.tl.rank_genes_groups(
        adata, col_cell_type, method=method, reference=key_reference,
        key_added=key_added, use_raw=use_raw, **kwargs)  # rank
    if isinstance(kws_plot, dict):
        figs["marker_rankings"] = cr.pl.plot_markers(
            adata, n_genes=n_genes, key_added=key_added, use_raw=use_raw,
            key_reference=key_reference, **{"col_wrap": 3, **kws_plot})
    ranks = sc.get.rank_genes_groups_df(
        adata, None, key=key_added, pval_cutoff=p_threshold,
        log2fc_min=None, log2fc_max=None, gene_symbols=col_gene_symbols)
    ranks = ranks.rename({"group": col_cell_type}, axis=1).set_index(
        [col_cell_type, "names"])  # format ranking dataframe
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
    n_jobs = kwargs.pop("n_jobs", 1)  # number of CPUs to use
    ctc = ["predicted_labels", "majority_voting"]  # celltypist columns

    # Load or Train Model
    if isinstance(model, AnnData):  # if anndata provided; train custom model
        if "n_jobs" not in kws_train:  # use cpus - 1 if # jobs unspecified
            kws_train["n_jobs"] = n_jobs
        if "col_cell_type" in kws_train:  # rename cell type argument if need
            kws_train["labels"] = kws_train.pop("col_cell_type")
        if "labels" not in kws_train and col_cell_type:
            kws_train["labels"] = col_cell_type
        print("\n\n<<< TRAINING CUSTOM CELLTYPIST MODEL >>>")
        mod = model.copy()
        # if cr.pp.check_normalization(mod) is False:
        print("*** Total-count & log-normalizing training data...")
        sc.pp.normalize_total(mod, target_sum=1e4)
        sc.pp.log1p(mod)
        model = celltypist.train(mod, **kws_train)  # custom model
    elif isinstance(model, str):  # if model name provided
        try:
            model = celltypist.models.Model.load(
                model=model if ".pkl" in model else model + ".pkl")  # model
        except Exception as err:
            print(f"{err}\n\nNo CellTypist model: {model}. Try:\n\n")
            print(celltypist.models.models_description())
    else:  # if CellTypist model object provided
        print(f"CellTypist model provided: {model}.")

    # Annotate Cells with CellTypist
    res = celltypist.annotate(
        adata, model=model, majority_voting=majority_voting,
        p_thres=p_threshold, mode=mode, over_clustering=over_clustering,
        min_prop=min_proportion, **kwargs)  # run celltypist
    if out_dir:  # save results?
        pass
        # res.to_plots(out_file=out_dir, plot_probability=True)  # save plots
        # res.to_table(out_file=out_dir, plot_probability=True)  # save tables
    # ann = res.to_adata(insert_labels=True, insert_prob=True)
    ann = res.to_adata(insert_labels=True)  # results object -> anndata

    # Plot Label Transfer (Pre-Existing Annotations vs. CellTypist)
    if col_cell_type not in [None] + ctc:  # plot membership overlap
        for x in ["majority_voting", "predicted_labels"]:
            dot = x == "predicted_labels" or (majority_voting is True and (
                col_cell_type != x))  # plot on this iteration?
            if dot:  # label transfer dot plot if appropriate
                figs[f"label_transfer_{x}"] = celltypist.dotplot(
                    res, use_as_reference=col_cell_type, use_as_prediction=x,
                    title=f"Label Transfer: {col_cell_type} vs. {x}",
                    cmap="magma")  # label transfer dot plot

    # Plot Markers
    if col_cell_type is not None and plot_markers is True:  # markers
        figs["markers"] = {}
        for y in ctc:  # plot markers
            figs["markers"][y] = {}
            for x in ann.obs[y].unique():
                try:
                    markers = model.extract_top_markers(x, 3)
                    figs["markers"][y][f"markers_{x}"] = sc.pl.violin(
                        ann, markers, groupby=col_cell_type, rotation=90)
                except Exception as err:
                    warnings.warn(f"{err}\n\n\nError in {y}={x} marker plot!")
                    figs["markers"][y][f"markers_{x}"] = err

    # Plot Label Transfer (Majority Voting vs. Predicted Labels)
    figs["label_transfer_mv_pl"] = celltypist.dotplot(
        res, use_as_reference=ctc[0], use_as_prediction=ctc[1],
        title="Majority Voting versus Predicted Labels",
        cmap="magma")  # mv vs. pl dot plot

    # Plot UMAP
    ccts = set(pd.unique(ctc + list(col_cell_type if col_cell_type else []))
               ).intersection(ann.obs.columns)  # celltypist & original column
    if space is None:  # space b/t celltypist & cell type plot facets
        cats = max([len(ann.obs[x].unique()) for x in ccts])
        space = 0.2 * int(cats / 15) if cats > 15 else 0.5
    figs["all"] = sc.pl.umap(ann, return_fig=True, legend_fontsize=6,
                             color=list(ccts), wspace=space)  # all 1 plot

    # Plot Confidence Scores
    if "majority_voting" in ann.obs:  # if did over-clustering/majority voting
        conf = ann.obs[["majority_voting", "predicted_labels", "conf_score"
                        ]].set_index("conf_score").stack().rename_axis(
                            ["Confidence Score", "Annotation"]).to_frame(
                                "Label").reset_index()  # scores ~ label

        aspect = int(len(conf[conf.Annotation == "predicted_labels"
                              ].Label.unique()) / 15)  # aspect ratio
        figs["confidence"] = sns.catplot(
            data=conf, y="Confidence Score", row="Annotation", height=40,
            aspect=aspect, x="Label", hue="Label", kind="violin")  # plot
        figs["confidence"].figure.suptitle("CellTypist Confidence Scores")
        for a in figs["confidence"].axes.flat:
            _ = a.set_xticklabels(a.get_xticklabels(), rotation=90)
        figs["confidence"].fig.show()
    return ann, res, figs


def annotate_by_markers(adata, data_assignment, method="overlap_count",
                        col_assignment="Type", n_top=20,
                        col_cell_type="leiden", col_new="Annotation",
                        renaming=False, **kwargs):
    """
    Annotate based on markers (adapted from Squidpy tutorial).

    The argument `data_assignment` should be specified as a dataframe
    indexed by gene symbols and a single column of assignments to cell
    types for each marker (or a file path returning the same).
    """
    adata = adata.copy()
    col_bc = adata.obs.index.names[0]
    if col_new in adata.obs:
        raise ValueError(f"`col_new ({col_new}) already exists in adata.obs!")

    # Load Marker Groups
    if isinstance(data_assignment, (str, os.PathLike)):
        data_assignment = pd.read_excel(data_assignment, index_col=0)
    assign = data_assignment.copy()
    if renaming is True:
        sources = assign[col_assignment].unique()
        rename = dict(zip(sources, [" ".join([i.capitalize() if i and i[
            0] != "(" and not i.isupper() and i not in [
                "IgG", "IgA"] else i for i in x.split(" ")]) if len(x.split(
                    " ")) > 1 else x for x in [re.sub("glia", "Glia", re.sub(
                        "_", " ", j)) for j in sources]]))
        assign.loc[:, col_assignment] = assign[col_assignment].replace(rename)
    if method.lower() in ["overlap_count", "overlap_coef", "jaccard"]:
        assign = assign.rename_axis("Gene")
        assign.columns = [col_assignment]
        assign = dict(assign.reset_index().groupby(col_assignment).apply(
            lambda x: list(pd.unique(x.Gene))))  # to marker dictionary
        overlap = sc.tl.marker_gene_overlap(
            adata, assign, method=method,
            top_n_markers=n_top, **kwargs)  # overlap scores
        overlap = overlap.T.join(overlap.apply(lambda x: overlap.index.values[
            np.argmax(x)]).to_frame(col_new))
        return adata, overlap
    else:
        nrow = assign.shape[0]
        if assign.reset_index().iloc[:, 0].duplicated().any():
            assign = assign.reset_index()
            assign = assign[~assign.iloc[:, 0].duplicated()]
            assign = assign.set_index(list(assign.columns)[0])
            print(f"Dropping {assign.shape[0]} duplicate genes of {nrow}.")
        assign.index.name = None
        assign.columns = [col_assignment]

        # Assign marker gene metadata using reference dataset
        meta_gene = deepcopy(adata.var)
        shared_marks = list(set(meta_gene.index.tolist()).intersection(
            assign[col_assignment].index.tolist()))  # available genes
        meta_gene.loc[shared_marks, "Markers"] = assign.loc[
            shared_marks, col_assignment]

        # Calculate Average Expression by Cluster
        ser_counts = adata.obs[col_cell_type].value_counts()
        ser_counts.name = "cell counts"
        meta_c = pd.DataFrame(ser_counts)
        sig_cl = pd.DataFrame(columns=adata.var_names, index=adata.obs[
            col_cell_type].cat.categories)  # cell types (rows) ~ genes
        for c in adata.obs[col_cell_type].cat.categories:  # iterate cluters
            sig_cl.loc[c] = adata[adata.obs[col_cell_type].isin(
                [c]), :].X.mean(0)
        sig_cl = sig_cl.transpose()
        leiden = [f"{col_cell_type}-" + str(x)
                  for x in sig_cl.columns.tolist()]
        sig_cl.columns = leiden
        meta_c.index = sig_cl.columns.tolist()
        meta_c[col_cell_type] = pd.Series(
            meta_c.index.tolist(), index=meta_c.index.tolist())
        meta_gene = pd.DataFrame(index=sig_cl.index.tolist())
        meta_gene["info"] = pd.Series("", index=meta_gene.index.tolist())
        meta_gene["Markers"] = pd.Series("N.A.", index=sig_cl.index.tolist())
        meta_gene.loc[shared_marks, "Markers"] = assign.loc[
            shared_marks, col_assignment]

        # Assign Cell Types
        meta_c[col_new] = pd.Series("N.A.", index=meta_c.index.tolist())
        for inst_cluster in sig_cl.columns.tolist():
            top_genes = (sig_cl[inst_cluster].sort_values(
                ascending=False).index.tolist()[:n_top])
            inst_ser = meta_gene.loc[top_genes, "Markers"]
            inst_ser = inst_ser[inst_ser != "N.A."]
            ser_counts = inst_ser.value_counts()
            max_count = ser_counts.max()
            max_cat = "_".join(sorted(ser_counts[
                ser_counts == max_count].index.tolist()))
            meta_c.loc[inst_cluster, col_new] = max_cat
        meta_c["name"] = meta_c.apply(
            lambda x: x[col_new] + "_" + x[col_cell_type], axis=1)  # rename
        leiden_names = meta_c["name"].values.tolist()
        meta_c.index = leiden_names
        leiden_to_cell_type = deepcopy(meta_c)
        leiden_to_cell_type.set_index(col_cell_type, inplace=True)
        leiden_to_cell_type.index.name = None
        adata.obs[col_new] = adata.obs["leiden"].apply(
            lambda x: leiden_to_cell_type.loc[
                f"{col_cell_type}-{x}", col_new])
        adata.obs[f"{col_new}_Cluster"] = adata.obs["leiden"].apply(
            lambda x: leiden_to_cell_type.loc[
                f"{col_cell_type}-" + str(x), "name"])
        if col_bc in adata.obs:
            adata.obs = adata.obs.set_index(col_bc)
        leiden_to_cell_type.index = ["-".join(x.split("-")[
            1:]) for x in leiden_to_cell_type.index.values]
        print(leiden_to_cell_type[col_new])
        return adata, leiden_to_cell_type

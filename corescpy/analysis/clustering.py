#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=broad-exception-caught
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

from warnings import warn
import os
import re
from copy import deepcopy
import seaborn as sns
import celltypist
from anndata import AnnData
import scanpy as sc
# import rapids_singlecell as rsc
import pandas as pd
import numpy as np
import corescpy as cr


def cluster(adata, layer=None, method_cluster="leiden", key_added=None,
            paga=False,  # if issues with disconnected clusters, etc.
            resolution=1, n_comps=None, use_highly_variable=True,
            kws_pca=None, kws_neighbors=None, kws_umap=None, kws_cluster=None,
            genes_subset=None, seed=0, use_gpu=False,
            restrict_to=None,  # for subclustering (column, keys in column)
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
    ann = adata.copy()
    if use_gpu is True:
        raise NotImplementedError("GPU-accelerated UMAP not yet implemented.")
    # pkg = rsc if use_gpu is True else sc  # Scanpy or Rapids?
    pkg = sc  # Scanpy, because Rapids not yet implemented
    if method_cluster not in ["leiden", "louvain"]:
        raise ValueError("`method_cluster` must be 'leiden' or 'louvain'.")
    if key_added is None:
        key_added = method_cluster
    if layer:
        print(f"\n\n*** Using layer: {layer}.\n\n")
        ann.X = adata.layers[layer].copy()  # set layer
    if ann.var.index.values[0] not in ann.var_names:
        raise ValueError("`adata.var_names` must be index of `.var`.")
    kws_pca_d = dict(use_highly_variable=use_highly_variable, n_comps=n_comps,
                     random_state=seed)  # start with PCA-specific arguments
    if kws_pca is not False:  # use or merge with main PCA keywords
        if isinstance(kws_pca, dict):  # if specified additional PCA arguments
            for k, x in zip(["n_comps", "use_highly_variable"],
                            [n_comps, use_highly_variable]):
                if k in kws_pca and x != kws_pca[k]:
                    raise ValueError(f"Can't use `{k}` & `kws_pca['{k}']`.")
        kws_pca = cr.tl.merge(kws_pca_d, {} if kws_pca is True else kws_pca)
        if use_highly_variable is True and "highly_variable" not in ann.var:
            warn("'highly_variable' not in `.var`. Setting to False.")
            kws_pca["use_highly_variable"] = False
    else:  # if kws_pca = False, use existing (e.g., for integrated samples)
        print(f"\n\n`kws_pca`=False. Using existing if present:\n\n{ann}\n\n")
    kws_neighbors, kws_umap, kws_cluster = [cr.tl.merge({
        "random_state": seed}, x) for x in [
            kws_neighbors, kws_umap, kws_cluster]]  # seed->arguments; None={}
    kws_cluster["restrict_to"] = restrict_to  # subclustering?
    kws_cluster["key_added"] = key_added  # column in which to store clusters
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
    # pkg = sc if use_gpu is False else rsc  # package
    pkg.tl.umap(ann, **kws_umap)  # UMAP

    # Clustering with Leiden or Louvain
    print(f"\n\n<<< CLUSTERING WITH {method_cluster.upper()} METHOD >>>"
          f"\nResolution={resolution}")
    f_x = pkg.tl.leiden if method_cluster == "leiden" else pkg.tl.louvain
    f_x(ann, resolution=resolution, **kws_cluster)  # clustering

    # PAGA Correction (Optional)
    if paga is True:  # recompute with PAGA (optional)
        print("\n\n<<< PERFORMING PAGA >>>")
        sc.tl.paga(ann, groups=method_cluster)
        if plot is True:
            sc.pl.paga(ann, plot=False)  # plot=True for coarse-grained graph
        pkg.tl.umap(ann, **kws_umap, init_pos="paga")  # UMAP with PAGA

    # Plotting
    print(f"\n\n<<< {'CREATING' if plot is True else 'SKIPPING'} PLOTS >>>")
    figs = cr.pl.plot_clustering(ann, method_cluster, colors=colors) if (
        plot is True) else {}  # plot, or just empty dictionary for `figs`
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
    ranks = make_marker_genes_df(
        adata, col_cell_type, key_added=key_added, p_threshold=p_threshold,
        log2fc_min=None, log2fc_max=None, gene_symbols=col_gene_symbols)
    return ranks, figs


def make_marker_genes_df(adata, col_cell_type, key_added="leiden",
                         p_threshold=None, **kwargs):
    """Make marker gene dictionary in `.uns` into a dataframe."""
    ranks = sc.get.rank_genes_groups_df(adata, None, key=key_added,
                                        pval_cutoff=p_threshold, **kwargs)
    ranks = ranks.rename({"group": col_cell_type}, axis=1).set_index(
        [col_cell_type, "names"])  # format ranking dataframe
    return ranks


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
                    warn(f"{err}\n\n\nError in {y}={x} marker plot!")
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
    key_add = kwargs.pop("key_added", f"rank_genes_groups_{col_cell_type}")

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
    if col_assignment in assign.columns:
        assign = assign[[col_assignment]]
    if key_add not in adata.uns:
        cr.ax.find_marker_genes(
            adata, col_cell_type=col_cell_type, n_genes=n_top, layer="log1p",
            p_threshold=None, method="wilcoxon", kws_plot=False,
            use_raw=False, key_added=key_add, **kwargs)  # find marker genes
    if method.lower() in ["overlap_count", "overlap_coef", "jaccard"]:
        assign = assign.rename_axis("Gene")
        assign.columns = [col_assignment]
        assign = dict(assign.reset_index().groupby(col_assignment).apply(
            lambda x: list(pd.unique(x.Gene))))  # to marker dictionary
        k_a = f"marker_gene_overlap__{key_add}"
        overlap = sc.tl.marker_gene_overlap(
            adata, assign, method=method, key=key_add, key_added=k_a,
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


def print_marker_info(adata, key_cluster, assign, col_cell_type=None,
                      key_added="rank_genes_groups", col_annotation=None,
                      layer="counts", count_threshold=1, p_threshold=1,
                      lfc_threshold=None, n_top_genes=20, key_compare=None,
                      show=False, print_threshold=25, **kwargs):
    """
    Print frequencies at which a cluster expresses at least
    <count_threshold> transcripts of genes in its top marker list,
    sorted by cell type annotations linked to those genes,
    provided a dataframe (`assign`) with genes as the index and
    linked annotations in <col_annotation> (1st if unspecified).
    If `n_top_genes` is a list of genes, will use those instead of
    top markers. If `key_compare` is specified, will return
    in the 'Percent_Total' column of the 2nd element of the output
    percentages of total cells in that comparison group
    and `key_cluster` rather than of all total cells.
    """
    c_t, kmk = col_cell_type, key_added  # for brevity
    ann = adata.copy()
    if layer is not None:
        adata.X = adata.layers[layer]
    if col_annotation is None:
        col_annotation = assign.columns[0]
    mks = kwargs.pop("marker_genes_df", cr.ax.make_marker_genes_df(
        ann, c_t, key_added=kmk))  # DEGs
    mks = mks[mks.pvals_adj <= p_threshold]  # filter by p-value
    if lfc_threshold is not None:
        mks = mks[mks.logfoldchanges >= lfc_threshold]  # filter by LFC
    mks = mks.groupby(c_t).apply(
        lambda x: pd.Series([np.nan]) if x.name not in x.index else x.loc[
            x.name].loc[n_top_genes] if isinstance(
                n_top_genes, list) else x.loc[x.name].iloc[:min(x.shape[
                    0], n_top_genes)])  # # top or pre-specified genes
    mks_grps = assign.loc[mks.loc[key_cluster].index.intersection(
        assign.index)].rename_axis("Gene")[[col_annotation]]  # only DEGs

    # Percent of Cluster's Cells Reaching GEX Threshold (Specificity)
    percs_exp = mks_grps.groupby("Gene").apply(
        lambda x: 100 * np.mean(ann[ann.obs[c_t] == key_cluster][
            :, x.name].X >= count_threshold))  # % cluster cell GEX>=threshold
    percs_exp = mks_grps[col_annotation].str.get_dummies(',').groupby(
        "Gene").max().groupby("Gene").apply(lambda g: g.replace(
            1, percs_exp.loc[g.name])).replace(0, "").reset_index(
                0, drop=True)  # rows=genes, columns=annotation 1s if present

    # Calculate Percent of GEX-Threshold+ Cells that are in Cluster
    # Using All Other Clusters' Cells, or Just Comparison Group (Sensitivity)
    kcn = "|".join([key_compare] if isinstance(
        key_compare, str) else key_compare) if key_compare else "Other"
    if key_compare is None or isinstance(key_compare, str):
        key_compare = [key_compare] if isinstance(
            key_compare, str) else list(ann.obs[c_t].unique())
    key_compare = list(set(key_compare).difference([key_cluster]))

    # Number of Cells by Cluster >= GEX Threshold
    n_exp = [mks_grps.groupby("Gene").apply(lambda x: np.sum(ann[
        subs][:, x.name].X >= count_threshold)) for subs in [ann.obs[
            c_t] == key_cluster, ann.obs[c_t].isin(key_compare)]]  # number
    n_exp = pd.concat(n_exp, keys=[key_cluster, kcn]).unstack(0)
    if n_exp.empty is False:
        n_exp = n_exp.join(n_exp.T.sum().to_frame("Total"))  # total >=
        n_exp = n_exp.assign(Percent_Total=100 * n_exp[
            key_cluster] / n_exp[
                "Total"])  # % of all cells with gene that are in cluster

    # % of All (or Comparison Group) GEX-Threshold+ Cells in Reference Cluster
    if n_exp.empty is False and n_exp[
            n_exp.Percent_Total >= print_threshold].empty is False:
        perc_rep = "Represents " + ", ".join(n_exp[
            n_exp.Percent_Total >= print_threshold].sort_values(
                "Percent_Total", ascending=False).groupby("Gene").apply(
                    lambda x: str(int(x["Percent_Total"])) + "%" + str(
                        f" of all {x.name}+ cells")))  # string description
    else:
        perc_rep = ""

    # Genes Reaching Threshold in Cluster, Sorted by Percent Positivity
    percs = percs_exp.stack().replace("", np.nan).dropna().reset_index(
        1, drop=True).drop_duplicates().sort_values(ascending=False)

    # Descriptive Messages & Display
    pos_rate = "; ".join(percs[percs >= print_threshold].reset_index(
        ).apply(lambda x: f"{int(x.iloc[1])}% {x['Gene']}+", axis=1)
                            ) + f" (>={count_threshold} counts)"
    msg = ("" if perc_rep == "" else perc_rep + ". ") + pos_rate  # describe
    genes = mks_grps.reset_index().groupby(col_annotation).apply(
        lambda x: ", ".join(x.Gene.unique()))  # markers ~ annotation
    print(f"\n{'=' * 80}\nCount Threshold: {count_threshold}\n{'=' * 80}")
    if show is True:  # print results?
        print(genes)
        print(percs_exp.applymap(lambda x: x if x == "" else str(int(x))))
        print(n_exp.applymap(lambda x: x if x == "" else str(int(x))))
    return percs_exp, percs, n_exp, genes, msg

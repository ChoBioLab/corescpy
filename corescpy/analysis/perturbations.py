#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Analyzing CRISPR experiment data.

@author: E. N. Aslinger
"""

import pertpy as pt
import scanpy as sc
from warnings import warn
import decoupler
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import traceback
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import os
# import blitzgsea as blitz
# from copy import deepcopy
import corescpy as cr
import pandas as pd
import numpy as np

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"
layer_perturbation = "X_pert"
ifn_pathways_default = [
    "REACTOME_INTERFERON_SIGNALING",
    "REACTOME_INTERFERON_ALPHA_BETA_SIGNALING",
    "REACTOME_INTERFERON_GAMMA_SIGNALING"]


def perform_mixscape(adata, assay=None, assay_protein=None, layer=None,
                     protein_of_interest=None, col_perturbed="perturbation",
                     key_control="NT", key_treatment="perturbed",
                     key_nonperturbed="NP", col_guide_rna="guide_ID",
                     col_split_by=None, col_target_genes="gene_target",
                     iter_num=10, min_de_genes=5, pval_cutoff=5e-2,
                     logfc_threshold=0.25, subsample_number=300,
                     n_comps_lda=10, guide_split="-", feature_split="|",
                     target_gene_idents=None, kws_perturbation_signature=None,
                     plot=True, **kwargs):
    """
    Identify perturbed cells based on target genes
    (`adata.obs['mixscape_class']`,
    `adata.obs['mixscape_class_global']`) and calculate posterior
    probabilities (`adata.obs['mixscape_class_p_<key_treatment>']`,
    and perturbation scores.

    Optionally, perform LDA to cluster cells based on perturbation
    response and gene expression jointly.
    Optionally, create figures related to differential gene
    (and protein, if available) expression, perturbation scores,
    and perturbation response-based clusters.

    Runs a differential expression analysis and creates a heatmap
    sorted by the posterior probabilities.

    Args:
        adata (AnnData): Scanpy data object.
        layer_perturbation (str, optional): Layer in `adata` that
            contains the data you want to use for the
            perturbation analysis.
        col_perturbed (str): Perturbation category column of
            `adata.obs` (should contain key_control).
        assay (str, optional): Assay slot of adata
            ('rna' for `adata['rna']`).
            Defaults to None (works if only one assay).
        assay_protein (str, optional): Protein assay slot name
            (if available). Defaults to None.
        key_control (str, optional): The label in `col_perturbed`
            that indicates control condition. Defaults to "NT".
        key_treatment (str, optional): The label in `col_perturbed`
            that indicates a treatment condition (e.g., drug
            administration, CRISPR knock-out/down). Defaults to "KO".
        col_split_by (str, optional): `adata.obs` column name of
            sample categories to calculate separately
            (e.g., replicates). Defaults to None.
        col_target_genes (str, optional): Name of column with target
            genes. Defaults to "gene_target".
        protein_of_interest (str, optional): If assay_protein is not
            None and plot is True, will allow creation of violin plot
            of protein expression (y) by
            <target_gene_idents> perturbation category (x),
            split/color-coded by Mixscape classification
            (`adata.obs['mixscape_class_global']`). Defaults to None.
        col_guide_rna (str, optional): Name of column with guide
            RNA IDs (full). Format may be something like
            STAT1-1|CNTRL-2-1. Defaults to "guide_ID".
        guide_split (str, optional): Guide RNA ID # split character
            before guide #(s) (as in "-" for "STAT3-1-2").
            Same as used in Crispr/crispr_class.py process guide RNA
            method. Defaults to "-".
        target_gene_idents (list or bool, optional): List of names of
            genes whose perturbations will determine cell grouping
            for the above-described violin plot and/or
            whose differential expression posterior probabilities
            will be plotted in a heatmap. Defaults to None.
            True to plot all in `adata.uns["mixscape"]`.
        min_de_genes (int, optional): Minimum number of genes a cell
            has to express differentially to be labeled 'perturbed'.
            For Mixscape and LDA (if applicable). Defaults to 5.
        pval_cutoff (float, optional): Threshold for significance
            to identify differentially-expressed genes.
            For Mixscape and LDA (if applicable). Defaults to 5e-2.
        logfc_threshold (float, optional): Will only test genes whose
            average logfold change across the two cell groups is at
            least this number. For Mixscape and LDA (if applicable).
            Defaults to 0.25.
        n_comps_lda (int, optional): Number of principal components
            (e.g., 10) for PCA xperformed as part of LDA for pooled
            CRISPR screen data. Defaults to 10.
        iter_num (float, optional): Iterations to run to converge
            if needed.
        plot (bool, optional): Make plots?
            Defaults to True.
        kws_perturbation_signature (dict, optional): Optional keyword
            arguments to pass to `pertpy.tl.PerturbationSignature()`
            (also see Pertpy documentation). "n_neighbors"
            (# of unperturbed neighbors to use for comparison
            when calculating perturbation signature),
            "n_pcs", "use_rep" (`X` or any `.obsm` keys),
            "batch_size" (if None, use full data, which is
                memory-intensive, or specify an integer to calculate
                signature in batches,
                which is inefficient for sparse data).
    """
    figs = {}
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    if kws_perturbation_signature is None:
        kws_perturbation_signature = {}
    key_nonperturbed = "NP"

    # Perturbation Signature
    adata_pert = (adata[assay] if assay else adata).copy()
    if layer is not None:
        adata_pert.X = adata_pert.layers[layer]
    layer = layer_perturbation  # new layer
    adata_pert.raw = None
    # so scanpy.tl.rank_genes_groups doesn't use raw
    # pertpy doesn't specify use_raw, so if raw available
    # layer and use_raw=True are both specified -> error
    mix = pt.tl.Mixscape()
    adata_pert = adata_pert[adata_pert.obs[col_perturbed].isin(
        [key_treatment, key_control])].copy()  # ensure in perturbed/control
    adata_pert = adata_pert[~adata_pert.obs[
        col_target_genes].isnull()].copy()  # ensure no NA target genes
    print("\n<<< CALCULATING PERTURBATION SIGNATURE >>>")
    mix.perturbation_signature(
        adata_pert, col_perturbed, key_control, split_by=col_split_by,
        **kws_perturbation_signature
        )  # subtract GEX of perturbed cells from their unperturbed neighbors
    adata_pert.X = adata_pert.layers[layer]

    # Mixscape Classification & Perturbation Scoring
    print("\n<<< RUNNING MIXSCAPE ROUTINE >>>")
    mix.mixscape(adata=adata_pert,
                 # adata=adata_pert,
                 labels=col_target_genes, control=key_control,
                 layer=layer,
                 perturbation_type=key_treatment,
                 min_de_genes=min_de_genes, pval_cutoff=pval_cutoff,
                 iter_num=iter_num)  # Mixscape classification
    if target_gene_idents is True:  # to plot all target genes
        target_gene_idents = list(adata_pert.uns["mixscape"].keys())
    if plot is True:
        figs = cr.pl.plot_mixscape(
            adata_pert, col_target_genes,
            key_treatment, key_control=key_control,
            key_nonperturbed=key_nonperturbed, layer=layer,
            target_gene_idents=target_gene_idents,
            subsample_number=subsample_number,
            col_guide_rna=col_guide_rna, guide_split=guide_split)

    # Perturbation-Specific Cell Clusters
    print("\n<<< RUNNING LINEAR DISCRIMINANT ANALYSIS (CLUSTERING) >>>")
    try:
        mix.lda(adata=adata_pert,
                # adata=adata_pert,
                labels=col_target_genes,
                layer=layer, control=key_control,
                min_de_genes=min_de_genes,
                split_by=col_split_by,
                copy=False,
                perturbation_type=key_treatment,
                mixscape_class_global="mixscape_class_global",
                n_comps=n_comps_lda, logfc_threshold=logfc_threshold,
                pval_cutoff=pval_cutoff)  # linear discriminant analysis (LDA)
    except Exception as error:
        warn(f"{error}\n\nCouldn't perform perturbation-specific clustering!")
        figs["lda"] = error
    if assay_protein:
        adata[assay_protein].obs[:, "mixscape_class_global"] = adata_pert[
            assay].obs["mixscape_class_global"].loc[
                adata[assay_protein].index]  # classification -> protein assay

    # Perturbation Score Plotting
    if plot is True and target_gene_idents is not None:  # G/P EX
        fff = cr.pl.plot_mixscape(
            adata_pert, col_target_genes, key_treatment,
            adata_protein=adata[assay_protein] if assay_protein else None,
            key_control=key_control, key_nonperturbed=key_nonperturbed,
            layer=layer, target_gene_idents=target_gene_idents,
            subsample_number=subsample_number, col_guide_rna=col_guide_rna,
            guide_split=guide_split, feature_split=feature_split)
        figs = {**figs, **fff}
    else:
        figs = None
    return figs, adata_pert


def perform_augur(adata, assay=None, layer=None, augur_mode="default",
                  classifier="random_forest_classifier", subsample_size=20,
                  select_variance_features=False, n_folds=3, seed=1618,
                  n_jobs=None, kws_augur_predict=None, col_cell_type="leiden",
                  col_perturbed=None, key_control="NT", key_treatment=None,
                  col_gene_symbols="gene_symbols",
                  plot=True, cmap="coolwarm", **kwargs):
    """Calculates AUC using Augur and a specified classifier.

    Args:
        adata (AnnData): Scanpy object.
        assay (str, optional): Assay slot of adata
            ('rna' for `adata['rna']`).n
            Defaults to None (works if only one assay).
        classifier (str, optional): Classifier.
            Defaults to "random_forest_classifier".
        augur_mode (str, optional): Augur or permute?
            Defaults to "default".
        subsample_size (int, optional): Per Pertpy code:
            "number of cells to subsample randomly per type
            from each experimental condition."
        n_folds (int, optional): Number of folds for cross-validation.
            Defaults to 3.
        n_jobs (int, optional): _description_. Defaults to 4.
        select_variance_features (bool, optional): Use Augur to select
            genes (True), or Scanpy's  highly_variable_genes (False).
            Defaults to False.
        col_cell_type (str, optional): Column name for cell type.
            Defaults to "cell_type_col".
        col_perturbed (str, optional): Experimental condition column
            name. Defaults to None.
        key_control (str, optional): Control category key
            (`adata.obs[col_perturbed]` entries).Defaults to "NT".
        key_treatment (str, optional): Name of value within
            col_perturbed. Defaults to None.
        seed (int, optional): Random state (for reproducibility).
            Defaults to 1618.
        plot (bool, optional): Plots? Defaults to True.
        kws_augur_predict (dict, optional): Optional additional keyword
            arguments to pass to Augur predict.
        kwargs (keyword arguments, optional): Additional keyword
            arguments. Use key "kws_umap" and "kws_neighbors" to pass
            arguments to the relevant functions.

    Returns:
        tuple: Augur AnnData object, results from
            Augur predict, figures
    """
    if n_jobs is True:
        n_jobs = os.cpu_count() - 1  # use available CPUs - 1
    if select_variance_features == "both":
        # both methods: select genes based on...
        # - original Augur (True)
        # - scanpy's highly_variable_genes (False)
        data, results = [[None, None]] * 2  # to store results
        figs = {}
        for i, x in enumerate([True, False]):  # iterate over methods
            data[i], results[i], figs[str(i)] = perform_augur(
                adata.copy(), assay=assay,
                layer=layer,
                select_variance_features=x, classifier=classifier,
                augur_mode=augur_mode, subsample_size=subsample_size,
                n_jobs=n_jobs, n_folds=n_folds,
                col_cell_type=col_cell_type, col_perturbed=col_perturbed,
                col_gene_symbols=col_gene_symbols,
                key_control=key_control, key_treatment=key_treatment,
                seed=seed, plot=plot, **kwargs,
                **kws_augur_predict)  # recursive -- run function both ways
        figs[f"vs_select_variance_feats_{x}"] = pt.pl.ag.scatterplot(
            results[0], results[1])  # compare  methods (diagonal=same)
    else:
        # Setup
        figs = {}
        if kws_augur_predict is None:
            kws_augur_predict = {}
        adata_pert = adata[assay].copy() if assay else adata.copy()
        if layer is not None:
            adata_pert.X = adata_pert.layers[layer]

        # Unfortunately, Augur renames columns INPLACE
        # Prevent this from overwriting existing column names
        col_pert_new, col_cell_new = "label", "cell_type"
        adata_pert.obs[col_pert_new] = adata_pert.obs[col_perturbed].copy()
        adata_pert.obs[col_cell_new] = adata_pert.obs[col_cell_type].copy()
        print(adata_pert)

        # Augur Model & Data
        ag_rfc = pt.tl.Augur(classifier)
        loaded_data = ag_rfc.load(
            adata_pert, condition_label=key_control,
            treatment_label=key_treatment,
            cell_type_col=col_cell_new, label_col=col_pert_new
            )  # add dummy variables, rename cell type & label columns

        # Run Augur Predict
        data, results = ag_rfc.predict(
            loaded_data, subsample_size=subsample_size, augur_mode=augur_mode,
            select_variance_features=select_variance_features,
            n_threads=n_jobs, random_state=seed,
            **kws_augur_predict)  # AUGUR model prediction
        print(results["summary_metrics"])  # results summary

        # Plotting & Output
        if plot is True:
            if "vcenter" not in kwargs:
                kwargs.update({"vcenter": 0})
            if "legend_loc" not in kwargs:
                kwargs.update({"legend_loc": "on data"})
            if "frameon" not in kwargs:
                kwargs.update({"frameon": False})
            if "palette" not in kwargs:
                kwargs.update({"palette": None})
            if "color_map" not in kwargs:
                kwargs.update({"color_map": "reds"})
            figs["perturbation_score_umap"] = sc.pl.umap(
                data, color=["augur_score", col_cell_type],
                cmap=cmap, vcenter=0, vmax=1)
            figs["perturbation_effect_by_cell_type"] = pt.pl.ag.lollipop(
                results)  # how affected each cell type is
            # TO DO: More Augur UMAP preprocessing options?
            kws_umap = kwargs.pop("kws_umap", {})
            kws_neighbors = kwargs.pop("kws_neighbors", {})
            try:
                # def_pal = {col_perturbed: dict(zip(
                #     [key_control, key_treatment], ["black", "red"]))}
                sc.pp.neighbors(data, **kws_neighbors)
                sc.tl.umap(data, **kws_umap)
                figs["perturbation_effect_umap"] = sc.pl.umap(
                    data, color=["augur_score", col_cell_type, col_perturbed],
                    color_map=kwargs["color_map"], palette=kwargs["palette"],
                    title=["Augur Score", col_cell_type, col_perturbed]
                )  # scores super-imposed on UMAP
            except Exception as err:
                figs["perturbation_effect_umap"] = err
                warn(f"{err}\n\nCould not plot perturbation effects on UMAP!")
            figs["important_features"] = pt.pl.ag.important_features(
                results)  # most important genes for prioritization
            figs["perturbation_scores"] = {}
    return data, results, figs


def perform_differential_prioritization(adata, col_perturbed="perturbation",
                                        key_treatment_list="NT", assay=None,
                                        label_col="label",
                                        n_permutations=1000,
                                        n_subsamples=50,
                                        col_cell_type="cell_type",
                                        classifier="random_forest_classifier",
                                        plot=True, kws_augur_predict=None,
                                        **kwargs):
    """
    Determine differential prioritization based on which cell types
    were most accurately (AUC) classified as (not) perturbed in
    different runs of Augur (different values of `col_perturbed`).

    Args:
        adata (AnnData): Scanpy object.
        col_perturbed (str): Column used to indicate experimental
            condition.
        key_treatment_list (list): List of two conditions
            (values in col_perturbed).
        label_col (str, optional): _description_.
            Defaults to "label_col".
        assay (str, optional): Assay slot of
            adata ('rna' for `adata['rna']`).n
            Defaults to None (works if only one assay).
        n_permutations (int, optional): According to Pertpy:
            'the total number of mean augur scores to calculate
            from a background distribution.'
            Defaults to 1000.
        n_subsamples (int, optional): According to Pertpy:
            'number of subsamples to pool when calculating the
            mean augur score for each permutation.'
            Defaults to 50.
        col_cell_type (str, optional): Column name for cell type.
            Defaults to "cell_type_col".
        assay (str, optional): Assay slot of adata
            ('rna' for `adata['rna']`).n
            Defaults to None (works if only one assay).
        classifier (str, optional): Classifier.
            Defaults to "random_forest_classifier".
        plot (bool, optional): Plots? Defaults to True.

    Returns:
        _type_: _description_
    """
    figs = {}
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    if kws_augur_predict is None:
        kws_augur_predict = {}
    augur_results, permuted_results = [], []  # to hold results
    if plot is True:
        figs["umap_augur"] = {}
    for x in key_treatment_list:
        ag_rfc = pt.tl.Augur(classifier)
        ddd = ag_rfc.load(
            adata[assay] if assay else adata,
            condition_label=col_perturbed,
            treatment_label=x,
            cell_type_col=col_cell_type,
            label_col=label_col
            )  # add dummy variables, rename cell type & label columns
        ddd = ag_rfc.load(adata[assay] if assay else adata,
                          condition_label=col_perturbed, treatment_label=x)
        dff_a, res_a = ag_rfc.predict(ddd, augur_mode="augur",
                                      **kws_augur_predict)  # augur
        dff_p, res_p = ag_rfc.predict(ddd, augur_mode="permute",
                                      **kws_augur_predict)  # permute
        if plot is True and (("umap" in adata[assay].uns) if assay else (
                "umap" in adata.uns)):
            figs["umap_augur_score_differential"][x] = sc.pl.umap(
                adata=adata[assay] if assay else adata, color="augur_score")
        augur_results.append([res_a])
        permuted_results.append([res_p])
    pvals = ag_rfc.predict_differential_prioritization(
        augur_results1=augur_results[0],
        augur_results2=augur_results[1],
        permuted_results1=permuted_results[0],
        permuted_results2=permuted_results[1],
        n_subsamples=n_subsamples, n_permutations=n_permutations,
        )
    if plot is True:
        figs["diff_pvals"] = pt.pl.ag.dp_scatter(pvals)
    return pvals, figs


def compute_distance(adata, distance_type="edistance",
                     col_target_genes="target_genes",
                     col_cell_type=None, key_target_genes=None,
                     key_cell_type=None,  obsm_key="X_pca", method="X_pca",
                     kws_plot=None, highlight_real_range=True,
                     alpha=0.05, correction="holm-sidak",
                     plot=True, n_jobs=1, n_perms=1000, **kwargs):
    """Compute distance & hierarchies; (optionally) make heatmaps."""
    figs, dff, res_linkage, data, res_contrasts = {}, None, None, None, {}
    if kws_plot is None:
        kws_plot = dict(robust=True, figsize=(10, 10))
    kwargs = {"n_jobs": n_jobs, **kwargs}

    # Initialize Distance Object
    model = pt.tl.Distance(distance_type, obsm_key=method)

    # Distance Metrics (Target Genes/Conditions)
    data = model.pairwise(adata, groupby=col_target_genes, **kwargs)

    # Cluster Hierarchies
    if col_cell_type is not None:
        dff = model.pairwise(adata, groupby=col_cell_type, **kwargs)
        res_linkage = linkage(dff, method="ward")  # linkage

    # Contrasts (vs. Reference Cell Type &/or Target Gene)
    cont = pt.tl.DistanceTest(distance_type, obsm_key=method, alpha=alpha,
                              correction=correction, n_perms=n_perms)
    for x in zip([col_cell_type, col_target_genes], [
            key_cell_type, key_target_genes]):  # iterate cell, gene reference
        if x[1] is not None:
            ref = x[1] if isinstance(x[1], str) else x[1][0]
            ann = adata if isinstance(x[1], str) else adata[
                adata.obs[x[0]].isin(x[1])].copy()  # subset if list of groups
            res_contrasts[" = ".join([x[0], ref])] = cont(
                ann, x[0], contrast=ref)  # contrast
    # Plot
    if plot is True:  # cluster hierarchies
        figs = cr.pl.plot_distance(
            res_pairwise_genes=data, res_pairwise_clusters=dff,
            res_linkage=res_linkage, distance_type=distance_type,
            res_contrasts=res_contrasts, **kws_plot)  # plots
    return model, data, dff, res_linkage, res_contrasts, figs


def perform_gsea(pdata, adata_sc=None,
                 col_cell_type=None, col_sample_id=None,
                 col_condition="leiden", key_condition="0",
                 col_label_new=None, ifn_pathways=True,
                 layer=None, p_threshold=0.0001, use_raw=False,
                 kws_pseudobulk=None, seed=1618, kws_run_gsea=None,
                 filter_by_highly_variable=False, geneset_size_range=None,
                 obsm_key="gsea_estimate", pseudobulk=True, **kwargs):
    """
    Perform gene set enrichment analysis
    (adapted from SC Best Practices).

    Example
    -------
    This way will automatically plot the pathways with the highest
    absolute scores (and with p-value below a defined threshold):
    >>> out = perform_gsea(pdata, adata_sc=None,  # will not do AUCell
    >>>                    col_condition=["cell_type", "Inflammation"],
    >>>                    key_condition="Epithelial_Inflamed",
    >>>                    col_label_new="Group", ifn_pathways=True,
    >>>                    p_threshold=0.0001,
    >>>                    filter_by_highly_variable=False)

    Or define your own pathways to plot:

    >>> ifn = []
    >>> out = perform_gsea(None,  # so will create pseudobulk from sc
    >>>                    adata_sc=adata_sc,  # will run AUCell
    >>>                    col_condition=["cell_type", "Inflammation"],
    >>>                    key_condition="Epithelial_Inflamed",
    >>>                    col_label_new="Group", ifn_pathways=True,
    >>>                    p_threshold=0.0001,
    >>>                    geneset_size_range=[15, 500],
    >>>                    filter_by_highly_variable=False)
    """
    figs, gsea_results_cell = {}, None
    if adata_sc is not None and pdata is None:  # if needed, create pseudobulk
        pdata = cr.tl.create_pseudobulk(
            adata_sc.copy(), col_cell_type, col_sample_id=col_sample_id,
            layer="counts", mode="sum", kws_process=True)  # pseudobulk
    if geneset_size_range is None:  # default gene set size range
        geneset_size_range = [15, 500]
    if kws_run_gsea is None:
        kws_run_gsea = dict(verbose=True)
    if layer and adata_sc:
        adata_sc.X = adata_sc.layers[layer].copy()
    if isinstance(col_condition, str):
        col_label_new = col_condition
    else:  # if multi-condition, create column representing combination
        if col_label_new is None:
            col_label_new = "_".join(col_condition)  # condition combo label
        pdata, adata_sc = [cr.tl.create_condition_combo(
            x.obs, col_condition, col_label_new) if x else None for x in [
                pdata, adata_sc]]  # create combination label (e.g., c1_c2)

    # Rank Genes
    sc.tl.rank_genes_groups(pdata, col_label_new, method="t-test",
                            key_added="t-test", use_raw=use_raw)
    t_stats = sc.get.rank_genes_groups_df(
        pdata, key_condition, key="t-test").set_index("names")  # rank genes
    print(t_stats)
    if filter_by_highly_variable is True:  # filter by HVGs?
        t_stats = t_stats.loc[pdata.var["highly_variable"]]
    t_stats = t_stats.sort_values("scores", key=np.abs, ascending=False)[[
        "scores"]].rename_axis([key_condition], axis=1)  # decoupler format
    print(t_stats)

    # Retrieve Reactome Pathways
    msigdb = decoupler.get_resource("MSigDB")
    rxome = msigdb.query("collection == 'reactome_pathways'")
    rxome = rxome[~rxome.duplicated(("geneset", "genesymbol"))]  # -duplicate

    # Filter Gene Sets for Compatibility
    geneset_size = rxome.groupby("geneset").size()
    genesets = geneset_size.index[(geneset_size > geneset_size_range[0]) & (
        geneset_size < geneset_size_range[1])]  # only gene sets in size range

    # Cluster-Level Analysis
    scores, norm, pvals = decoupler.run_gsea(
        t_stats.T, rxome[rxome["geneset"].isin(genesets)], seed=seed,
        **kws_run_gsea, use_raw=use_raw,
        source="geneset", target="genesymbol")  # run cluster-level GSEA
    # pdata.obsm[obsm_key] = scores
    gsea_results = pd.concat({
        "score": scores.T, "norm": norm.T, "pval": pvals.T
        }, axis=1).droplevel(level=1, axis=1).sort_values("pval")  # results
    gsea_results = gsea_results.assign(
        **{"-log10(pval)": lambda x: -np.log10(x["pval"])})
    gsea_results = gsea_results.assign(
        col=[col_condition] * gsea_results.shape[0]).assign(
        key=[key_condition] * gsea_results.shape[0]
        )  # DO NOT CHANGE THESE NAMES. Plotting fx depends on it
    print(gsea_results.head(20))
    score_sort = gsea_results[gsea_results.pval < p_threshold]  # filter ~ p
    score_sort = score_sort.loc[score_sort.score.abs().sort_values(
        ascending=False).index]  # sort by absolute score

    # Cell-Level
    if adata_sc is not None:
        adata_sc = adata_sc.copy()
        sc.tl.rank_genes_groups(adata_sc, col_label_new, method="t-test",
                                key_added="t-test", use_raw=use_raw)
        gsea_results_cell = decoupler.run_aucell(
            adata_sc, rxome, source="geneset", target="genesymbol",
            use_raw=False)  # run individual cell-level GSEA

    # Plots & Other Output
    try:
        figs = cr.pl.plot_gsea_results(
            adata_sc if adata_sc else pdata, gsea_results,
            p_threshold=p_threshold, **kwargs, ifn_pathways=ifn_pathways,
            use_raw=use_raw, layer=layer)  # plot GSEA results
    except Exception as err:
        warn(f"{err}\n\n\nPlotting GSEA results failed.")
        figs = err
    res = {"gsea_results": gsea_results, "score_sort": score_sort,
           "gsea_results_cell": gsea_results_cell}
    return pdata, adata_sc, res, figs


def perform_gsea_pt(adata, col_condition, key_condition=None, layer=None,
                    correction="benjamini-hochberg",
                    absolute=False, library_blitz=None, **kwargs):
    raise NotImplementedError("Pertpy GSEA not yet released.")
    """Perform GSEA (Pertpy-style)."""
    # res, fig, key_add = {}, {}, "pertpy_enrichment"  # for results
    # ref = "rest" if key_condition is None else key_condition if (
    #     isinstance(key_condition, str)) else None  # 1st key=reference
    # if isinstance(key_condition, (list, np.ndarray)):  # if condition subset
    #     adata = adata[adata.obs[col_condition].isin(key_condition)
    #                     ]  # only keep conditions specified in key_condition
    # if layer:
    #     adata.X = adata.layers[layer].copy()
    # rgg = "rank_genes_groups"  # rank genes key (cell types RGG)
    # rgo = "rank_genes_groups_o"  # original rank genes key
    # if rgg in adata.uns:  # preserve original rank genes
    #     adata.uns[rgo] = adata.uns[rgg]
    # model = pt.tl.Enrichment()
    # kws = cr.tl.merge(dict(
    #     nested=False, categories=None, method="mean", n_bins=25,
    #     ctrl_size=50), {**kwargs, "key_added": key_add}, how="left")
    # kws["targets"] = blitz.enrichr.get_library(
    #     library_blitz) if library_blitz else None  # custom resource?
    # model.score(adata, layer=layer, key_added=key_add, **kws)  # ~ cell
    # sc.tl.rank_genes_groups(adata, method="wilcoxon", reference=ref,
    #                         groupby=col_condition)  # rank genes ~ condition
    # model.plot_dotplot(adata, groupby=col_condition)  # GEX dotplot
    # res["gsea"] = model.gsea(absolute=absolute)  # run blitzgsea GSEA
    # model.plot_gsea(adata, res["gsea"], interactive_plot=True)  # plot
    # fig["gsea"] = plt.gcf()
    # geo = model.hypergeometric(adata, absolute=absolute,
    #                            corr_method=correction)  # significance test
    # res["hypergeometric"] = model.gsea(adata)
    # model.plot_gsea(adata, res["hypergeometric"], interactive_plot=True)
    # fig["hypergeometric"] = plt.gcf()
    # fig["blitz"] = {}
    # for x in adata.obs[col_condition].unique():  # iterate cell types
    #     try:
    #         fig["blitz"][x] = blitz.plot.running_sum(
    #             signature=adata.uns[f"{key_add}_gsea"]["scores"][x],
    #             library=adata.uns["{key_add}_gsea"]["targets"],
    #             geneset="MHC class II receptor activity (GO:0032395)",
    #             result=res[x], interactive_plot=True)
    #         fig["blitz"].show()
    #     except:
    #         print(traceback.format_exc(), "\n\nGSEA plot failed!")
    # adata.uns[f"{rgg}_{col_condition}"] = adata.uns[rgg]  # new rank genes
    # if rgo in adata.uns:  # if originally had rank ~ cluster...
    #     adata.uns[rgg] = adata.uns[f"{rgg}_o"]  # restore original ranks
    #     _ = adata.uns.pop(rgo)  # remove placeholder rank genes from `.uns`
    # return adata, res, fig


def perform_pathway_interference(adata, layer=None, n_top=500, copy=True,
                                 organism="human", obsm_key="mlm_estimate",
                                 col_cell_type="louvain", pathways=True,
                                 **kwargs):
    """Perform Pathway Interference Analysis."""
    if copy is True:
        adata = adata.copy()
    if layer:
        adata.X = adata.layers[layer].copy()
    figs = {}
    prog = decoupler.get_progeny(organism=organism, top=n_top)
    decoupler.run_mlm(mat=adata, net=prog, source="source", target="target",
                      weight="weight", verbose=True)
    if pathways:
        if pathways is True:  # plot all available pathways
            adata = decoupler.get_acts(adata, obsm_key=obsm_key)
            pathways = list(adata.obsm[obsm_key].columns)
        for p in pathways:
            try:
                figs[p] = cr.pl.plot_pathway_interference_results(
                    adata, p, col_cell_type=col_cell_type, obsm_key=obsm_key,
                    **kwargs)  # plots for pathway
            except Exception as err:
                warn(f"{err}\n\n\nPlotting pathway {p} failed!")
    print(adata.obsm["mlm_estimate"])
    return adata, figs


def perform_dea(adata, col_cell_type, col_covariates, layer=None,
                col_sample_id=None, figsize=30, uns_key="pca_anova",
                obsm_key="X_pca", plot_stat="p_adj"):
    """
    Perform functional analysis of pseudobulk data
    (created by this method), then differential expression analysis.
    """
    adata = adata.copy()
    if isinstance(col_covariates, str):
        col_covariates = [col_covariates]
    if isinstance(figsize, (int, float)):
        figsize = (figsize, figsize)
    if layer:
        adata.X = adata.layers[layer]

    # Create Pseudo-Bulk Data
    pdata = cr.tl.create_pseudobulk(adata, col_cell_type, col_sample_id=None,
                                    layer=layer, mode="sum", kws_process=True)

    # Calculate Associations
    decoupler.get_metadata_associations(
        pdata, obs_keys=[col_cell_type, "psbulk_n_cells", "psbulk_counts"
                         ] + col_covariates, obsm_key=obsm_key,
        uns_key=uns_key, inplace=True)

    # Plot
    fig = plt.figure(figsize=figsize)
    axs, legend_axes = decoupler.plot_associations(
        pdata, uns_key=uns_key, obsm_key=obsm_key,
        stat_col=plot_stat, obs_annotation_cols=col_covariates,
        titles=["Adjusted p-Values from ANOVA", "Principle Component Scores"])
    plt.show()
    return pdata, fig


def calculate_dea_deseq2(pdata, col_cell_type, col_condition, key_control,
                         key_treatment, top_n=20, n_jobs=4, col_subject=None,
                         layer_counts="counts", col_gene_symbols=None,
                         min_prop=0, min_count=0, min_total_count=0,
                         shrink_lfc=True, p_threshold=0.05, **kwargs):
    """
    Calculate DEA based on Liana tutorial usage of DESeq2.

    Extra keyword arguments are passed to DeseqDataset.
    """
    dea, quiet, figsize = {}, True, kwargs.pop("figsize", (20, 20))
    filt_c, filt_i = [kwargs.pop(x, True) for x in [
        "cooks_filter", "independent_filter"]]  # DESeqStats filter arguments
    if col_gene_symbols is None:  # if gene name column unspecified...
        col_gene_symbols = pdata.var.index.names[0]  # ...index=gene names
    facs = col_condition if not col_subject else [col_condition, col_subject]
    pdata = pdata[~pdata.obs[col_condition].isna()].copy()  # no condition NAs
    cts = pdata.obs.groupby(col_cell_type).apply(
            lambda x: np.nan if any((sum(x[col_condition] == k) < 2 for k in [
                key_treatment, key_control])) else False).dropna(
                    ).index.values.tolist()  # types w/ enough N/condition

    # Run DEA for Each Cell Type
    for t in cts:  # iterate cell types from above: w/ both conditions & n > 3

        # Set Up Pseudo-Bulk Data
        psub = pdata[pdata.obs[col_cell_type] == t].copy()  # subset c type t
        # genes = decoupler.filter_by_expr(  # edgeR-based filtering function
        #     psub, group=col_condition, min_count=min_count, min_prop=min_prop,
        #     min_total_count=min_total_count)  # genes with enough counts/reads
        # psub = psub[:, genes].copy()  # filter data ~ those genes

        # Skip Cell Type if Not Enough Data or Doesn't Contain Both Conditions
        if any(psub[psub.obs[col_condition].isin([key_control, key_treatment])
                    ].obs.value_counts() < 2):
            dea[t] = None
            warn(f"Skipping {t} DEA: levels missing in {col_condition}")
            continue  # skip if doesn't contain both conditions or n < 4

        # Perform DESeq2
        psub.X = psub.layers[layer_counts].copy()  # counts layer
        # dds = DeseqDataSet(
        #     adata=psub, design_factors=facs, quiet=quiet, n_cpus=n_jobs,
        #     ref_level=[col_condition, key_control], **kwargs)  # DESeq adata
        dds = DeseqDataSet(
            adata=psub, design_factors=facs, quiet=quiet,
            ref_level=[col_condition, key_control], **kwargs)  # DESeq adata
        dds.deseq2()  # estimate dispersion & logfold change
        dea[t] = DeseqStats(dds, alpha=p_threshold, contrast=[
            col_condition, key_treatment, key_control], quiet=quiet,

                            cooks_filter=filt_c, independent_filter=filt_i)
        dea[t].quiet = quiet
        dea[t].summary()  # print Wald test summary
        if shrink_lfc is True:
            dea[t].lfc_shrink(
                coeff=f"{col_condition}_{key_treatment}_vs_{key_control}")

    # Construct Unified Results DataFrame & Plot
    dea_df = [None if dea[t] is None else dea[t].results_df for t in dea]
    if all((x is None for x in dea_df)):  # if no cell types had enough data
        warn("DESeq2 FAILED: No cell types passed filtering conditions.")
        return None, None, None
    else:  # concatenate results dfs list for all cell types -> dataframe
        dea_df = pd.concat(dea_df, names=[
            col_cell_type, col_gene_symbols], keys=dea.keys())  # concatenate
        dea_df = dea_df.reset_index().set_index(col_gene_symbols)  # ix = gene
    try:  # try to plot
        p_dims = cr.pl.square_grid(len(dea))
        fig, axs = plt.subplots(p_dims[0], p_dims[1], figsize=figsize)
        for i, x in enumerate(dea):
            try:
                decoupler.plot_volcano_df(
                    dea[x].results_df, x="log2FoldChange", y="padj",
                    sign_thr=0.05, top=top_n, lFCs_thr=0.5, sign_limit=None,
                    lFCs_limit=None, dpi=200, ax=axs.ravel()[i])  # plot
                axs.ravel()[i].set_title(x)
            except Exception:
                print(traceback.format_exc(), f"\n\nDEA volcano failed ({x})")
        fig.tight_layout()
    except Exception:
        fig = None
        print(traceback.format_exc(), "\n\nDEA volcano plotting failed!")
    return dea, dea_df, fig

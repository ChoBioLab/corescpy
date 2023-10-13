import pertpy as pt
import muon
import mudata
import arviz as az
import scanpy as sc
import functools
import matplotlib.pyplot as plt
from seaborn import clustermap
import warnings
import decoupler
from scipy.cluster.hierarchy import linkage, dendrogram
import re
import pandas as pd
import numpy as np
import crispr as cr

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"
    
    
def perform_mixscape(adata, col_perturbed="perturbation",
                     key_control="NT",
                     key_treatment="perturbed",
                     assay=None,
                     col_guide_rna="guide_ID",
                     col_split_by=None,
                     col_target_genes="gene_target",
                     layer_perturbation="X_pert", 
                     iter_num=10,
                     min_de_genes=5, pval_cutoff=5e-2, logfc_threshold=0.25,
                     subsample_number=300,
                     n_comps_lda=None, 
                     plot=True, 
                     assay_protein=None,
                     protein_of_interest=None,
                     guide_split="-",
                     target_gene_idents=None, 
                     kws_perturbation_signature=None,
                     **kwargs):
    """
    Identify perturbed cells based on target genes (`adata.obs['mixscape_class']`,
    `adata.obs['mixscape_class_global']`) and calculate posterior probabilities
    (`adata.obs['mixscape_class_p_<key_treatment>']`, e.g., KO) and
    perturbation scores. 
    
    Optionally, perform LDA to cluster cells based on perturbation response.
    Optionally, create figures related to differential gene 
    (and protein, if available) expression, perturbation scores, 
    and perturbation response-based clusters. 
    
    Runs a differential expression analysis and creates a heatmap
    sorted by the posterior probabilities.

    Args:
        adata (AnnData): Scanpy data object.
        col_perturbed (str): Perturbation category column of `adata.obs` 
            (should contain key_control).
        key_control (str, optional): The label in `col_perturbed`
            that indicates control condition. Defaults to "NT".
        key_treatment (str, optional): The label in `col_perturbed`
            that indicates a treatment condition (e.g., drug administration, 
            CRISPR knock-out/down). Defaults to "KO".
            Will also be 
        col_split_by (str, optional): `adata.obs` column name of 
            sample categories to calculate separately (e.g., replicates). 
            Defaults to None.
        col_target_genes (str, optional): Name of column with target genes. 
            Defaults to "gene_target".
        assay (str, optional): Assay slot of adata ('rna' for `adata['rna']`).n
            Defaults to None (works if only one assay).
        assay_protein (str, optional): Protein assay slot name (if available).
            Defaults to None.
        protein_of_interest (str, optional): If assay_protein is not None 
            and plot is True, will allow creation of violin plot of 
            protein expression (y) by 
            <target_gene_idents> perturbation category (x),
            split/color-coded by Mixscape classification 
            (`adata.obs['mixscape_class_global']`). Defaults to None.
        col_guide_rna (str, optional): Name of column with guide RNA IDs (full).
            Format may be something like STAT1-1|CNTRL-2-1. 
            Defaults to "guide_ID".
        guide_split (str, optional): Guide RNA ID # split character
            before guide #(s) (as in "-" for "STAT3-1-2"). Same as used in
            Crispr/crispr_class.py process guide RNA method. Defaults to "-".
        target_gene_idents (list or bool, optional): List of names of genes 
            whose perturbations will determine cell grouping 
            for the above-described violin plot and/or
            whose differential expression posterior probabilities 
            will be plotted in a heatmap. Defaults to None.
            True to plot all in `adata.uns["mixscape"]`.
        layer_perturbation (str, optional): `adata.layers` slot name. 
            Defaults to None.
        min_de_genes (int, optional): Minimum number of genes a cell has 
            to express differentially to be labeled 'perturbed'. 
            For Mixscape and LDA (if applicable). Defaults to 5.
        pval_cutoff (float, optional): Threshold for significance 
            to identify differentially-expressed genes. 
            For Mixscape and LDA (if applicable). Defaults to 5e-2.
        logfc_threshold (float, optional): Will only test genes whose average 
            logfold change across the two cell groups is at least this number. 
            For Mixscape and LDA (if applicable). Defaults to 0.25.
        n_comps_lda (int, optional): Number of principal components (e.g., 10)
            for PCA xperformed as part of LDA for pooled CRISPR screen data. 
            Defaults to None.
        iter_num (float, optional): Iterations to run to converge if needed.
        plot (bool, optional): Make plots? Defaults to True.
        kws_perturbation_signature (dict, optional): Optional keyword arguments
             to pass to `pertpy.tl.PerturbationSignature()` 
             (also see Pertpy documentation).
            "n_neighbors" (# of unperturbed neighbors to use for comparison
                when calculating perturbation signature), 
            "n_pcs", "use_rep" (`X` or any `.obsm` keys), 
            "batch_size" (if None, use full data, which is memory-intensive,
                or specify an integer to calculate signature in batches,
                which is inefficient for sparse data).
    """
    figs = {}
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    if kws_perturbation_signature is None:
        kws_perturbation_signature = {}
        
    # Perturbation Signature
    mix = pt.tl.Mixscape()
    adata_pert = mix.perturbation_signature(
        adata[assay] if assay else adata, col_perturbed, 
        key_control, split_by=col_split_by, copy=True,
        **kws_perturbation_signature
        )  # subtract GEX of perturbed cells from their unperturbed neighbors
    adata_pert = adata_pert[adata_pert.obs[col_perturbed].isin(
        [key_treatment, key_control])].copy()  # ensure in perturbed/control
    adata_pert = adata_pert[~adata_pert.obs[
        col_target_genes].isnull()].copy()  # ensure no NA target genes
    if layer_perturbation != "X_pert":
        adata_pert.layers[layer_perturbation] = adata_pert.layers["X_pert"]
    adata_pert.X = adata_pert.layers[layer_perturbation]
    
    # Mixscape Classification & Perturbation Scoring
    mix.mixscape(adata=adata_pert, 
                 # adata=adata_pert,
                 labels=col_target_genes, control=key_control, 
                 layer=layer_perturbation, 
                 perturbation_type=key_treatment,
                 min_de_genes=min_de_genes, pval_cutoff=pval_cutoff,
                 iter_num=iter_num)  # Mixscape classification
    if target_gene_idents is True:  # to plot all target genes
        target_gene_idents = list(adata_pert.uns["mixscape"].keys())  # targets
    if plot is True:
        if target_gene_idents is not None:  # G/P EX
            figs["mixscape_DEX_ordered_by_ppp_heat"] = {}
            figs["mixscape_ppp_violin"] = {}
            figs["mixscape_perturb_score"] = {}
            try:
                fpp = cr.pl.plot_perturbation_scores_by_guide(
                    adata_pert, guide_rna_column=col_guide_rna, 
                    guide_split=guide_split)
                figs["mixscape_targeting_efficacy"] = fpp
            except Exception as err:
                figs["perturbation_clusters"] = err
                warnings.warn(f"{err}\n\nCould not plot targeting efficiency!")
            for g in target_gene_idents:  # iterate target genes of interest
                if g not in list(adata_pert.obs["mixscape_class"]):
                    print(f"\n\nTarget gene {g} not in mixscape_class!")
                    continue  # skip to next target gene if missing
                figs["mixscape_perturb_score"][g] = pt.pl.ms.perturbscore(
                    adata=adata_pert, labels=col_target_genes, 
                    target_gene=g, color="red")
                try:
                    figs["mixscape_DEX_ordered_by_ppp_heat"][
                        g] = pt.pl.ms.heatmap(
                            adata=adata_pert, 
                            subsample_number=subsample_number,
                            labels=col_target_genes, target_gene=g, 
                            layer=layer_perturbation, control=key_control
                            )  # differential expression heatmap ordered by PPs
                except Exception as err:
                    figs["mixscape_DEX_ordered_by_ppp_heat"][g] = err
                    warnings.warn(f"{err}\n\nCould not plot DEX heatmap!")
                tg_conds = [
                    key_control, f"{g} NP", 
                    f"{g} {key_treatment}"]  # conditions: gene g
                figs["mixscape_ppp_violin"][g] = pt.pl.ms.violin(
                    adata=adata_pert, target_gene_idents=tg_conds, rotation=45,
                    keys=f"mixscape_class_p_{key_treatment}".lower(),
                    groupby="mixscape_class")  # gene: perturbed, NP, control
            figs["mixscape_ppp_violin"][f"global"] = pt.pl.ms.violin(
                adata=adata_pert, target_gene_idents=[
                    key_control, "NP", key_treatment], rotation=45,
                keys=f"mixscape_class_p_{key_treatment}".lower(),
                groupby="mixscape_class_global")  # same, but global
            tg_conds = [key_control] + functools.reduce(
                lambda i, j: i + j, [[f"{g} NP", 
                        f"{g} {key_treatment}"] 
                for g in target_gene_idents])  # conditions: all genes
            figs["mixscape_ppp_violin"]["all"] = pt.pl.ms.violin(
                adata=adata_pert,
                keys=f"mixscape_class_p_{key_treatment}".lower(),
                target_gene_idents=tg_conds, rotation=45,
                groupby="mixscape_class")  # gene: perturbed, NP, control
            
    # Perturbation-Specific Cell Clusters
    try:
        mix.lda(adata=adata_pert, 
                # adata=adata_pert,
                labels=col_target_genes, 
                layer=layer_perturbation, control=key_control, 
                min_de_genes=min_de_genes,
                split_by=col_split_by, 
                copy=False,
                perturbation_type=key_treatment,
                mixscape_class_global="mixscape_class_global",
                n_comps=n_comps_lda, logfc_threshold=logfc_threshold,
                pval_cutoff=pval_cutoff)  # linear discriminant analysis (LDA)
        if plot is True:
            try:
                figs["perturbation_clusters"] = pt.pl.ms.lda(
                    adata=adata_pert, 
                    control=key_control)  # cluster perturbation
            except Exception as err:
                figs["perturbation_clusters"] = err
                warnings.warn(f"{err}\n\nCouldn't plot perturbation clusters!")
            if n_comps_lda is not None:  # LDA clusters
                try:
                    figs["cluster_perturbation_response"] = pt.pl.ms.lda(
                        adata_pert, control=key_control)  # perturbation response
                except Exception as err:
                    figs["perturbation_clusters"] = err
                    warnings.warn(
                        f"{err}\n\nCould not plot cluster perturbation response!")
                try:
                    if assay_protein is not None and (
                        target_gene_idents is not None):
                        f_pr = pt.pl.ms.violin(
                            adata=adata_pert,
                            # adata=adata_pert, 
                            target_gene_idents=target_gene_idents,
                            keys=protein_of_interest, groupby=col_target_genes,
                            hue="mixscape_class_global")
                        figs[f"mixscape_protein_{protein_of_interest}"] = f_pr 
                except Exception as err:
                    figs[f"mixscape_protein_{protein_of_interest}"] = err
                    warnings.warn(
                        f"{err}\n\nCould not plot protein expression!")
    except Exception as error:
        warnings.warn(
            f"{error}\n\nCouldn't perform perturbation-specific clustering!")
        
    # Store Results
    try:
        if assay:
            adata[assay].uns["mixscape"] = adata_pert.uns[
                "mixscape"]  # `.uns` join
        else:
            adata.uns["mixscape"] = adata_pert.uns["mixscape"]  # `.uns` join
    except Exception as err:
        warnings.warn(f"\n{err}\n\nCould not update `adata.uns`. In figs.")
        figs.update({"results_mixscape": adata_pert.uns["mixscape"]})
    if assay:
        adata[assay].obs = adata.obs.join(adata_pert.obs[
            ["mixscape_class", "mixscape_class_global"]], lsuffix="_o")  # data
    else:
        adata.obs = adata.obs.join(adata_pert.obs[
            ["mixscape_class", "mixscape_class_global"]], lsuffix="_o")  # data
    return figs


def perform_augur(adata, assay=None, layer_perturbation=None,
                  classifier="random_forest_classifier", 
                  augur_mode="default", 
                  subsample_size=20, n_folds=3,
                  select_variance_features=False, 
                  col_cell_type="leiden",
                  col_perturbed=None, 
                  col_gene_symbols="gene_symbols",
                  key_control="NT", key_treatment=None,
                  seed=1618, plot=True, n_threads=None,
                  kws_augur_predict=None, **kwargs):
    """Calculates AUC using Augur and a specified classifier.

    Args:
        adata (AnnData): Scanpy object.
        assay (str, optional): Assay slot of adata ('rna' for `adata['rna']`).n
            Defaults to None (works if only one assay).
        classifier (str, optional): Classifier. 
            Defaults to "random_forest_classifier".
        augur_mode (str, optional): Augur or permute? Defaults to "default".
        subsample_size (int, optional): Per Pertpy code: 
            "number of cells to subsample randomly per type 
            from each experimental condition."
        n_folds (int, optional): Number of folds for cross-validation. 
            Defaults to 3.
        n_threads (int, optional): _description_. Defaults to 4.
        select_variance_features (bool, optional): Use Augur to select 
            genes (True), or Scanpy's  highly_variable_genes (False). 
            Defaults to False.
        col_cell_type (str, optional): Column name for cell type. 
            Defaults to "cell_type_col".
        col_perturbed (str, optional): Experimental condition column name. 
            Defaults to None.
        key_control (str, optional): Control category key
            (`adata.obs[col_perturbed]` entries).Defaults to "NT".
        key_treatment (str, optional): Name of value within col_perturbed. 
            Defaults to None.
        seed (int, optional): Random state (for reproducibility). 
            Defaults to 1618.
        plot (bool, optional): Plots? Defaults to True.
        kws_augur_predict (dict, optional): Optional additional keyword 
            arguments to pass to Augur predict.
        kwargs (keyword arguments, optional): Additional keyword arguments.
            Use key "kws_umap" and "kws_neighbors" to pass arguments 
            to the relevant

    Returns:
        tuple: Augur AnnData object, results from Augur predict, figures
    """
    if select_variance_features == "both":  
        # both methods: select genes based on...
        # - original Augur (True)
        # - scanpy's highly_variable_genes (False)
        data, results = [[None, None]] * 2  # to store results
        figs = {}
        for i, x in enumerate([True, False]):  # iterate over methods
            data[i], results[i], figs[str(i)] = perform_augur(
                adata.copy(), assay=assay, layer_perturbation=None,
                select_variance_features=x, classifier=classifier,
                augur_mode=augur_mode, subsample_size=subsample_size,
                n_threads=n_threads, n_folds=n_folds,
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
        if layer_perturbation is not None:
            adata_pert.X = adata_pert.layers[layer_perturbation]
        # if adata_pert.var_names[0] != adata_pert.var.reset_index()[
        #     col_gene_symbols][0]:  # so gene names plots use if index differs
        #     adata_pert.var_names = pd.Index(adata_pert.var.reset_index()[
        #         col_gene_symbols])
        
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
            cell_type_col=col_cell_new,
            label_col=col_pert_new
            )  # add dummy variables, rename cell type & label columns

        # Run Augur Predict
        data, results = ag_rfc.predict(
            loaded_data, subsample_size=subsample_size, augur_mode=augur_mode, 
            select_variance_features=select_variance_features,
            n_threads=n_threads, random_state=seed,
            **kws_augur_predict)  # AUGUR model prediction
        print(results["summary_metrics"])  # results summary
        
        # Plotting & Output
        if plot is True:
            if "vcenter" not in kwargs:
                kwargs.update({"vcenter": 0})
            if "legend_loc" not in kwargs:
                kwargs.update({"legend_loc": "on_data"})
            if "frameon" not in kwargs:
                kwargs.update({"frameon": False})
            figs["perturbation_score_umap"] = sc.pl.umap(
                data, color=["augur_score", col_cell_type], 
                cmap="coolwarm", vcenter=0, vmax=1)
            figs["perturbation_effect_by_cell_type"] = pt.pl.ag.lollipop(
                results)  # how affected each cell type is
            # TO DO: More Augur UMAP preprocessing options?
            kws_umap = kwargs.pop("kws_umap") if "kws_umap" in kwargs else {}
            kws_neighbors = kwargs.pop(
                "kws_neighbors") if "kws_neighbors" in kwargs else {}
            try:
                # def_pal = {col_perturbed: dict(zip(
                #     [key_control, key_treatment], ["black", "red"]))}
                sc.pp.neighbors(data, **kws_neighbors)
                sc.tl.umap(data, **kws_umap)
                figs["perturbation_effect_umap"] = sc.pl.umap(
                    data, color=["augur_score", col_cell_type, 
                                 col_perturbed],
                    color_map=kwargs[
                        "color_map"] if "color_map" in kwargs else "reds",
                    palette=kwargs[
                        "palette"] if "palette" in kwargs else None,
                    title=["Augur Score", col_cell_type, col_perturbed]
                )  # scores super-imposed on UMAP
            except Exception as err:
                figs["perturbation_effect_umap"] = err
                warnings.warn(
                    f"{err}\n\nCould not plot perturbation effects on UMAP!")
            figs["important_features"] = pt.pl.ag.important_features(
                results)  # most important genes for prioritization
            figs["perturbation_scores"] = {}
        adata_pert
    return data, results, figs


def perform_differential_prioritization(adata, col_perturbed="perturbation", 
                                        key_treatment_list="NT",
                                        label_col="label",
                                        assay=None,
                                        n_permutations=1000,
                                        n_subsamples=50,
                                        col_cell_type="cell_type",
                                        classifier="random_forest_classifier", 
                                        plot=True, kws_augur_predict=None, 
                                        **kwargs):
    """
    Determine differential prioritization based on which cell types 
    were most accurately (AUC) classified as (not) perturbed in different
    runs of Augur (different values of col_perturbed).

    Args:
        adata (AnnData): Scanpy object.
        col_perturbed (str): Column used to indicate experimental condition.
        key_treatment_list (list): List of two conditions 
            (values in col_perturbed).
        label_col (str, optional): _description_. Defaults to "label_col".
        assay (str, optional): Assay slot of adata ('rna' for `adata['rna']`).n
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
        assay (str, optional): Assay slot of adata ('rna' for `adata['rna']`).n
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
        if plot is True and (
            ("umap" in adata[assay].uns) if assay else ("umap" in adata.uns)): 
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


def analyze_composition(adata, reference_cell_type,
                        assay=None, 
                        analysis_type="cell_level",
                        generate_sample_level=True, 
                        col_cell_type="cell_type",
                        sample_identifier="batch",
                        col_perturbed="condition",
                        est_fdr=0.05,
                        plot=True,
                        out_file=None, **kwargs):
    """Perform SCCoda compositional analysis."""
    figs, results = {}, {}
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    if generate_sample_level is True and sample_identifier is None:
        warnings.warn("""
                      Can't generate sample level if `sample_identifier`=None. 
                      Setting `generate_sample_level` to False.
                      """)
        generate_sample_level = False
    mod = "coda"
    sccoda_model = pt.tl.Sccoda()
    sccoda_data = sccoda_model.load(
        adata.copy(), type=analysis_type, 
        modality_key_1=assay if assay else "rna",
        modality_key_2="coda",
        generate_sample_level=generate_sample_level,
        cell_type_identifier=col_cell_type, 
        sample_identifier=sample_identifier, 
        covariate_obs=[col_perturbed])  # load data
    sccoda_data = sccoda_model.prepare(
        sccoda_data, formula=col_perturbed, 
        reference_cell_type=reference_cell_type)
    print(sccoda_data)
    print(sccoda_data["coda"].X)
    print(sccoda_data["coda"].obs)
    # mod = assay if assay else list(set(sccoda_data.mod.keys()).difference(
    #     set(["coda"])))[0]  # original modality
    if plot is True:
        figs[
            "find_reference"] = pt.pl.coda.rel_abundance_dispersion_plot(
                sccoda_data, modality_key=mod, 
                abundant_threshold=0.9)  # helps choose rference cell type
        figs["proportions"] = pt.pl.coda.boxplots(
            sccoda_data, modality_key=mod, 
            feature_name=col_perturbed, add_dots=True)
    sccoda_data = sccoda_model.prepare(
        sccoda_data, modality_key=mod, formula=col_perturbed,
        reference_cell_type=reference_cell_type)  # setup model
    sccoda_model.run_nuts(sccoda_data, 
                          modality_key=mod)  # no-U-turn HMV sampling 
    sccoda_model.summary(sccoda_data, modality_key=mod)  # result
    results["original"]["effects_credible"] = sccoda_model.credible_effects(
        sccoda_data, modality_key=mod)  # filter credible effects
    results["original"]["intercept"] = sccoda_model.get_intercept_df(
        sccoda_data, modality_key=mod)  # intercept df
    results["original"]["effects"] = sccoda_model.get_effect_df(
        sccoda_data, modality_key=mod)  # effects df
    if out_file is not None:
        sccoda_data.write_h5mu(out_file)
    if est_fdr is not None:
        sccoda_model.set_fdr(sccoda_data, modality_key=mod, 
                             est_fdr=est_fdr)  # adjust for expected FDR
        sccoda_model.summary(sccoda_data, modality_key=mod)
        results[f"fdr_{est_fdr}"]["intercept"] = sccoda_model.get_intercept_df(
            sccoda_data, modality_key=mod)  # intercept df
        results[f"fdr_{est_fdr}"]["effects"] = sccoda_model.get_effect_df(
            sccoda_data, modality_key=mod)  # effects df
        results[f"fdr_{est_fdr}"][
            "effects_credible"] = sccoda_model.credible_effects(
                sccoda_data, modality_key=mod)  # filter credible effects
        if out_file is not None:
            sccoda_data.write_h5mu(f"{out_file}_{est_fdr}_fdr")
    if plot is True:
        figs["proportions_stacked"] = pt.pl.coda.stacked_barplot(
            sccoda_data, modality_key=mod, 
            feature_name=col_perturbed)
        plt.show()
        figs["effects"] = pt.pl.coda.effects_barplot(
            sccoda_data, modality_key=mod, 
            parameter="Final Parameter")
        data_arviz = sccoda_model.make_arviz(sccoda_data, 
                                             modality_key="coda_salm")
        figs["mcmc_diagnostics"] = az.plot_trace(
            data_arviz, divergences=False,
            var_names=["alpha", "beta"],
            coords={"cell_type": data_arviz.posterior.coords["cell_type_nb"]}
        )
        plt.tight_layout()
        plt.show()
    return (results, figs)


def compute_distance(adata, col_target_genes="target_genes", 
                     col_cell_type="leiden",
                     distance_type="edistance", method="X_pca",
                     kws_plot=None, highlight_real_range=False, plot=True,
                     **kwargs):
    """Compute distance and hierarchies and (optionally) make heatmaps."""
    figs = {}
    distance = {}
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    if kws_plot is None:
        kws_plot = dict(robust=True, figsize=(10, 10))
        
    # Distance Metrics
    distance = pt.tl.Distance(distance_type, method)
    data = distance.pairwise(adata, groupby=col_target_genes, verbose=True)
    if plot is True:  # cluster heatmaps
        if highlight_real_range is True:
            vmin = np.min(np.ravel(data.values)[np.ravel(data.values) != 0])
            if "vmin" in kws_plot:
                warnings.warn(
                    f"""
                    vmin already set in kwargs plot: {kws_plot['vmin']}
                    Setting to {vmin} because highlight_real_range is True.""")
            kws_plot.update(dict(vmin=vmin))
        if "figsize" not in kws_plot:
            kws_plot["figsize"] = (20, 20)
        if "cmap" not in kws_plot:
            kws_plot["cmap"] = "Reds_r"
        figs[f"distance_heat_{distance_type}"] = clustermap(
            data, **kws_plot)
        plt.show()
        
    # Cluster Hierarchies
    dff = distance.pairwise(adata, groupby=col_cell_type, verbose=True)
    mat = linkage(dff, method="ward")
    if plot is True:  # cluster hierarchies
        _ = dendrogram(mat, labels=dff.index, orientation='left', 
                       color_threshold=0)
        plt.xlabel('E-distance')
        plt.ylabel('Leiden clusters')
        plt.gca().yaxis.set_label_position("right")
        plt.show()
        figs[f"distance_cluster_hierarchies_{distance_type}"] = plt.gcf()
    return distance, data, dff, mat, figs


def perform_gsea(adata, key_condition="Perturbed", 
                 filter_by_highly_variable=False, **kwargs):
    """Perform a gene set enrichment analysis (adapted from SC Best Practices)."""
    
    # Extract DEGs
    if filter_by_highly_variable is True:
        t_stats = (
            # Get dataframe of DE results for condition vs. rest
            sc.get.rank_genes_groups_df(adata, key_condition, key="t-test")
            .set_index("names")
            # Subset to highly variable genes
            .loc[adata.var["highly_variable"]]
            # Sort by absolute score
            .sort_values("scores", key=np.abs, ascending=False)
            # Format for decoupler
            [["scores"]]
            .rename_axis([key_condition], axis=1)
        )
    else:
        t_stats = (
            # Get dataframe of DE results for condition vs. rest
            sc.get.rank_genes_groups_df(adata, key_condition, key="t-test")
            .set_index("names")
            # Sort by absolute score
            .sort_values("scores", key=np.abs, ascending=False)
            # Format for decoupler
            [["scores"]]
            .rename_axis([key_condition], axis=1)
        )
    print(t_stats)
    
    # Retrieve Reactome Pathways
    msigdb = decoupler.get_resource("MSigDB")
    reactome = msigdb.query("collection == 'reactome_pathways'")
    reactome = reactome[~reactome.duplicated((
        "geneset", "genesymbol"))]  # filter duplicates
    
    # Filter Gene Sets for Compatibility
    geneset_size = reactome.groupby("geneset").size()
    gsea_genesets = geneset_size.index[(
        geneset_size > 15) & (geneset_size < 500)]
    scores, norm, pvals = decoupler.run_gsea(
        t_stats.T, reactome[reactome["geneset"].isin(gsea_genesets)], 
        source="geneset", target="genesymbol")

    # Rank Genes by T-Statistics
    gsea_results = (
        pd.concat({"score": scores.T, "norm": norm.T, "pval": pvals.T}, 
                axis=1).droplevel(level=1, axis=1).sort_values("pval"))
    # fig[] = so.Plot(data=(gsea_results.head(20).assign(
    #     **{"-log10(pval)": lambda x: -np.log10(x["pval"])})),
    #                 x="-log10(pval)", y="source").add(so.Bar())
    return gsea_results


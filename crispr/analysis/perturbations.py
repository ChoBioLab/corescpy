import pertpy as pt
import muon
import mudata
import arviz as az
import scanpy as sc
import functools
import matplotlib.pyplot as plt
import pandas as pd
    
    
def perform_mixscape(adata, label_perturbation,
                     assay=None, 
                     key_control="NT",
                     perturbation_type="KO",
                     split_by=None,
                     label_target_genes="gene_target", 
                     layer="X_pert", iter_num=10,
                     min_de_genes=5, pval_cutoff=5e-2, logfc_threshold=0.25,
                     n_comps_lda=None, 
                     plot=True, 
                     assay_protein=None,
                     protein_of_interest=None,
                     target_gene_idents=None):
    """Identify perturbed cells based on target genes (`adata.obs['mixscape_class']`,
    `adata.obs['mixscape_class_global']`) and calculate posterior probabilities
    (`adata.obs['mixscape_class_p_<perturbation_type>']`, e.g., KO) and
    perturbation scores. 
    Optionally, perform LDA to cluster cells based on perturbation response.
    Optionally, create figures related to differential gene 
    (and protein, if available) expression, perturbation scores, 
    and perturbation response-based clusters. 

    Args:
        adata (AnnData): Scanpy data object.
        label_perturbation (str): Perturbation category column  of `adata.obs` 
            (should contain key_control).
        key_control (str, optional): Control category key
            (`adata.obs[label_perturbation]` entries).Defaults to "NT".
        perturbation_type (str, optional): CRISPR perturbation type 
            to be expected by Mixscape classification labeling process. 
            Defaults to "KO".
        split_by (str, optional): `adata.obs` column name of replicates 
            to calculate them separately. Defaults to None.
        label_target_genes (str, optional): Name of column with target genes. 
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
        target_gene_idents (list, optional): List of names of genes 
            whose perturbations will determine cell grouping 
            for the above-described violin plot and/or
            whose differential expression posterior probabilities 
            will be plotted in a heatmap. Defaults to None.
        layer (str, optional): `adata.layers` slot name. Defaults to "X_pert".
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
            for PCA performed as part of LDA for pooled CRISPR screen data. 
            Defaults to None (no LDA performed).
        iter_num (float, optional): Iterations to run to converge if needed.
        plot (bool, optional): Make plots? Defaults to True.
    """
    figs = {}
    mix = pt.tl.Mixscape()
    mix.perturbation_signature(
        adata[assay] if assay else adata, label_perturbation, 
        key_control, split_by=split_by)  # perturbation signature
    adata_pert = adata[assay].copy() if assay else adata.copy()
    adata_pert.X = adata_pert.layers['X_pert']
    mix.mixscape(adata=adata[assay] if assay else adata, 
                 labels=label_target_genes, control=key_control, 
                 layer=layer, perturbation_type=perturbation_type,
                 min_de_genes=min_de_genes, pval_cutoff=pval_cutoff,
                 iter_num=iter_num)  # Mixscape classification
    if n_comps_lda is not None:
        mix.lda(adata=adata[assay] if assay else adata, 
                labels=label_target_genes, 
                layer=layer, control=key_control, min_de_genes=min_de_genes,
                split_by=split_by, perturbation_type=perturbation_type,
                mixscape_class_global="mixscape_class_global",
                n_comps=n_comps_lda, logfc_threshold=logfc_threshold,
                pval_cutoff=pval_cutoff)  # linear discriminant analysis (LDA) 
    if plot is True:
        figs["gRNA_targeting_efficiency_by_class"] = pt.pl.ms.barplot(
            adata[assay] if assay else adata, 
            guide_rna_column=key_control)  # targeting efficiency by condition 
        if n_comps_lda is not None:  # LDA clusters
            figs["cluster_perturbation_response"] = pt.pl.ms.lda(
                adata=adata[assay] if assay else adata, 
                control=key_control)  # perturbation response
        if target_gene_idents is not None:  # G/P EX
            figs["mixscape_de_ordered_by_ppp_heat"] = {}
            figs["mixscape_ppp_violin"] = {}
            figs["mixscape_perturb_score"] = {}
            for g in target_gene_idents:  # iterate target genes of interest
                figs["mixscape_perturb_score"][g] = pt.pl.ms.perturbscore(
                    adata = adata[assay] if assay else adata, 
                    labels=label_target_genes, target_gene=g, color="red")
                figs["mixscape_de_ordered_by_ppp_heat"][g] = pt.pl.ms.heatmap(
                    adata=adata[assay] if assay else adata, 
                    labels=label_target_genes, target_gene=g, 
                    layer=layer, control=key_control)  # DE heatmap
                tg_conds = [key_control, f"{g} NP", 
                            f"{g} {perturbation_type}"]  # conditions: gene g
                figs["mixscape_ppp_violin"][g] = pt.pl.ms.violin(
                    adata=adata[assay], 
                    target_gene_idents=tg_conds, 
                    groupby="mixscape_class")  # gene: perturbed, NP, control
                figs["mixscape_ppp_violin"][f"{g}_global"] = pt.pl.ms.violin(
                    adata=adata[assay] if assay else adata, 
                    target_gene_idents=[key_control, "NP", perturbation_type], 
                    groupby="mixscape_class_global")  # global: P, NP, control 
            tg_conds = [key_control] + functools.reduce(
                lambda i, j: i + j, [[f"{g} NP", 
                        f"{g} {perturbation_type}"] 
                for g in target_gene_idents])  # conditions: all genes
            figs["mixscape_ppp_violin"]["all"] = pt.pl.ms.violin(
                adata=adata[assay] if assay else adata, 
                target_gene_idents=tg_conds, 
                groupby="mixscape_class")  # gene: perturbed, NP, control
        if assay_protein is not None and target_gene_idents is not None:
            figs[f"mixscape_protein_{protein_of_interest}"] = pt.pl.ms.violin( 
                adata=adata[assay_protein], 
                target_gene_idents=target_gene_idents,
                keys=protein_of_interest, groupby=label_target_genes,
                hue="mixscape_class_global")
    return figs


def perform_augur(adata, assay=None, 
                  classifier="random_forest_classifier", 
                  augur_mode="default", subsample_size=20, n_threads=4, 
                  select_variance_features=False, 
                  label_col="label_col", label_cell_type="cell_type_col",
                  label_condition=None, treatment=None, 
                  seed=1618,
                  plot=True, **kws_augur_predict):
    """Calculates AUC using Augur and a specified classifier.

    Args:
        adata (AnnData): Scanpy object.
        assay (str, optional): Assay slot of adata ('rna' for `adata['rna']`).n
            Defaults to None (works if only one assay).
        classifier (str, optional): Classifier. Defaults to "random_forest_classifier".
        augur_mode (str, optional): Augur or permute? Defaults to "default".
        subsample_size (int, optional): Per Pertpy code: 
            "number of cells to subsample randomly per type 
            from each experimental condition"
        n_threads (int, optional): _description_. Defaults to 4.
        select_variance_features (bool, optional): Use Augur to select genes (True), or 
            Scanpy's  highly_variable_genes (False). Defaults to False.
        label_col (str, optional): _description_. Defaults to "label_col".
        label_cell_type (str, optional): Column name for cell type. 
            Defaults to "cell_type_col".
        label_condition (str, optional): Column name for experimental condition. 
            Defaults to None.
        treatment (str, optional): Name of value within label_condition. 
            Defaults to None.
        seed (int, optional): Random state (for reproducibility). 
            Defaults to 1618.
        plot (bool, optional): Plots? Defaults to True.
        kws_augur_predict (keywords, optional): Optional additional keyword 
            arguments to pass to Augur predict.

    Returns:
        _type_: _description_
    """
    figs = {}
    if select_variance_features == "both":  
        # both methods: select genes based on...
        # - original Augur (True)
        # - scanpy's highly_variable_genes (False)
        data, results, figs = [[None, None]] * 3  # to store results
        for i, x in enumerate([True, False]):  # iterate over methods
            data[i], results[i], figs[i] = perform_augur(
                adata[assay] if assay else adata, 
                subsample_size=subsample_size, 
                select_variance_features=x, 
                n_threads=n_threads,
                **kws_augur_predict)  # recursive -- run function both ways
            if plot is True:
                figs["vs_select_variance_features"] = pt.pl.ag.scatterplot(
                    results[0], results[1])  # compare  methods (diagonal=same)
    else:
        ag_rfc = pt.tl.Augur(classifier)
        loaded_data = ag_rfc.load(
            adata[assay] if assay else adata, 
            condition_label=label_condition, 
            treatment_label=treatment,
            cell_type_col=label_cell_type,
            label_col=label_col
            )  # add dummy variables, rename cell type & label columns
        data, results = ag_rfc.predict(
            loaded_data, subsample_size=subsample_size, augur_mode=augur_mode, 
            select_variance_features=select_variance_features,
            n_threads=n_threads, random_state=seed,
            **kws_augur_predict)  # AUGUR model prediction
        print(results["summary_metrics"])  # results summary
        if plot is True:  # plotting
            figs["lollipop"] = pt.pl.ag.lollipop(results)
            # TODO: UMAP?
            if ("umap" in adata[assay].uns) if assay else ("umap" in adata.uns):
                figs["umap_augur_score"] = sc.pl.umap(adata=data, 
                                                      color="augur_score")
            figs["important_features"] = pt.pl.ag.important_features(
                results)  # feature importances
    return data, results, figs


def perform_differential_prioritization(adata, label_condition, labels_treatments,
                                        label_col="label",
                                        assay=None,
                                        n_permutations=1000,
                                        n_subsamples=50,
                                        label_cell_type="cell_type",
                                        classifier="random_forest_classifier", 
                                        plot=True, **kws_augur_predict):
    """Determine which cell types were most/least perturbed (ranking).

    Args:
        adata (AnnData): Scanpy object.
        label_condition (str): Column used to indicate experimental condition.
        labels_treatments (str): List of two conditions 
            (values in label_condition).
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
        label_cell_type (str, optional): Column name for cell type. 
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
    augur_results, permuted_results = [], []  # to hold results
    if plot is True:
        figs["umap_augur"] = {}
    for x in labels_treatments:
        ag_rfc = pt.tl.Augur(classifier)
        ddd = ag_rfc.load(
            adata[assay] if assay else adata, 
            condition_label=label_condition, 
            treatment_label=x,
            cell_type_col=label_cell_type,
            label_col=label_col
            )  # add dummy variables, rename cell type & label columns
        ddd = ag_rfc.load(adata[assay] if assay else adata, 
                          condition_label=label_condition, treatment_label=x)
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
                        analysis_type="cell_level",
                        generate_sample_level=True, 
                        label_cell_type="cell_type",
                        sample_identifier="batch",
                        modality_key="coda",
                        label_condition="condition",
                        est_fdr=0.05,
                        plot=True,
                        out_file=None
                        ):
    """Perform SCCoda compositional analysis."""
    figs, results = {}, {}
    sccoda_model = pt.tl.Sccoda()
    sccoda_data = sccoda_model.load(
        adata, type=analysis_type, 
        generate_sample_level=generate_sample_level,
        cell_type_identifier=label_cell_type, 
        sample_identifier=sample_identifier, 
        covariate_obs=["condition"])  # load data
    print(sccoda_data)
    print(sccoda_data["coda"].X)
    print(sccoda_data["coda"].obs)
    if plot is True:
        figs[
            "find_reference"] = pt.pl.coda.rel_abundance_dispersion_plot(
                sccoda_data, modality_key=modality_key, 
                abundant_threshold=0.9)  # helps choose rference cell type
        figs["proportions"] = pt.pl.coda.boxplots(
            sccoda_data, modality_key=modality_key, 
            feature_name=label_condition, add_dots=True)
    sccoda_data = sccoda_model.prepare(
        sccoda_data, modality_key=modality_key, formula=label_condition,
        reference_cell_type=reference_cell_type)  # setup model
    sccoda_model.run_nuts(sccoda_data, 
                          modality_key=modality_key)  # no-U-turn HMV sampling 
    sccoda_model.summary(sccoda_data, modality_key=modality_key)  # result
    results["original"]["effects_credible"] = sccoda_model.credible_effects(
        sccoda_data, modality_key=modality_key)  # filter credible effects
    results["original"]["intercept"] = sccoda_model.get_intercept_df(
        sccoda_data, modality_key=modality_key)  # intercept df
    results["original"]["effects"] = sccoda_model.get_effect_df(
        sccoda_data, modality_key=modality_key)  # effects df
    if out_file is not None:
        sccoda_data.write_h5mu(out_file)
    if est_fdr is not None:
        sccoda_model.set_fdr(sccoda_data, modality_key=modality_key, 
                            est_fdr=est_fdr)  # adjust for expected FDR
        sccoda_model.summary(sccoda_data, modality_key=modality_key)
        results[f"fdr_{est_fdr}"]["intercept"] = sccoda_model.get_intercept_df(
            sccoda_data, modality_key=modality_key)  # intercept df
        results[f"fdr_{est_fdr}"]["effects"] = sccoda_model.get_effect_df(
            sccoda_data, modality_key=modality_key)  # effects df
        results[f"fdr_{est_fdr}"][
            "effects_credible"] = sccoda_model.credible_effects(
                sccoda_data, 
                modality_key=modality_key)  # filter credible effects 
        if out_file is not None:
            sccoda_data.write_h5mu(f"{out_file}_{est_fdr}_fdr")
    else:
        res_intercept_fdr, res_effects_fdr = None, None
    if plot is True:
        figs["proportions_stacked"] = pt.pl.coda.stacked_barplot(
            sccoda_data, modality_key=modality_key, 
            feature_name=label_condition)
        plt.show()
        figs["effects"] = pt.pl.coda.effects_barplot(
            sccoda_data, modality_key=modality_key, 
            parameter="Final Parameter")
        data_arviz = sccoda_model.make_arviz(sccoda_data, 
                                             modality_key="coda_salm")
        figs["mcmc_diagnostics"] = az.plot_trace(
            data_arviz, divergences=False,
            var_names=["alpha", "beta"],
            coords={"cell_type": data_arviz.posterior.coords["cell_type_nb"]},
        )
        plt.tight_layout()
        plt.show()
    return (results, figs)

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
import pandas as pd
import numpy as np
    
    
def perform_mixscape(adata, col_perturbation="perturbation",
                     key_control="NT",
                     label_perturbation_type="KO",
                     assay=None,
                     layer_perturbation=None,
                     col_guide_rna="guide_ID",
                     col_split_by=None,
                     col_target_genes="gene_target", 
                     iter_num=10,
                     min_de_genes=5, pval_cutoff=5e-2, logfc_threshold=0.25,
                     n_comps_lda=None, 
                     plot=True, 
                     assay_protein=None,
                     protein_of_interest=None,
                     target_gene_idents=None, **kwargs):
    """
    Identify perturbed cells based on target genes (`adata.obs['mixscape_class']`,
    `adata.obs['mixscape_class_global']`) and calculate posterior probabilities
    (`adata.obs['mixscape_class_p_<label_perturbation_type>']`, e.g., KO) and
    perturbation scores. 
    
    Optionally, perform LDA to cluster cells based on perturbation response.
    Optionally, create figures related to differential gene 
    (and protein, if available) expression, perturbation scores, 
    and perturbation response-based clusters. 
    
    Runs a differential expression analysis and creates a heatmap
    sorted by the posterior probabilities.

    Args:
        adata (AnnData): Scanpy data object.
        col_perturbation (str): Perturbation category column  of `adata.obs` 
            (should contain key_control).
        key_control (str, optional): Control category key
            (`adata.obs[col_perturbation]` entries).Defaults to "NT".
        label_perturbation_type (str, optional): CRISPR perturbation type 
            (purely for Mixscape labeling process, but is carried through).
            Defaults to "KO".
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
    """
    figs = {}
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    mix = pt.tl.Mixscape()
    mix.perturbation_signature(
        adata[assay] if assay else adata, col_perturbation, 
        key_control, split_by=col_split_by)  # perturbation signature
    # adata_pert = adata[assay].copy() if assay else adata.copy()
    # if layer_perturbation is not None:
    #     adata_pert.X = adata_pert.layers[layer_perturbation]
    mix.mixscape(adata=adata[assay] if assay else adata, 
                 # adata=adata_pert,
                 labels=col_target_genes, control=key_control, 
                 layer=layer_perturbation, 
                 perturbation_type=label_perturbation_type,
                 min_de_genes=min_de_genes, pval_cutoff=pval_cutoff,
                 iter_num=iter_num)  # Mixscape classification
    mix.lda(adata=adata[assay] if assay else adata, 
            # adata=adata_pert,
            labels=col_target_genes, 
            layer=layer_perturbation, control=key_control, 
            min_de_genes=min_de_genes,
            split_by=col_split_by, 
            perturbation_type=label_perturbation_type,
            mixscape_class_global="mixscape_class_global",
            n_comps=n_comps_lda, logfc_threshold=logfc_threshold,
            pval_cutoff=pval_cutoff)  # linear discriminant analysis (LDA)
    if plot is True:
        figs["perturbation_clusters"] = pt.pl.ms.lda(
            adata=adata[assay] if assay else adata, 
            control=key_control)  # cluster perturbation
    if target_gene_idents is True:  # to plot all target genes
        target_gene_idents = list(adata[assay].uns[
            "mixscape"].keys()) if assay else list(
                adata.uns["mixscape"].keys())  # target genes
    if plot is True:
        figs["gRNA_targeting_efficiency_by_class"] = pt.pl.ms.barplot(
            adata[assay] if assay else adata,
            # adata_pert, 
            guide_rna_column=key_control)  # targeting efficiency by condition 
        if n_comps_lda is not None:  # LDA clusters
            figs["cluster_perturbation_response"] = pt.pl.ms.lda(
                adata[assay] if assay else adata,
                # adata=adata_pert, 
                control=key_control)  # perturbation response
        if target_gene_idents is not None:  # G/P EX
            figs["mixscape_DEX_ordered_by_ppp_heat"] = {}
            figs["mixscape_ppp_violin"] = {}
            figs["mixscape_perturb_score"] = {}
            figs["mixscape_targeting_efficacy"] = pt.pl.ms.barplot(
                adata[assay] if assay else adata,
                # adata_pert, 
                guide_rna_column=col_guide_rna)
            for g in target_gene_idents:  # iterate target genes of interest
                if g not in list(adata[assay].obs["mixscape_class"] 
                                 if assay else adata.obs["mixscape_class"]):
                    print(f"\n\nTarget gene {g} not in mixscape_class!")
                    continue  # skip to next target gene if missing
                figs["mixscape_perturb_score"][g] = pt.pl.ms.perturbscore(
                    adata=adata[assay] if assay else adata,
                    # adata=adata_pert, 
                    labels=col_target_genes, target_gene=g, color="red")
                figs["mixscape_DEX_ordered_by_ppp_heat"][g] = pt.pl.ms.heatmap(
                    adata=adata[assay] if assay else adata,
                    # adata=adata_pert, 
                    labels=col_target_genes, target_gene=g, 
                    layer=layer_perturbation, control=key_control
                    )  # differential expression heatmap ordered by PPs
                tg_conds = [
                    key_control, f"{g} NP", 
                    f"{g} {label_perturbation_type}"]  # conditions: gene g
                figs["mixscape_ppp_violin"][g] = pt.pl.ms.violin(
                    adata=adata[assay] if assay else adata,
                    # adata=adata_pert, 
                    target_gene_idents=tg_conds, 
                    groupby="mixscape_class")  # gene: perturbed, NP, control
                figs["mixscape_ppp_violin"][f"{g}_global"] = pt.pl.ms.violin(
                    adata=adata[assay] if assay else adata,
                    # adata=adata_pert, 
                    target_gene_idents=[key_control, "NP", 
                                        label_perturbation_type], 
                    groupby="mixscape_class_global")  # global: P, NP, control 
            tg_conds = [key_control] + functools.reduce(
                lambda i, j: i + j, [[f"{g} NP", 
                        f"{g} {label_perturbation_type}"] 
                for g in target_gene_idents])  # conditions: all genes
            figs["mixscape_ppp_violin"]["all"] = pt.pl.ms.violin(
                adata=adata[assay] if assay else adata,
                # adata=adata_pert, 
                target_gene_idents=tg_conds, 
                groupby="mixscape_class")  # gene: perturbed, NP, control
        if assay_protein is not None and target_gene_idents is not None:
            figs[f"mixscape_protein_{protein_of_interest}"] = pt.pl.ms.violin( 
                adata=adata[assay] if assay else adata,
                # adata=adata_pert, 
                target_gene_idents=target_gene_idents,
                keys=protein_of_interest, groupby=col_target_genes,
                hue="mixscape_class_global")
    return figs


def perform_augur(adata, assay=None, layer_perturbation=None,
                  classifier="random_forest_classifier", 
                  augur_mode="default", 
                  subsample_size=20, n_folds=3,
                  select_variance_features=False, 
                  col_cell_type="leiden",
                  col_perturbation=None, 
                  col_gene_symbols="gene_symbols",
                  key_control="NT", key_treatment=None,
                  seed=1618, plot=True, n_threads=None,
                  kws_augur_predict=None, **kwargs):
    """Calculates AUC using Augur and a specified classifier.

    Args:
        adata (AnnData): Scanpy object.
        assay (str, optional): Assay slot of adata ('rna' for `adata['rna']`).n
            Defaults to None (works if only one assay).
        classifier (str, optional): Classifier. Defaults to "random_forest_classifier".
        augur_mode (str, optional): Augur or permute? Defaults to "default".
        subsample_size (int, optional): Per Pertpy code: 
            "number of cells to subsample randomly per type 
            from each experimental condition."
        n_folds (int, optional): Number of folds for cross-validation. Defaults to 3.
        n_threads (int, optional): _description_. Defaults to 4.
        select_variance_features (bool, optional): Use Augur to select genes (True), or 
            Scanpy's  highly_variable_genes (False). Defaults to False.
        col_cell_type (str, optional): Column name for cell type. 
            Defaults to "cell_type_col".
        col_perturbation (str, optional): Column name for experimental condition. 
            Defaults to None.
        key_control (str, optional): Control category key
            (`adata.obs[col_perturbation]` entries).Defaults to "NT".
        key_treatment (str, optional): Name of value within col_perturbation. 
            Defaults to None.
        seed (int, optional): Random state (for reproducibility). 
            Defaults to 1618.
        plot (bool, optional): Plots? Defaults to True.
        kws_augur_predict (dict, optional): Optional additional keyword 
            arguments to pass to Augur predict.

    Returns:
        _type_: _description_
    """
    # Setup
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    
    # Run Augur
    if select_variance_features == "both":  
        # both methods: select genes based on...
        # - original Augur (True)
        # - scanpy's highly_variable_genes (False)
        data, results = [[None, None]] * 2  # to store results
        figs = {}
        for i, x in enumerate([True, False]):  # iterate over methods
            data[i], results[i], figs[str(i)] = perform_augur(
                adata.copy(), assay=None, layer_perturbation=None,
                select_variance_features=x, classifier=classifier,
                augur_mode=augur_mode, subsample_size=subsample_size,
                n_threads=n_threads, n_folds=n_folds,
                col_cell_type=col_cell_type, col_perturbation=col_perturbation,
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
        adata_pert.obs[col_pert_new] = adata_pert.obs[col_perturbation].copy()
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
        
        # Plotting
        if plot is True:
            figs["perturbation_effect_by_cell_type"] = pt.pl.ag.lollipop(
                results)  # how affected each cell type is
            # TO DO: More Augur UMAP preprocessing options?
            sc.pp.neighbors(data)
            sc.tl.umap(data)
            figs["perturbation_effect_umap"] = sc.pl.umap(
                data, color=["augur_score", col_cell_type, col_perturbation],
                palette=str(kwargs["palette"] if "palette" in kwargs 
                            else "bright"))  # scores super-imposed on UMAP
            figs["important_features"] = pt.pl.ag.important_features(
                results)  # most important genes for prioritization
            figs["perturbation_scores"] = {}
    return data, results, figs


def perform_differential_prioritization(adata, col_perturbation="perturbation", 
                                        label_treatments="NT",
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
    runs of Augur (different values of col_perturbation).

    Args:
        adata (AnnData): Scanpy object.
        col_perturbation (str): Column used to indicate experimental condition.
        label_treatments (list): List of two conditions 
            (values in col_perturbation).
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
    for x in label_treatments:
        ag_rfc = pt.tl.Augur(classifier)
        ddd = ag_rfc.load(
            adata[assay] if assay else adata, 
            condition_label=col_perturbation, 
            treatment_label=x,
            cell_type_col=col_cell_type,
            label_col=label_col
            )  # add dummy variables, rename cell type & label columns
        ddd = ag_rfc.load(adata[assay] if assay else adata, 
                          condition_label=col_perturbation, treatment_label=x)
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
                        col_perturbation="condition",
                        est_fdr=0.05,
                        plot=True,
                        out_file=None, **kwargs):
    """Perform SCCoda compositional analysis."""
    figs, results = {}, {}
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    sccoda_model = pt.tl.Sccoda()
    sccoda_data = sccoda_model.load(
        adata, type=analysis_type, 
        generate_sample_level=generate_sample_level,
        cell_type_identifier=col_cell_type, 
        sample_identifier=sample_identifier, 
        covariate_obs=["condition"])  # load data
    print(sccoda_data)
    print(sccoda_data["coda"].X)
    print(sccoda_data["coda"].obs)
    if plot is True:
        figs[
            "find_reference"] = pt.pl.coda.rel_abundance_dispersion_plot(
                sccoda_data, modality_key=assay, 
                abundant_threshold=0.9)  # helps choose rference cell type
        figs["proportions"] = pt.pl.coda.boxplots(
            sccoda_data, modality_key=assay, 
            feature_name=col_perturbation, add_dots=True)
    sccoda_data = sccoda_model.prepare(
        sccoda_data, modality_key=assay, formula=col_perturbation,
        reference_cell_type=reference_cell_type)  # setup model
    sccoda_model.run_nuts(sccoda_data, 
                          modality_key=assay)  # no-U-turn HMV sampling 
    sccoda_model.summary(sccoda_data, modality_key=assay)  # result
    results["original"]["effects_credible"] = sccoda_model.credible_effects(
        sccoda_data, modality_key=assay)  # filter credible effects
    results["original"]["intercept"] = sccoda_model.get_intercept_df(
        sccoda_data, modality_key=assay)  # intercept df
    results["original"]["effects"] = sccoda_model.get_effect_df(
        sccoda_data, modality_key=assay)  # effects df
    if out_file is not None:
        sccoda_data.write_h5mu(out_file)
    if est_fdr is not None:
        sccoda_model.set_fdr(sccoda_data, modality_key=assay, 
                            est_fdr=est_fdr)  # adjust for expected FDR
        sccoda_model.summary(sccoda_data, modality_key=assay)
        results[f"fdr_{est_fdr}"]["intercept"] = sccoda_model.get_intercept_df(
            sccoda_data, modality_key=assay)  # intercept df
        results[f"fdr_{est_fdr}"]["effects"] = sccoda_model.get_effect_df(
            sccoda_data, modality_key=assay)  # effects df
        results[f"fdr_{est_fdr}"][
            "effects_credible"] = sccoda_model.credible_effects(
                sccoda_data, 
                modality_key=assay)  # filter credible effects 
        if out_file is not None:
            sccoda_data.write_h5mu(f"{out_file}_{est_fdr}_fdr")
    if plot is True:
        figs["proportions_stacked"] = pt.pl.coda.stacked_barplot(
            sccoda_data, modality_key=assay, 
            feature_name=col_perturbation)
        plt.show()
        figs["effects"] = pt.pl.coda.effects_barplot(
            sccoda_data, modality_key=assay, 
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


def compute_distance(adata, col_perturbation="perturbation", 
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
    data = distance.pairwise(adata, groupby=col_perturbation, verbose=True)
    if plot is True:  # cluster heatmaps
        if highlight_real_range is True:
            vmin = np.min(np.ravel(data.values)[np.ravel(data.values) != 0])
            if "vmin" in kws_plot:
                warnings.warn(
                    f"""
                    vmin already set in kwargs plot: {kws_plot['vmin']}
                    Setting to {vmin} because highlight_real_range is True.""")
            kws_plot.update(dict(vmin=vmin))
        figs[f"distance_heat_{distance_type}"] = clustermap(data, **kws_plot)
        plt.show()
    # Cluster Hierarchies
    dff = distance.pairwise(adata, groupby=col_cell_type, verbose=True)
    mat = linkage(dff, method="ward")
    if plot is True:  # cluster hierarchies
        hierarchy = dendrogram(mat, labels=dff.index, orientation='left', 
                               color_threshold=0)
        plt.xlabel('E-distance')
        plt.ylabel('Leiden clusters')
        plt.gca().yaxis.set_label_position("right")
        plt.show()
        figs[f"distance_cluster_hierarchies_{distance_type}"] = plt.gcf()
    return distance, data, dff, mat, figs


def perform_gsea(adata, label_condition, filter_by_highly_variable=False):
    """Perform a gene set enrichment analysis (adapted from SC Best Practices)."""
    
    # Extract DEGs
    if filter_by_highly_variable is True:
        t_stats = (
            # Get dataframe of DE results for condition vs. rest
            sc.get.rank_genes_groups_df(adata, label_condition, key="t-test")
            .set_index("names")
            # Subset to highly variable genes
            .loc[adata.var["highly_variable"]]
            # Sort by absolute score
            .sort_values("scores", key=np.abs, ascending=False)
            # Format for decoupler
            [["scores"]]
            .rename_axis([label_condition], axis=1)
        )
    else:
        t_stats = (
            # Get dataframe of DE results for condition vs. rest
            sc.get.rank_genes_groups_df(adata, label_condition, key="t-test")
            .set_index("names")
            # Sort by absolute score
            .sort_values("scores", key=np.abs, ascending=False)
            # Format for decoupler
            [["scores"]]
            .rename_axis([label_condition], axis=1)
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


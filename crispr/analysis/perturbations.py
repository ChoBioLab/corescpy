import pertpy as pt
import numpy as np


def calculate_targeting_efficiency(adata, assay=None, guide_rna_column="NT"):
    """_summary_

    Args:
        adata (_type_): _description_
        assay (_type_, optional): _description_. Defaults to None.
        guide_rna_column (str, optional): _description_. Defaults to "NT".

    Returns:
        _type_: _description_
    """
    figs = {}  # for figures
    figs.update({"barplot": pt.pl.ms.barplot(adata[assay] if assay else adata, 
                            guide_rna_column=guide_rna_column)})
    return figs


def calculate_perturbations(adata, target_gene, target_gene_idents, 
                            assay=None, control="NT", 
                            color="green", plot=True):
    """Calculate perturbation scores (from Pertpy Mixscape tutorial)."""
    
    # Identify Cells without Detectible Pertubations
    mix = pt.tl.Mixscape()  # mixscape object
    mix.perturbation_signature(adata[assay] if assay else adata, 
                               "perturbation", "NT", "replicate")  # signatures
    mix.mixscape(adata=adata[assay] if assay else adata, control=control, 
                 labels="gene_target", layer="X_pert")  # mixscape routine
    mix.lda(adata=adata[assay] if assay else adata, labels="gene_target", 
            layer="X_pert")  # linear discriminant analysis (LDA)
    
    # Cell Perturbation Scores
    figs = {}
    fig_ps = pt.pl.ms.perturbscore(adata=adata[assay] if assay else adata, 
                                   labels='gene_target', 
                                   target_gene=target_gene,
                                   color=color)  # plot perturbation scores
    if plot is True: 
        figs.update({"perturbation_scores": fig_ps})
    if plot is True:
        fig_ppp = pt.pl.ms.violin(adata=adata[assay] if assay else adata, 
                                  target_gene_idents=target_gene_idents,
                                  groupby="mixscape_class")  # plot PPPs
        figs.update({"PPP": fig_ppp})
        fig_dehm = pt.pl.ms.heatmap(adata=adata[assay] if assay else adata, 
                                    labels="gene_target", 
                                    target_gene=target_gene,
                                    layer="X_pert", control=control)  # plot DE
        figs.update({"DE_heatmap": fig_dehm})
        fig_lda = pt.pl.ms.lda(
            adata=adata[assay] if assay else adata)  # plot LDA
        figs.update({"lda": fig_lda})
    if plot is True:
        return figs
    
    
def perform_augur(adata, classifier="random_forest_classifier", 
                  augur_mode="default", subsample_size=20, n_threads=4, 
                  select_variance_features=False, 
                  label_col="label_col", label_cell_type="cell_type_col",
                  label_condition=None, label_treatment=None, 
                  seed=1618,
                  plot=True, **kws_augur_predict):
    """Calculates AUC using Augur and a specified classifier.

    Args:
        adata (AnnData): Scanpy object.
        classifier (str, optional): Classifier. Defaults to "random_forest_classifier".
        augur_mode (str, optional): Augur or permute? Defaults to "default".
        subsample_size (int, optional): Per Pertpy code: 
            "number of cells to subsample randomly per type from each experimental condition"
        n_threads (int, optional): _description_. Defaults to 4.
        select_variance_features (bool, optional): Use Augur to select genes (True), or 
            Scanpy's  highly_variable_genes (False). Defaults to False.
        label_col (str, optional): _description_. Defaults to "label_col".
        label_cell_type (str, optional): Column name for cell type. Defaults to "cell_type_col".
        label_condition (str, optional): Column name for experimental condition. Defaults to None.
        label_treatment (str, optional): Name of value within label_condition. Defaults to None.
        seed (int, optional): Random state (for reproducibility). Defaults to 1618.
        plot (bool, optional): Plots? Defaults to True.
        kws_augur_predict (keywords, optional): Optional keyword arguments to pass to Augur predict.

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
                adata, subsample_size=subsample_size, 
                select_variance_features=x, 
                n_threads=n_threads,
                **kws_augur_predict)  # recursive -- run function both ways
            if plot is True:
                figs["vs_select_variance_features"] = pt.pl.ag.scatterplot(
                    results[0], results[1])  # compare  methods (diagonal=same)
    else:
        ag_rfc = pt.tl.Augur(classifier)
        loaded_data = ag_rfc.load(
            adata, condition_label=label_condition, 
            treatment_label=label_treatment,
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
            # sc.pp.neighbors(adata, use_rep="X")
            # sc.tl.umap(adata)
            # figs["umap"] = sc.pl.umap(adata=data, color="augur_score")
            figs["important_features"] = pt.pl.ag.important_features(
                results)  # feature importances
    return data, results, figs


def perform_differential_prioritization(adata, label_condition, labels_treatments,
                                        classifier="random_forest_classifier", 
                                        plot=True, **kws_augur_predict):
    """Determine which cell types were most/least perturbed (ranking).

    Args:
        adata (AnnData): Scanpy object.
        label_condition (str): Column used to indicate experimental condition.
        labels_treatments (str): List of two conditions (values in label_condition).
        classifier (str, optional): Classifier. Defaults to "random_forest_classifier".
        plot (bool, optional): Plots? Defaults to True.

    Returns:
        _type_: _description_
    """
    figs = {}
    augur_results, permuted_results = [], []  # to hold results
    for x in labels_treatments:
        ag_rfc = pt.tl.Augur(classifier)
        ddd = ag_rfc.load(adata, condition_label=label_condition, treatment_label=x)
        dff_a, res_a = ag_rfc.predict(ddd, augur_mode="augur", 
                                      **kws_augur_predict)  # augur
        dff_p, res_p = ag_rfc.predict(ddd, augur_mode="permute", 
                                      **kws_augur_predict)  # permute
        augur_results.append([res_a])
        permuted_results.append([res_p])
    pvals = ag_rfc.predict_differential_prioritization(
        augur_results1=augur_results[0],
        augur_results2=augur_results[1],
        permuted_results1=permuted_results[0],
        permuted_results2=permuted_results[1]
        )
    if plot is True:
        figs["diff"] = pt.pl.ag.dp_scatter(pvals)
    return pvals, figs
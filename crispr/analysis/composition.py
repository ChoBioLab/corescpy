#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Analyses focused on cell type composition analyses.

@author: E. N. Aslinger
"""

import pertpy as pt
import arviz as az
import warnings
import matplotlib.pyplot as plt


def analyze_composition(
    adata, col_condition,  col_cell_type,
    assay=None, layer=None,
    reference_cell_type="automatic", 
    analysis_type="cell_level",  # only for scCoda
    generate_sample_level=True,
    col_list_lineage_tree=None,  # only for TASCCoda
    col_sample_id=None, covariates=None,
    est_fdr=0.05, key_treatment="Perturbed", key_control="Control", 
    key_reference_cell_type="automatic", 
    plot=True, out_file=None, copy=False, **kwargs):
    """Perform SCCoda compositional analysis.
        copy (bool, optional): If False (default), 
            modify adata in place; otherwise, copy the object.

    Returns:
        tuple: results summary, dictionary of figures, modified adata
    """
    figs, results = {}, {}
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    if copy is True:
        adata = adata.copy()
    if layer is not None:
        adata.X = adata.layers[layer]
    if col_list_lineage_tree is None:  # scCoda
        results, figures, adata = perform_sccoda(
            adata, col_condition, col_cell_type, covariates=covariates,
            reference_cell_type=reference_cell_type, assay=assay, 
            analysis_type=analysis_type, col_cell_type=col_cell_type,
            generate_sample_level=generate_sample_level, est_fdr=est_fdr, 
            sample_identifier=col_sample_id, plot=plot, out_file=out_file)
    else:  # TASCCODA
        results, figures, adata = perform_tasccoda(
            adata, col_condition, col_cell_type, col_list_lineage_tree, 
            key_treatment=key_treatment, key_control=key_control, 
            col_sample_id=col_sample_id,
            key_reference_cell_type=key_reference_cell_type, 
            covariates=covariates, plot=True)
        
    return (results, figures, adata)


def perform_sccoda(
    adata, col_condition, col_cell_type, 
    assay=None, reference_cell_type="automatic", 
    analysis_type="cell_level", 
    generate_sample_level=True, sample_identifier="batch", 
    covariates=None, est_fdr=0.05, plot=True, out_file=None):
    """Perform SCCoda compositional analysis."""
    figs, results = {}, {}
    if generate_sample_level is True and sample_identifier is None:
        warnings.warn("""
                      Can't generate sample level if `sample_identifier`=None. 
                      Setting `generate_sample_level` to False.
                      """)
        generate_sample_level = False
    mod = "coda"
    covariate_obs = [covariates] + col_condition if isinstance(covariates, str
        ) else covariates + [col_condition] if covariates else [col_condition]
    model = pt.tl.Sccoda()
    scodata = model.load(
        (adata[assay] if assay else adata).copy(), 
        type=analysis_type, modality_key_1=assay if assay else None,
        modality_key_2=mod, generate_sample_level=generate_sample_level,
        cell_type_identifier=col_cell_type, 
        covariate_obs=covariate_obs, 
        sample_identifier=sample_identifier)  # load data
    # mod = assay if assay else list(set(sccoda_data.mod.keys()).difference(
    #     set(["coda"])))[0]  # original modality
    if plot is True:
        figs[
            "find_reference"] = pt.pl.coda.rel_abundance_dispersion_plot(
                scodata, modality_key=mod, 
                abundant_threshold=0.9)  # helps choose rference cell type
        figs["proportions"] = pt.pl.coda.boxplots(
            scodata, modality_key=mod, 
            feature_name=col_condition, add_dots=True)
    scodata = model.prepare(scodata, modality_key=mod, formula=col_condition,
                            reference_cell_type=reference_cell_type)  # setup
    print(scodata)
    print(scodata["coda"].X)
    print(scodata["coda"].obs)
    model.run_nuts(scodata, modality_key=mod)  # no-U-turn HMV sampling 
    model.summary(scodata, modality_key=mod)  # result
    results["original"]["effects_credible"] = model.credible_effects(
        scodata, modality_key=mod)  # filter credible effects
    results["original"]["intercept"] = model.get_intercept_df(
        scodata, modality_key=mod)  # intercept df
    results["original"]["effects"] = model.get_effect_df(
        scodata, modality_key=mod)  # effects df
    if out_file is not None:
        scodata.write_h5mu(out_file)
    if est_fdr is not None:
        model.set_fdr(scodata, modality_key=mod, 
                      est_fdr=est_fdr)  # adjust for expected FDR
        model.summary(scodata, modality_key=mod)
        results[f"fdr_{est_fdr}"]["intercept"] = model.get_intercept_df(
            scodata, modality_key=mod)  # intercept df
        results[f"fdr_{est_fdr}"]["effects"] = model.get_effect_df(
            scodata, modality_key=mod)  # effects df
        results[f"fdr_{est_fdr}"][
            "effects_credible"] = scodata.credible_effects(
                scodata, modality_key=mod)  # filter credible effects
        if out_file is not None:
            scodata.write_h5mu(f"{out_file}_{est_fdr}_fdr")
    if plot is True:
        figs["proportions_stacked"] = pt.pl.coda.stacked_barplot(
            scodata, modality_key=mod, feature_name=col_condition)
        plt.show()
        figs["effects"] = pt.pl.coda.effects_barplot(
            scodata, modality_key=mod, parameter="Final Parameter")
        data_arviz = model.make_arviz(scodata, modality_key=mod)
        figs["mcmc_diagnostics"] = az.plot_trace(
            data_arviz, divergences=False,
            var_names=["alpha", "beta"],
            coords={"cell_type": data_arviz.posterior.coords["cell_type_nb"]}
        )
        plt.tight_layout()
        plt.show()
    return (results, figs, scodata)


def perform_tasccoda(adata, col_condition, col_cell_type, 
                     col_list_lineage_tree,
                     col_sample_id=None, covariates=None, 
                     key_treatment="Perturbed", key_control="Control", 
                     key_reference_cell_type="automatic",
                     plot=True):
    """
    Perform tree-aggregated compositional analysis (TASCCODA) in order
    to investigate differences in cell type composition 
    between conditions (e.g., healthy vs. diseased, 
    CRISPR KD versus untreated, etc.) and plot these effects 
    along the lineage tree.

    Args:
        adata (AnnData): AnnData object. Expects the gene expression 
            modality, so if you have multi-modal data, be sure to
            subset (e.g., `adata[]`) when you pass it to the function.
        col_condition (str): Name of the column containing 
            the condition labels you wish to use in analysis. 
        col_cell_type (str): Name of the column containing 
            the cell type labels you wish to use in analysis.
        col_sample_id (str or list): Name(s) of the column(s) 
            containing sample identifies (e.g., `"sample_id"` or
            `["sample_id", "batch"]`), if applicable. 
            The default is None.
        col_list_lineage_tree (list): Names of the columns 
            containing the lineage tree (i.e., multiple columns 
            that contain increasing specificity) leading to the 
            specific cell type contained in `col_cell_type`. 
            For instance, the first in the list may be a column
            of "epiethlial" versus "immune," the second may be
            "epithelial" versus "myeloid" versus "lymphoid," etc, with 
            `col_cell_type` being "enterocytes," "macrophages," etc.
        key_treatment (str, optional): Label in `col_condition` that 
            corresponds to the treatment/comparison condition. 
            Defaults to "Perturbed".
        key_control (str, optional): Label in `col_condition` that 
            corresponds to the control/reference comparison condition. 
            Defaults to "Control".
        reference_cell_type (str, optional): The cell type label in 
            `col_cell_type` to use as the reference cell type. 
            Defaults to "automatic" 
            (will automatically detect a suitable reference, i.e., 
            that has the smallest composition effects).
        covariates (list, optional): Additional covariates 
            (in addition). Defaults to None.
        plot (bool, optional): Plots? Defaults to True.
    """
    # Setup
    figs, results = {}, {}
    model = pt.tl.Tasccoda()
    covariates = [covariates] if isinstance(covariates, str) else list(
        covariates) if covariates else None  # ensure covariates list or None
    ts_data = model.load(
        adata, type="cell_level", cell_type_identifier=col_cell_type,
        sample_identifier=col_sample_id, 
        covariate_obs=covariates + [col_condition], 
        levels_orig=col_list_lineage_tree, 
                         add_level_name=True)
    pt.pl.coda.draw_tree(ts_data["coda"])
    ts_data.mod["coda_subset"] = ts_data["coda"][ts_data["coda"].obs[
        col_condition].isin([key_control, key_treatment])]  # subset if needed
    if plot is True:
        figs["tree"] = pt.pl.coda.draw_tree(ts_data["coda"])
        figs["descriptives_abundance"] = pt.pl.coda.boxplots(
            ts_data, modality_key="coda_subset", feature_name=col_condition, 
            figsize=(20, 8))
        plt.show()
    
    # TASCCODA
    ts_data = model.prepare(
        ts_data, modality_key="coda_subset", tree_key="tree", 
        reference_cell_type=key_reference_cell_type, formula=col_condition, 
        pen_args={"phi": 0})  # prepare model
    model.run_nuts(ts_data, modality_key="coda_subset")  # NUTS
    results["summary"] = model.summary(
        ts_data, modality_key="coda_subset")  # NUTS results summary
    results["credible_effects"] = model.credible_effects(
        ts_data, modality_key="coda_subset")  # credible effects
    if plot:
        figs["credible_effects"] = pt.pl.coda.draw_effects(
            ts_data, modality_key="coda_subset", 
            tree=ts_data["coda_subset"].uns["tree"], 
            covariate=f"{col_condition}[T.{key_treatment}]"
            )  # effects as sizes/colors of nodes on the lineage tree
        figs["credible_effects_dual"] = pt.pl.coda.draw_effects(
            ts_data, modality_key="coda_subset", 
            tree=ts_data["coda_subset"].uns["tree"], 
            covariate=f"{col_condition}[T.{key_treatment}]", 
            show_legend=False, show_leaf_effects=True)
    return results, figs, ts_data
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Analyses focused on single-cell gene expression data.

@author: E. N. Aslinger
"""

import pertpy as pt
import matplotlib.pyplot as plt


def perform_tasccoda(adata, col_condition, 
                     col_cell_type, col_list_lineage_tree,
                     key_treatment="Perturbed", key_control="Control", 
                     reference_cell_type="automatic", covariates=None, 
                     plot=True, copy=False):
    """
    Perform tree-aggregated compositional analysis (TASCCODA) in order
    to investigate differences in cell type composition 
    between conditions (e.g., healthy vs. diseased, 
    CRISPR KD versus untreated, etc.).

    Args:
        adata (AnnData): AnnData object. Expects the gene expression 
            modality, so if you have multi-modal data, be sure to
            subset (e.g., `adata[]`) when you pass it to the function.
        col_cell_type (str): Name of the column containing 
            the condition labels you wish to use in analysis. 
        col_cell_type (str): Name of the column containing 
            the cell type labels you wish to use in analysis.
        col_list_lineage_tree (list): Names of the columns 
            containing the lineage tree (i.e., multiple columns 
            that contain increasing specificity) leading to the 
            specific cell type contained in `col_cell_type`. 
            For instance, the first in the list may be a column
            of "epiethlial" versus "immune," the second may be
            "epithelial" versus "myeloid" versus "lymphoid," etc, with 
            `col_cell_type` being "enterocytes," "macrophages," etc.
        key_treatment (str, optional): _description_. Defaults to "Perturbed".
        key_control (str, optional): _description_. Defaults to "Control".
        reference_cell_type (str, optional): _description_. Defaults to "automatic".
        covariates (_type_, optional): _description_. Defaults to None.
        plot (bool, optional): Plots? Defaults to True.
        copy (bool, optional): If False (default), 
            modify adata in place; otherwise, copy the object.

    Returns:
        tuple: results summary, dictionary of figures, modified adata
    """
    # Setup
    figs = {}
    if copy is True:
        adata = adata.copy()
    model = pt.tl.Tasccoda()
    covariates = [covariates] if isinstance(covariates, str) else list(
        covariates) if covariates else None  # ensure covariates list or None
    ts_data = model.load(
        adata, type="cell_level", cell_type_identifier=col_cell_type,
        sample_identifier=["Subject", "Sample"], 
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
    
    # TASCODA
    ts_data = model.prepare(
        ts_data, modality_key="coda_subset", tree_key="tree", 
        reference_cell_type=reference_cell_type, formula=col_condition, 
        pen_args={"phi": 0})  # prepare model
    model.run_nuts(ts_data, modality_key="coda_subset")  # NUTS
    results = model.summary(
        ts_data, modality_key="coda_subset")  # NUTS results summary
    model.credible_effects(
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
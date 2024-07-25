#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Analyses focused on cell type composition analyses.

@author: E. N. Aslinger
"""

import pandas as pd
import numpy as np


def classify_gex_cells(adata, col_cell_type=None, genes=None,
                       layer="counts", threshold=0):
    """
    Classify expression, namely, the number and percent of cells
    (by cluster if `col_cell_type` is not None)
    expressing above (>, not >=)`threshold` level of gene expression
    (default metric is counts) for at least `min_genes` genes
    (the default is all genes in the combination).
    """
    adata = adata.copy()
    if isinstance(genes, str):
        genes = [genes]
    if layer:
        adata.X = adata.layers[layer].copy()
    if genes is None:
        genes = list(adata.var_names)  # quantify all genes if unspecified
    if col_cell_type is None:  # just calculate overall if unspecified
        col_cell_type = "Cluster"
        adata.obs.loc[:, col_cell_type] = "Overall"
    quants = {}
    for gene in genes:
        if gene not in adata.var_names:
            print(f"Gene '{gene}' not found in adata.var_names")
            continue
        obs = adata.obs.copy()
        gex = adata[:, gene].X > threshold
        if "toarray" in dir(gex):
            gex = gex.toarray()
        obs.loc[:, "gene_positive"] = gex.flatten()
        grouped = obs.groupby(col_cell_type)["gene_positive"]
        result = grouped.agg(["sum", "count"])
        result.columns = ["N_Cells_Positive", "N_Cluster"]
        result.loc[:, "Percent"] = 100 * result["N_Cells_Positive"] / result[
            "N_Cluster"]
        quants[gene] = result
    quants = pd.concat(quants, names=["Gene"])
    quants.loc[:, "N_Sample"] = adata.obs.shape[0]
    return quants


def classify_coex_cells(adata, col_cell_type=None, genes=None,
                        layer="counts", threshold=0, min_genes="all"):
    """
    Classify coexpression (by cluster if `col_cell_type` is not None),
    namely, the number and percent of cells
    expressing above (>, not >=)`threshold` level of gene expression
    (default metric is counts) for at least `min_genes` genes
    (the default is all genes in the combination).
    The `genes` argument can also be a list of list to quantify several
    groups of gene combinations.
    """
    if layer:
        adata.X = adata.layers[layer].copy()
    if isinstance(genes[0], (list, np.ndarray, set, tuple)):
        quants = pd.concat([classify_coex_cells(
            adata, col_cell_type=col_cell_type, genes=g, layer=layer,
            threshold=threshold, min_genes=min_genes) for g in genes], keys=[
                "/".join(g) for g in genes], names=["Gene"])
        return quants
    if genes is None:
        genes = [list(adata.var_names)]  # quantify all genes if unspecified
    if min_genes in ["all", None]:
        min_genes = len(genes)  # require all genes coexpressed if unspecified
    if col_cell_type is None:  # just calculate overall if unspecified
        col_cell_type = "Cluster"
        adata.obs.loc[:, col_cell_type] = "Overall"
    coex = adata[:, genes].X > threshold
    if "toarray" in dir(coex):
        coex = coex.toarray()
    coex = pd.DataFrame(coex, index=adata.obs.index, columns=genes)
    coex = coex.sum(axis=1) >= min_genes
    adata.obs["coexpressed"] = coex
    grouped = adata.obs.groupby(col_cell_type)["coexpressed"]
    quants = grouped.agg(["sum", "count"])
    quants.columns = ["N_Cells_Positive", "N_Cluster"]
    quants.loc[:, "Percent"] = 100 * quants["N_Cells_Positive"] / quants[
        "N_Cluster"]
    quants.loc[:, "N_Sample"] = adata.obs.shape[0]
    quants = quants.assign(min_genes=min_genes)
    return quants


def classify_tx(adata, genes=None, col_cell_type=None, layer="counts"):
    """Quantify transcript counts (optionally, by cluster)."""
    adata = adata.copy()
    if isinstance(genes, str):
        genes = [genes]
    if layer:
        adata.X = adata.layers[layer].copy()
    if genes is None:
        genes = list(adata.var_names)  # quantify all genes if unspecified
    if col_cell_type is None:  # just calculate overall if unspecified
        col_cell_type = "Cluster"
        adata.obs.loc[:, col_cell_type] = "Overall"
    results = {}
    for gene in genes:  # iterate genes
        if gene not in adata.var_names:
            print(f"*** {gene} not in adata.var_names")
            continue
        gex = adata[:, gene].X
        if "toarray" in dir(gex):
            gex = gex.toarray()
        gex = pd.DataFrame(gex, index=adata.obs.index, columns=[gene])
        results[gene] = gex.join(adata.obs[[col_cell_type]]).set_index(
            col_cell_type).groupby(adata.obs[col_cell_type]).sum()[gene]
    txs_cts = pd.concat(results, axis=1)
    return txs_cts

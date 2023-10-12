#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""

import scanpy as sc
# import subprocess
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pertpy as pt
import crispr as cr
from crispr.defaults import (names_layers)
import pandas as pd
import numpy as np

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Crispr(object):
    """An object class for CRISPR analysis and visualization."""
    
    _columns_created = dict(guide_percent="Percent of Cell Guides")

    def __init__(self, file_path, 
                 assay=None, 
                 assay_protein=None,
                 col_gene_symbols="gene_symbols", 
                 col_cell_type="leiden",
                 col_sample_id="standard_sample_id", 
                 col_batch=None,
                 col_perturbation="perturbation",
                 col_target_genes=None,
                 col_guide_rna="guide_ids",
                 col_num_umis="num_umis",
                 key_control="NT", 
                 key_treatment="KO", 
                 key_nonperturbed="NP", 
                 kws_process_guide_rna=None,
                 remove_multi_transfected=True,
                 **kwargs):
        """_summary_Args:
            file_path (str, AnnData, or dictionary): 
                Path to a 10x directory 
                    (with matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz),
                path to an .h5ad or .mu file (Scanpy/AnnData/Muon-compatible),
                an AnnData or MuData object (e.g., already loaded with Scanpy),
                    or
                a dictionary containing keyword arguments to pass to
                `crispr.pp.combine_matrix_protospacer()` 
                (in order to load information about perturbations from
                other file(s); see function documentation).
            assay (str, optional): Name of the gene expression assay if
                loading a multi-modal data object (e.g., "rna"). 
                Defaults to None (i.e., `self.adata` is single-modal).
            assay_protein (str, optional): Name of the assay containing 
                the protein expression modality, if available. For instance, 
                if `assay_protein="adt"`, self.adata["mod"] would be expected
                to contain the AnnData object for the protein modality.
                ONLY FOR MULTI-MODAL DATA for certain bonus visualization
                methods. Defaults to None.
            col_gene_symbols (str, optional): Column name in `.var` 
                for gene symbols. Defaults to "gene_symbols".
            col_cell_type (str, optional): Column name in `.obs` for cell type, 
                whether pre-existing in data (e.g., pre-run clustering column 
                or manual annotations) or a column expected to be 
                created via Crispr.cluster(). Defaults to "leiden" 
                (e.g., if you expect this column to be created 
                by running `self.cluster(...)` with `method_cluster="leiden"`).
                In certain methods, you can specify a new column to use just
                    for that function. For instance, if you have a column
                    containing CellTypist annotations and want to use
                    those clusters instead of the "leiden" ones for 
                    the `run_dialogue()` method, you can specify in that method
                    without changing the attribute that contains your 
                    original specification here.
            col_sample_id (str, optional): Column in `.obs` with sample IDs. 
                Defaults to "standard_sample_id".
            col_batch (str, optional): Column in `.obs` with batch IDs. 
                Defaults to None.
            col_perturbation (str, optional): Column where class methods will
                be able to find the experimental/perturbation condition, 
                whether
                    (a) that column is already in `self.adata.obs` 
                        (when the file is loaded in  as an object, 
                        if `file` is a file path), or 
                    (b) that column must be created by specifying 
                    `kws_process_guide_rna` as a dictionary of arguments 
                    (which'll be passed to `crispr.pp.filter_by_guide_counts()`)
                    in order to process the guide IDs (see function docstring) 
                    and make a binary perturbed/control column 
                    (with entries in that column following your specifications 
                    in `key_control` and `key_treatment` arguments).
                    All entries containing the patterns specified in
                    `kws_process_guide_rna["key_control_patterns"]` will
                    be changed to `key_control`, and all cells with
                    targeting guides will be changed to `key_treatment`. 
                If a pre-existing column, should consist of entries 
                that are either `key_control` or `key_treatment`
                (i.e., a binary version of `col_target_genes` and `col_guide_rna`). 
                Defaults to "perturbation".
            col_target_genes (str, optional): The column where each guide RNA's
                target gene is or will be stored, whether pre-existing or 
                (if `kws_process_guide_rna` is not None) created 
                during the Crispr object initialization by passing 
                `kws_process_guide_rna` to
                `crispr.pp.filter_by_guide_counts()`. Defaults to None.
            col_guide_rna (str, optional): Column in `.obs` with guide RNA IDs.
                Entries in this column should be gene names in 
                `self.adata.var_names`, plus, optionally, 
                suffixes separating guide #s (e.g., STAT1-1-2, CTRL-1) and/or
                with a character that splits separate guide RNAs within that cell
                (if multiply-transfected cells are present). These characters
                should be specified in `kws_process_guide_rna["guide_split"]`
                and `kws_process_guide_rna["feature_split"]`, respectively.
                For instance, they would be "-" and "|", if `col_guide_rna`
                entries for a cell multiply transfected by two sgRNAs 
                targeting STAT1, two control guide RNAs, and a guide targeting 
                CCL5  would look like "STAT1-1-1|STAT1-1-2|CNTRL-1-1|CCL5-1".
                Currently, everything after the first dash is discarded when
                creating `col_target_genes`, so keep that in mind.
                This column will be stored as `<col_guide_rna>_original` if 
                `kws_process_guide_rna` is not None, as that will result in 
                a processed version of this column being stored under 
                `.obs[<col_guide_rna>]`. Defaults to "guide_ids".
            col_num_umis (str, optional): Name of column in `.obs` with the 
                UMI counts. This should be specified if `kws_process_guide_rna`
                is not None. Defaults to "num_umis".
            key_control (str, optional): The label in `col_target_genes`
                and in `col_guide_rna`
                The name you want the control 
                entries to be categorized as under the new `col_guide_rna`. 
                for instance, `CNTRL-1`, `NEGCNTRL`, etc. would all be replaced 
                by "Control" if that's what you specify here. 
                Defaults to "NT".
            key_treatment (str, optional): What entries in `col_perturbation`
                indicate a treatment condition (e.g., drug administration, 
                CRISPR knock-out/down), etiher already in `.obs` in the object
                as passed to `file` or loaded using hte file specified in `file`,
                or to be used in the column(s) created by 
                `crispr.pp.filter_by_guide_counts()` 
                if `kws_process_guide_rna` is not None.
                Defaults to "KO".
            key_nonperturbed (str, optional): What will be stored in the 
                `mixscape_class_global` and related columns/labels after
                running Mixscape methods for cells without a detectible
                perturbation. Defaults to "NP".
            kws_process_guide_rna (dict, optional): Dictionary of keyword
                arguments to pass to `crispr.pp.filter_by_guide_counts()`. 
                (See below and crispr.processing.preprocessing documentation). 
                Defaults to None.
            remove_multi_transfected (bool, optional): In designs with 
                multiple guides per cell, remove multiply-transfected cells
                (i.e., cells where more than one target guide survived 
                application of any filtering criteria set 
                in `kws_process_guide_rna`).
                If `kws_process_guide_rna["max_percent_umis_control_drop"]`
                    is greater than 0, then cells with one target guide
                    and control guides which together make up less than 
                    `max_percent_umis_control_drop`% of total UMI counts 
                    will be considered pseudo-single-transfected 
                    for the target guide.
                Defaults to True. Some functionality may be limited if
                set to False and if multiply-transfected cells remain in data. 
            
        Notes:
            kws_process_guide_rna:
                key_control_patterns (list, optional): List (or single string) 
                    of patterns in guide RNA column entries that correspond to a 
                    control. For instance, if control entries in the original 
                    `col_guide_rna` column include `NEGCNTRL` and
                    `Control.D`, you should specify ['Control', 'CNTRL'] 
                    (assuming no non-control sgRNA names contain those patterns). 
                    If blank entries should be interpreted as control guides, 
                    then include np.nan/numpy.nan in this list.
                    Defaults to None, but you almost certainly shouldn't leave this 
                    blank if you're using this function.
            max_percent_umis_control_drop (int, optional): If control UMI counts 
                are less than or equal to this percentage of the total counts for 
                that cell, and if a non-control sgRNA is also present and 
                meets other filtering criteria, then consider that cell 
                pseudo-single-transfected (non-control gene). Defaults to 75.
            min_percent_umis (int, optional): sgRNAs with counts below this 
                percentage will be considered noise for that guide. 
                Defaults to 40.
            feature_split (str, optional): For designs with multiple guides,
                the character that splits guide names in `col_guide_rna`. 
                For instance, "|" for `STAT1-1|CNTRL-1|CDKN1A`. Defaults to "|".
                If only single guides, set to None.
            guide_split (str, optional): The character that separates 
                guide (rather than gene target)-specific IDs within gene. 
                For instance, guides targeting STAT1 may include 
                STAT1-1, STAT1-2, etc.; the argument would be "-" so the 
                function can identify all of those as targeting 
                STAT1. Defaults to "-".
        """
        print("\n\n<<<INITIALIZING CRISPR CLASS OBJECT>>>\n")
        self._assay = assay
        self._assay_protein = assay_protein
        self._layer_perturbation = "X_pert"
        self._file_path = file_path
        if kwargs:
            print(f"\nUnused keyword arguments: {kwargs}.\n")
        
        # Create Object & Attributes to Store Results/Figures
        self.adata = cr.pp.create_object(
            self._file_path, assay=None, col_gene_symbols=col_gene_symbols)
        self.figures = {"main": {}}
        self.results = {"main": {}}
        self._info = {"descriptives": {}, 
                      "guide_rna": {}}  # extra info to store by methods
        print(self.adata.obs, "\n\n") if assay else None
        print("\n\n", self.adata[assay].obs if assay else self.adata)
        
        # Process Guide RNAs (optional)
        self._info["guide_rna"]["keywords"] = kws_process_guide_rna
        if kws_process_guide_rna is not None:
            print("\n\n<<<PERFORMING gRNA PROCESSING AND FILTERING>>>\n")
            tg_info, feats_n = cr.pp.filter_by_guide_counts(
                self.adata[assay] if assay else self.adata, 
                col_guide_rna, col_num_umis=col_num_umis,
                key_control=key_control, **kws_process_guide_rna
                )  # process (e.g., multi-probe names) & filter by sgRNA counts
            kws_process_guide_rna["feature_split"] = tg_info.feature_split.iloc[0]
            self._info["guide_rna"]["keywords"] = kws_process_guide_rna
            self._info["guide_rna"]["counts_unfiltered"] = feats_n
            for q in [col_guide_rna, 
                      col_num_umis]:  # replace w/ processed entries
                tg_info.loc[:, q] = tg_info[q + "_filtered"]
            if remove_multi_transfected is True:
                self._info["guide_rna"]["counts_single_multi"] = tg_info.copy()
                tg_info = tg_info.loc[tg_info[
                    f"{col_guide_rna}_list_filtered"].apply(
                        lambda x: np.nan if len(x) > 1 else x).dropna().index]
            self._info["guide_rna"]["counts"] = tg_info.copy()
            if assay:
                self.adata[assay].obs = self.adata[assay].obs.join(
                    tg_info, lsuffix="_original")
                self.adata[assay].obs.loc[:,  col_target_genes] = self.adata[
                    assay].obs[col_guide_rna]  
                # ^ col_target_genes=processed col_guide_rna 
                if remove_multi_transfected is True:
                    self.adata[assay] = self.adata[assay][
                        ~self.adata[assay].obs[col_guide_rna].isnull()]
            else:
                self.adata.obs = self.adata.obs.join(
                    tg_info, lsuffix="_original")
                if remove_multi_transfected is True:
                    self.adata = self.adata[
                        ~self.adata.obs[col_guide_rna].isnull()]
                self.adata.obs.loc[:,  col_target_genes] = self.adata.obs[
                    col_guide_rna]  # col_target_genes=processed col_guide_rna 
                if remove_multi_transfected is True:
                    # drop cells w/ multiple guides that survived filtering
                    self.adata = self.adata[
                        ~self.adata.obs[col_guide_rna].isnull()]
                    
            # Perturbed vs. Untreated Column
            if col_perturbation not in self.adata.obs:
                print("\n\n<<<CREATING PERTURBED/CONTROL COLUMN>>>\n")
                if assay:
                    self.adata[assay].obs.loc[
                        :, col_perturbation] = self.adata[assay].obs[
                            col_target_genes].apply(
                                lambda x: key_control if pd.isnull(x) or (
                                    x == key_control) else key_treatment)
                else:
                    self.adata.obs.loc[:, col_perturbation] = self.adata.obs[
                        col_target_genes].apply(
                            lambda x: key_control if pd.isnull(x) or (
                                x == key_control) else key_treatment)
        
            # Store Columns & Keys within Columns as Dictionary Attributes
            if col_target_genes is None:  # if unspecified...
                col_target_genes = col_guide_rna  # ...= guide ID (maybe processed)
            self._columns = dict(col_gene_symbols=col_gene_symbols,
                                 col_cell_type=col_cell_type, 
                                 col_sample_id=col_sample_id, 
                                 col_batch=col_batch,
                                 col_perturbation=col_perturbation,
                                 col_guide_rna=col_guide_rna,
                                 col_num_umis=col_num_umis,
                                 col_target_genes=col_target_genes)
            self._keys = dict(key_control=key_control, 
                              key_treatment=key_treatment,
                              key_nonperturbed=key_nonperturbed)
            for q in [self._columns, self._keys]:
                cr.tl.print_pretty_dictionary(q)
            print("\n\n", self.adata[assay].obs if assay else self.adata)
        
            # Correct 10x CellRanger Guide Count Incorporation
            # raise NotImplementedError(
            #     "Delete guides from var_names not yet implemented!")

    # @property
    # def adata(self):
    #     """Specify file path to data."""
    #     return self._adata

    # @adata.setter
    # def adata(self, value):
    #     """Set file path and load object."""
    #     self._file_path = value
    #     if isinstance(value, str):
    #         print("\n\n<<<CREATING OBJECT>>>")
    #         self._adata = cr.pp.create_object(
    #             value, assay=None, col_gene_symbols=self._columns[
    #                 "col_gene_symbols"])
    #     else:
    #         self._adata = value
    
    def describe(self, group_by=None, plot=False):
        """Describe data."""
        desc, figs = {}, {}
        gbp = [self._columns["col_cell_type"]]
        if group_by:
            gbp += [group_by]
        if "descriptives" not in self._info:
            self._info["descriptives"] = {}
        print("\n\n\n", self.adata.obs.describe().round(2),
              "\n\n\n")
        print(f"\n\n{'=' * 80}\nDESCRIPTIVES\n{'=' * 80}\n\n")
        print(self.adata.obs.describe())
        for g in gbp:
            print(self.adata.obs.groupby(g).describe().round(2), 
                  f"\n\n{'-' * 40}\n\n") 
        try:
            if "guide_rna_counts" not in self._info["descriptives"]:
                self._info["descriptives"][
                    "guide_rna_counts"] = self.count_gRNAs()
            desc["n_grnas"] = self._info["descriptives"][
                "guide_rna_counts"].groupby("gene").describe().rename(
                    {"mean": "mean sgRNA count/cell"})
            print(desc["n_grnas"])
            if plot is True:
                n_grna = self._info["descriptives"][
                    "guide_rna_counts"].to_frame("sgRNA Count")
                if group_by is not None:
                    n_grna = n_grna.join(self.adata.obs[
                        group_by].rename_axis("bc"))  # join group_by variable
                figs["n_grna"] = sns.catplot(data=n_grna.reset_index(),
                    y="gene", x="sgRNA Count", kind="violin", hue=group_by,
                    height=len(self._info["descriptives"][
                        "guide_rna_counts"].reset_index().gene.unique())
                    )  # violin plot of gNRA counts per cell
        except Exception as err:
            warnings.warn(f"{err}\n\n\nCould not describe sgRNA count.")
        # try:
        print(f"\n\n{'=' * 80}\nCELL COUNTS\n{'=' * 80}\n")
        print(f"\t\t\tTotal Cells: {self.adata.shape[0]}")
        desc["n_cells"], figs["n_cells"] = {}, {}
        desc["n_cells"]["all"] = self.adata.shape[0]
        for g in gbp:
            if g in self.adata.obs:
                print(f"\t\t\n***** By {g} *****\n")
                desc["n_cells"][g] = self.adata.obs.groupby(g).apply(
                    lambda x: x.shape[0])
                print("\t\t\t\n", dict(desc["n_cells"][g].T))
                if plot is True:
                    figs["n_cells"][g] = sns.catplot(
                        data=desc["n_cells"][g].to_frame("N").reset_index(), 
                        y="N", kind="bar", x=g, hue=g)  # bar plot of cell #s
        # except Exception as err:
        #     warnings.warn(f"{err}\n\n\nCould not describe cell counts.")
        self._info["descriptives"].update(desc)
        return figs
    
    def get_guide_rna_counts(self, target_gene_idents=None, group_by=None,
                             col_cell_type=None, **kwargs):
        """Plot guide RNA counts by cell type & up to 1 other variable."""
        if isinstance(target_gene_idents, str):
            target_gene_idents = [target_gene_idents]
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if group_by is None:
            group_by = [col_cell_type]
        if isinstance(group_by, str):
            group_by = [group_by]
        if group_by is True:
            group_by = [col_cell_type]
        if len(group_by) > 1:
            kwargs.update({"row": group_by[1]})
        if "share_x" not in kwargs: 
            kwargs.update({"share_x": True})
        # sufxs = ["all", "filtered"]
        # cols = [self._columns[x] for x in ["col_guide_rna", "col_num_umis"]] 
        # grna = [self._info["guide_rna"]["counts_raw"][
        #     [f"{i}_list_{x}" for i in cols]] for x in sufxs]
        # for g in grna:
        #     g.columns = cols
        # grna = pd.concat(grna, keys=sufxs, names=["version", "bc"])
        # dff = grna.loc["all"].apply(
        #     lambda x: pd.Series([np.nan] * x.shape[0], index=x.index) 
        #     if len(pd.unique(x[cols[0]])) > 1 else pd.Series(
        #         pd.unique(x[cols[1]], index=pd.unique(x[cols[0]]))), 
        #         axis=1).dropna()
        # dff = dff.apply(
        #     lambda x: pd.Series(x[cols[1]], index=x[cols[0]]), 
        #     axis=1).stack().sort_index().rename_axis(
        #         ["b", "Target"]).to_frame("N gRNAs").join(
        #             self.adata.obs[[col_cell_type]].rename_axis(
        #                 "b")).reset_index()
        dff = self._info["guide_rna"]["counts_unfiltered"].reset_index(
            "Gene").rename({"Gene": "Target"}, axis=1).join(
                self.adata.obs[group_by])  # gRNA counts + cell types
        if target_gene_idents:
            dff = dff[dff.Target.isin(target_gene_idents)]
        if group_by is not None:
            kwargs.update({"col": group_by[0]})
        if len(group_by) == 1:
            kwargs.update({"col_wrap": cr.pl.square_grid(len(
                dff[group_by[0]].unique()))[1]})
        # fig = sns.catplot(data=dff[dff.Target.isin(
        #     target_gene_idents)] if target_gene_idents else dff, 
        #             x="N gRNAs", y="Target", col_wrap=cr.pl.square_grid(
        #                 len(self.adata.obs[col_cell_type].unique()))[1],
        #             col=col_cell_type, kind="violin", 
        #             **kwargs)
        fig = sns.catplot(data=dff, y="Target",
                          x=self._columns_created["guide_percent"],
                          kind="violin", **kwargs)
        fig.fig.suptitle("Guide RNA Counts by Cell Type")
        fig.fig.tight_layout()
        return self._info["guide_rna"]["counts_unfiltered"], fig
    
    def preprocess(self, assay=None, assay_protein=None,
                   clustering=False, run_label="main", 
                   remove_doublets=True, **kwargs):
        """Preprocess data."""
        if assay_protein is None:
            assay_protein = self._assay_protein
        if assay is None:
            assay = self._assay
        self.adata, figs = cr.pp.process_data(
            self.adata, assay=assay, assay_protein=assay_protein, 
            remove_doublets=remove_doublets, **self._columns,
            **kwargs)  # preprocess
        if run_label not in self.figures:
            self.figures[run_label] = {}
        self.figures[run_label]["preprocessing"] = figs
        if clustering not in [None, False]:
            if clustering is True:
                clustering = {}
            if isinstance(clustering, dict):
                figs_cl = self.cluster(**clustering)  # clustering
                figs = figs + [figs_cl]
            else:
                raise TypeError(
                    "`clustering` must be dict (keyword arguments) or bool.")
        return figs
                
    def cluster(self, assay=None, method_cluster="leiden", 
                plot=True, colors=None, paga=False, 
                kws_pca=None, kws_neighbors=None, 
                kws_umap=None, kws_cluster=None, 
                run_label="main", test=False, **kwargs):
        """Perform dimensionality reduction and create UMAP."""
        if assay is None:
            assay = self._assay
        figs_cl = cr.ax.cluster(
            self.adata.copy() if test is True else self.adata, 
            assay=assay, method_cluster=method_cluster,
            **self._columns, **self._keys,
            plot=plot, colors=colors, paga=paga, 
            kws_pca=kws_pca, kws_neighbors=kws_neighbors,
            kws_umap=kws_umap, kws_cluster=kws_cluster, **kwargs)
        if test is False:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"clustering": figs_cl})
        return figs_cl
    
    def annotate_clusters(self, model, **kwargs):
        """Use CellTypist to annotate clusters."""
        preds = cr.ax.perform_celltypist(
            self.adata[self._assay] if self._assay else self.adata,
            model, majority_voting=True, **kwargs)  # annotate
        self.results["celltypist"] = preds
        self.adata.obs = self.adata.obs.join(preds.predicted_labels, 
                                             lsuffix="_last")
        if self._assay:
            self.adata[self._assay].obs = self.adata[self._assay].obs.join(
                preds.predicted_labels)
        sc.pl.umap(self.adata, color=[self._columns["col_cell_type"]] + list(
            preds.predicted_labels.columns))  # UMAP
        return preds
    
    def find_markers(self, assay=None, n_genes=5, layer="scaled", 
                     method="wilcoxon", key_reference="rest", 
                     plot=True, col_cell_type=None, **kwargs):
        if assay is None:
            assay = self._assay
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        marks, figs_m = cr.ax.find_markers(
            self.adata, assay=assay, method=method, n_genes=n_genes, 
            layer=layer, key_reference=key_reference, plot=plot,
            col_cell_type=col_cell_type, **kwargs)
        print(marks)
        return marks, figs_m
        
    def run_mixscape(self, assay=None, col_cell_type=None,
                     col_perturbation=None,
                     target_gene_idents=True, 
                     col_split_by=None, min_de_genes=5,
                     run_label="main",
                     test=False, plot=True, **kwargs):
        """
        Identify/classify and quantify perturbation of cells.
        
        Optionally, perform LDA to cluster cells 
        based on perturbation response.
        Optionally, create figures related to differential gene 
        (and protein, if available) expression, perturbation scores, 
        and perturbation response-based clusters. 
        
        Runs a differential expression analysis and creates a heatmap
        sorted by the posterior probabilities.

        Args:
            adata (AnnData): Scanpy data object.
            col_perturbation (str): The name of the column containing 
                the perturbation information. If not provided, 
                `self._columns["col_perturbation"]` will be used.
                Usually, this argument should not be provided. 
                Allowing specification here rather than just 
                using `self._columns["col_perturbation"]`
                is just meant to allow the user maximum flexibility.
            col_cell_type (str, optional): Column name in `.obs` for cell type.
                If unspecified, will use `self._columns["col_cell_type"]`.
            col_split_by (str, optional): `adata.obs` column name of 
                sample categories to calculate separately (e.g., replicates).
                Defaults to None.
            assay (str, optional): Assay slot of adata 
                ('rna' for `adata['rna']`). If not provided, will use
                self._assay. Typically, users should not specify this argument.
            assay_protein (str, optional): Protein assay slot name 
                (if available). If True, use `self._assay_protein`. If None,
                don't run extra protein expression analyses, even if 
                protein expression modality is available.
            protein_of_interest (str, optional): If assay_protein is not None  
                and plot is True, will allow creation of violin plot of 
                protein expression (y) by 
                <target_gene_idents> perturbation category (x),
                split/color-coded by Mixscape classification 
                (`adata.obs['mixscape_class_global']`). Defaults to None.
            target_gene_idents (list or bool, optional): List of names of 
                genes whose perturbations will determine cell grouping 
                for the above-described violin plot and/or
                whose differential expression posterior probabilities 
                will be plotted in a heatmap. Defaults to None.
                True to plot all in `self.adata.uns["mixscape"]`.
            min_de_genes (int, optional): Minimum number of genes a cell has
                to express differentially to be labeled 'perturbed'. 
                For Mixscape and LDA (if applicable). Defaults to 5.
            pval_cutoff (float, optional): Threshold for significance 
                to identify differentially-expressed genes. 
                For Mixscape and LDA (if applicable). Defaults to 5e-2.
            logfc_threshold (float, optional): Will only test genes whose 
                average logfold change across the two cell groups is 
                at least this number. For Mixscape and LDA (if applicable). 
                Defaults to 0.25.
            n_comps_lda (int, optional): Number of principal components
                for PCA. Defaults to None (LDA not performed).
            iter_num (float, optional): Iterations to run 
                in order to converge (if needed).
            plot (bool, optional): Make plots? Defaults to True.
            
        Returns:
            dict: A dictionary containing figures visualizing
                results, for instance, 
        
        Notes:
            - Classifications are stored in 
            `self.adata.obs['mixscape_class_global']`
            (detectible perturbation (`self._keys["key_treatment"]`) vs. 
            non-detectible perturbation (`self._keys["key_nonperturbed"]`) 
            vs. control (`self._keys["key_control"]`)) and in
            `self.adata.obs['mixscape_class']` (like the previous, 
                but specific to target genes, e.g., "STAT1 KO" vs. just "KO"). 
            - Posterial p-values are stored in 
                `self.adata.obs['mixscape_class_p_<key_treatment>']`.
            - Other result output will be stored in `self.figures` and 
                `self.results` under the `run_label` key under `mixscape`
                (e.g., `self.figures["main"]["mixscape"]`).
            
        """
        if assay is None:
            assay = self._assay
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if col_perturbation is None:
            col_perturbation = self._columns["col_perturbation"]
        figs_mix = cr.ax.perform_mixscape(
            self.adata.copy() if test is True else self.adata, assay=assay,
            **{**self._columns, "col_perturbation": col_perturbation,
               "col_cell_type": col_cell_type}, **self._keys,
            layer_perturbation=self._layer_perturbation, 
            target_gene_idents=target_gene_idents,
            min_de_genes=min_de_genes, col_split_by=col_split_by, 
            plot=plot, **kwargs)
        if test is False:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"mixscape": figs_mix})
        return figs_mix
    
    def run_augur(self, assay=None, col_cell_type=None,
                  col_perturbation=None, key_treatment=None,
                  augur_mode="default", classifier="random_forest_classifier", 
                  kws_augur_predict=None, n_folds=3,
                  subsample_size=20, n_threads=True, 
                  select_variance_features=False, 
                  seed=1618, plot=True, run_label="main", test=False,
                  **kwargs):        
        """
        Runs the Augur perturbation scoring and prediction analysis.

        Parameters:
            assay (str): The name of the gene expression assay 
                (for multi-modal data objects). 
                If not provided, `self._assay` will be used.
            col_perturbation (str): The name of the column containing 
                the perturbation information. If not provided, 
                `self._columns["col_perturbation"]` will be used.
                Usually, this argument should not be provided. 
                Allowing specification here rather than just 
                using `self._columns["col_perturbation"]`
                is just meant to allow the user maximum flexibility,
                but it is expected the most users will not need
                to specify this argument.
            key_treatment (str): The key used to identify the treatment group
                or cells perturbed by targeting guide RNAs.
                If not provided, the default key will be used...which, again, 
                is the expected scenario, but the argument is available
                for rare cases where needed. For instance,
                for experimental conditions that have
                multiple levels (e.g., control, drug A treatment, 
                drug B treatment), allowing this argument 
                (and `col_perturbation`, for instance, 
                if self._columns["col_perturbation"] 
                is a binary treatment vs. drug column, and you want to use 
                the more specific column for Augur)
                to be specified allows for more flexibility in analysis.
            augur_mode (str, optional): Augur or permute? 
                Defaults to "default". (See pertpy documentation.)
            classifier (str): The classifier to be used for the analysis. 
                Defaults to "random_forest_classifier".
            kws_augur_predict (dict): Additional keyword arguments to be 
                passed to the `perform_augur()` function.
            n_folds (int): The number of folds to be used for cross-validation.
                Defaults to 3. Should be >= 3 as a matter of best practices.
            subsample_size (int, optional): Per Pertpy code: 
                "number of cells to subsample randomly per type 
                from each experimental condition." Defaults to 20.
            n_threads (bool): The number of threads to be used for 
                parallel processing. If set to True, the available 
                CPUs minus 1 will be used. Defaults to True.
            select_variance_features (bool, optional): Use Augur to select 
                genes (True), or Scanpy's  highly_variable_genes (False). 
                Defaults to False.
            seed (int): The random seed to be used for reproducibility. 
                Defaults to 1618.
            plot (bool): Whether to plot the results. Defaults to True.
            run_label (str): The label for the current run. Defaults to "main".
            test (bool): Whether the function is being called as a test run. 
                If True,self.adata will be copied so 
                that no alterations are made inplace via Augur,
                and results are not stored in object attributes.
                Defaults to False.
            **kwargs: Additional keyword arguments to be passed to 
                the `crispr.ax.perform_augur()` function.

        Returns:
            tuple: A tuple containing the AnnData object, results, and figures 
                generated by the Augur analysis.
        """
        if key_treatment is None:
            key_treatment = self._keys["key_treatment"]
        if col_perturbation is None:
            col_perturbation = self._columns["col_perturbation"]
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if n_threads is True:
            n_threads = os.cpu_count() - 1 # use available CPUs - 1
        if assay is None:
            assay = self._assay
        if kws_augur_predict is None:
            kws_augur_predict = {}
        # if run_label != "main":
        #     kws_augur_predict.update(
        #         {"key_added": f"augurpy_results_{run_label}"}
        #         )  # run label incorporated in key in adata
        data, results, figs_aug = cr.ax.perform_augur(
            self.adata.copy() if test is True else self.adata, 
            assay=assay, classifier=classifier, 
            augur_mode=augur_mode, subsample_size=subsample_size,
            select_variance_features=select_variance_features, 
            n_folds=n_folds,
            **{**self._columns, "col_perturbation": col_perturbation,
               "col_cell_type": col_cell_type},  
            kws_augur_predict=kws_augur_predict,
            key_control=self._keys["key_control"], key_treatment=key_treatment,
            layer=self._layer_perturbation,
            seed=seed, n_threads=n_threads, plot=plot, **kwargs)
        if test is False:
            if run_label not in self.results:
                self.results[run_label] = {}
            self.results[run_label].update(
                {"Augur": {"results": results, "data": data}})
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"Augur": figs_aug})
        return data, results, figs_aug
    
    def compute_distance(self, distance_type="edistance", method="X_pca", 
                         kws_plot=None, highlight_real_range=False,
                         run_label="main", plot=True):
        """
        Compute and visualize distance metrics.

        Args:
            distance_type (str, optional): The type of distance 
                calculation to perform. Defaults to "edistance".
            method (str, optional): The method to use for 
                dimensionality reduction. Defaults to "X_pca".
            kws_plot (dict, optional): Additional keyword arguments 
                for plotting. Defaults to None.
            highlight_real_range (bool, optional): Whether to highlight 
                the real range by setting minimum and maximum color scaling
                based on properties of the data. Defaults to False.
            run_label (str, optional): The label for the current run. Affects
                the key under which results and figures are stored in internal 
                attributes. Defaults to "main".
            plot (bool, optional): Whether to create plots. 
                Defaults to True.

        Returns:
            output: A tuple containing output from `cr.ax.compute_distance()`, 
                including distance matrices, figures, etc. 
                See function documentation.

        """
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if col_perturbation is None:
            col_perturbation = self._columns["col_perturbation"]
        output = cr.ax.compute_distance(
            self.adata, **{
                **self._columns, "col_perturbation": col_perturbation,
                "col_cell_type": col_cell_type}, 
            **self._keys, distance_type=distance_type, method=method,
            kws_plot=kws_plot, highlight_real_range=highlight_real_range, 
            plot=plot)
        if plot is True:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label]["distances"] = {}
            self.figures[run_label]["distances"].update(
                {f"{distance_type}_{method}": output[-1]})
            if run_label not in self.results:
                self.results[run_label] = {}
            self.results[run_label]["distances"] = {}
            self.results[run_label]["distances"].update(
                {f"{distance_type}_{method}": output})
        return output
    
    def run_gsea(self, key_condition=None, 
                 filter_by_highly_variable=False, 
                 run_label="main", **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        output = cr.ax.perform_gsea(
            self.adata, filter_by_highly_variable=filter_by_highly_variable, 
            **{**self._keys, "key_condition": self._keys[
                "key_condition"] if key_condition is None else key_condition
               }, **self._columns, **kwargs)  # GSEA
        if run_label not in self.results:
            self.results[run_label] = {}
        self.results[run_label]["gsea"] = output
        return output
          
    def run_composition_analysis(self, reference_cell_type, 
                                 assay=None, analysis_type="cell_level", 
                                 col_cell_type=None,
                                 col_perturbation=None,
                                 est_fdr=0.05, generate_sample_level=False,
                                 plot=True, run_label="main", **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        if col_perturbation is None:
            col_perturbation = self._columns["col_perturbation"]
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        output = cr.ax.analyze_composition(
            self.adata, reference_cell_type,
            assay=assay if assay else self._assay, analysis_type=analysis_type,
            generate_sample_level=generate_sample_level, 
            est_fdr=est_fdr, plot=plot, 
            **{**self._columns, "col_cell_type": col_cell_type,
               "col_perturbation": col_perturbation}, 
            **self._keys, **kwargs)
        if run_label not in self.results:
            self.results[run_label] = {}
        self.results[run_label]["composition"] = output
        return output
    
    def run_dialogue(self, n_programs=3, col_cell_type=None,
                     cmap="coolwarm", vcenter=0, 
                     run_label="main", **kws_plot):
        """Analyze <`n_programs`> multicellular programs."""
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        d_l = pt.tl.Dialogue(
            sample_id=self._columns["col_perturbation"], n_mpcs=n_programs,
            celltype_key=col_cell_type, 
            n_counts_key=self._columns["col_num_umis"])
        pdata, mcps, ws, ct_subs = d_l.calculate_multifactor_PMD(
            self.adata.copy(), normalize=True)
        mcp_cols = list(set(pdata.obs.columns).difference(
            self.adata.obs.columns))
        cols = cr.pl.square_grid(len(mcp_cols) + 2)[1]
        fig = sc.pl.umap(
            pdata, color=mcp_cols + [
                self._columns["col_perturbation"], col_cell_type],
            ncols=cols, cmap=cmap, vcenter=vcenter, **kws_plot)
        self.results[run_label]["dialogue"] = pdata, mcps, ws, ct_subs
        self.figures[run_label]["dialogue"] = fig
        return fig
            
    def plot(self, genes=None, genes_highlight=None,
             cell_types_circle=None,
             assay="default", 
             title=None,  # NOTE: will apply to multiple plots
             layers="all",
             marker_genes_dict=None,
             kws_gex=None, kws_clustering=None, 
             kws_gex_violin=None, kws_gex_matrix=None, 
             run_label="main"):
        """Create a variety of plots."""
        # Setup  Arguments & Empty Output
        figs = {}
        if kws_clustering is None:
            kws_clustering = {"frameon": False, "legend_loc": "on_data"}
        if "legend_loc" not in kws_clustering:
            kws_clustering.update({"legend_loc": "on_data"})
        if "vcenter" not in kws_clustering:
            kws_clustering.update({"vcenter": 0})
        if "col_cell_type" in kws_clustering:  # if provide cell column
            lab_cluster = kws_clustering.pop("col_cell_type")
        else: 
            lab_cluster = self._columns["col_cell_type"] 
        if not isinstance(genes, (list, np.ndarray)) and (
            genes is None or genes == "all"):
            names = self.adata.var.reset_index()[
                self._columns["col_gene_symbols"]].copy()  # gene names
            if genes == "all": 
                genes = names.copy()
        else:
            names = genes.copy()
        if genes_highlight and not isinstance(genes_highlight, list):
            genes_highlight = [genes_highlight] if isinstance(
                genes_highlight, str) else list(genes_highlight)
        if cell_types_circle and not isinstance(cell_types_circle, list):
            cell_types_circle = [cell_types_circle] if isinstance(
                cell_types_circle, str) else list(cell_types_circle)
        genes, names = list(pd.unique(genes)), list(pd.unique(names))
        if assay == "default":
            assay = self._assay
        if assay:  # store available `.obs` columns
            cols_obs = self.adata[assay].obs.columns
        else:
            cols_obs = self.adata.obs.columns
        if names_layers["scaled"] not in self.adata.layers:
            self.adata.layers[names_layers["scaled"]] = sc.pp.scale(
                self.adata, copy=True).X  # scaling (Z-scores)
        if layers == "all":  # to include all layers
            layers = list(self.adata.layers.copy())
        if None in layers:
            layers.remove(None)
        gene_symbols = None
        if self._columns["col_gene_symbols"] != self.adata.var.index.names[0]:
            gene_symbols = self._columns["col_gene_symbols"]
            
        # Pre-Processing/QC
        if "preprocessing" in self.figures[run_label]:
            print("\n<<< PLOTTING PRE-PROCESSING >>>")
            figs["preprocessing"] = self.figures[run_label]["preprocessing"]
            
        # Gene Expression Heatmap(s)
        if kws_gex is None:
            kws_gex = {"dendrogram": True, "show_gene_labels": True}
        if "cmap" not in kws_gex:
            kws_gex.update({"cmap": COLOR_MAP})
        print("\n<<< PLOTTING GEX (Heatmap) >>>")
        for j, i in enumerate([None] + list(layers)):
            lab = f"gene_expression_{i}" if i else "gene_expression"
            if i is None:
                hm_title = "Gene Expression"
            elif i == self._layer_perturbation:
                hm_title = f"Gene Expression (Perturbation Layer)"
            else:
                hm_title = f"Gene Expression ({i})"
            try:
                # sc.pl.heatmap(
                #     self.adata[assay] if assay else self.adata, names,
                #     lab_cluster, layer=i,
                #     ax=axes_gex[j], gene_symbols=gene_symbols, 
                #     show=False)  # GEX heatmap
                figs[lab] = sc.pl.heatmap(
                    self.adata[assay] if assay else self.adata, names,
                    lab_cluster, layer=i,
                    gene_symbols=gene_symbols, show=False)  # GEX heatmap
                # axes_gex[j].set_title("Raw" if i is None else i.capitalize())
                figs[lab] = plt.gcf(), figs[lab]
                figs[lab][0].suptitle(title if title else hm_title)
                # figs[lab][0].supxlabel("Gene")
                figs[lab][0].show()
            except Exception as err:
                warnings.warn(
                    f"{err}\n\nCould not plot GEX heatmap ('{hm_title}').")
                figs[lab] = err
        
        # Gene Expression Violin Plots
        if kws_gex_violin is None:
            kws_gex_violin = {}
        if "color_map" in kws_gex_violin:
            kws_gex_violin["cmap"] = kws_gex_violin.pop("color_map")
        if "cmap" not in kws_gex_violin:
            kws_gex_violin.update({"cmap": COLOR_MAP})
        if "groupby" in kws_gex_violin or "col_cell_type" in kws_gex_violin:
            lab_cluster = kws_gex_violin.pop(
                "groupby" if "groupby" in kws_gex_violin else "col_cell_type")
        else:
            lab_cluster = self._columns["col_cell_type"]
        if lab_cluster not in cols_obs:
            lab_cluster = None   # None if cluster label N/A in `.obs`
        for i in zip(["dendrogram", "swap_axes", "cmap"], 
                     [True, False, COLOR_MAP]):
            if i[0] not in kws_gex_violin:  # add default arguments
                kws_gex_violin.update({i[0]: i[1]})
        print("\n<<< PLOTTING GEX (Violin) >>>")
        for i in [None] + list(layers):
            try:
                lab = f"gene_expression_violin"
                title_gexv = title if title else "Gene Expression"
                if i:
                    lab += "_" + str(i)
                    if not title:
                        title_gexv = f"{title_gexv} ({i})"
                figs[lab] = sc.pl.stacked_violin(
                    self.adata[assay] if assay else self.adata, 
                    marker_genes_dict if marker_genes_dict else genes,
                    layer=i, groupby=lab_cluster, return_fig=True, 
                    gene_symbols=gene_symbols, title=title_gexv, show=False,
                    **kws_gex_violin)  # violin plot of GEX
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster)
                figs[lab].show()
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot GEX violins.")
                figs["gene_expression_violin"] = err
                
        # Gene Expression Dot Plot
        figs["gene_expression_dot"] = sc.pl.dotplot(
            self.adata[assay] if assay else self.adata, genes, lab_cluster, show=False)
        if genes_highlight is not None:
            for x in figs["gene_expression_dot"][
                "mainplot_ax"].get_xticklabels():
                # x.set_style("italic")
                if x.get_text() in genes_highlight:
                    x.set_color('#A97F03')
        
        # Gene Expression Matrix Plots
        if kws_gex_matrix is None:
            kws_gex_matrix = {}
        if "color_map" in kws_gex_matrix:
            kws_gex_matrix["cmap"] = kws_gex_matrix.pop("color_map")
        if "cmap" not in kws_gex_matrix:
            kws_gex_matrix.update({"cmap": COLOR_MAP})
        if "groupby" in kws_gex_matrix or "col_cell_type" in kws_gex_matrix:
            lab_cluster_mat = kws_gex_matrix.pop(
                "groupby" if "groupby" in kws_gex_matrix else "col_cell_type")
        else:
            lab_cluster_mat = self._columns["col_cell_type"]
        if lab_cluster_mat not in cols_obs:
            lab_cluster_mat = None   # None if cluster label N/A in `.obs`
        for i in zip(["dendrogram", "swap_axes", "cmap"], 
                     [True, False, COLOR_MAP]):
            if i[0] not in kws_gex_matrix:  # add default arguments
                kws_gex_matrix.update({i[0]: i[1]})
        print("\n<<< PLOTTING GEX (Matrix) >>>")
        print(kws_gex_matrix)
        try:
            for i in [None] + list(self.adata.layers):
                lab = f"gene_expression_matrix"
                title_gexm = title if title else "Gene Expression"
                if i:
                    lab += "_" + str(i)
                    if not title:
                        title_gexm = f"{title_gexm} ({i})"
                if i == names_layers["scaled"]:
                    bar_title = "Expression (Mean Z-Score)"
                else: 
                    bar_title = "Expression"
                figs[lab] = sc.pl.matrixplot(
                    self.adata[assay] if assay else self.adata, genes,
                    layer=i, return_fig=True, groupby=lab_cluster_mat,
                    title=title_gexm, gene_symbols=gene_symbols,
                    **{"colorbar_title": bar_title, **kws_gex_matrix
                       },  # colorbar title overriden if already in kws_gex
                    show=False)  # violin plot of GEX
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster_mat)
                figs[lab].show()
        except Exception as err:
            warnings.warn(f"{err}\n\nCould not plot GEX matrix plot.")
            figs["gene_expression_matrix"] = err
        
        # UMAP
        title_umap = str(title if title else kws_clustering[
            "title"] if "title" in kws_clustering else "UMAP")
        color = kws_clustering.pop(
            "color") if "color" in kws_clustering else None  # color-coding
        if "cmap" in kws_clustering:  # in case use wrong form of argument
            kws_clustering["color_map"] = kws_clustering.pop("cmap")
        if "palette" not in kws_clustering:
            kws_clustering.update({"palette": COLOR_PALETTE})
        if "color_map" not in kws_clustering:
            kws_clustering.update({"color_map": COLOR_MAP})
        if "X_umap" in self.adata.obsm or lab_cluster in (
            self.adata[assay].obs if assay else self.adata.obs).columns:
            print("\n<<< PLOTTING UMAP >>>")
            try:
                figs["clustering"] = sc.pl.umap(
                    self.adata[assay] if assay else self.adata, 
                    color=lab_cluster, return_fig=True, 
                    title=title_umap,  **kws_clustering)  # UMAP ~ cell type
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot UMAP clusters.")
                figs["clustering"] = err
            if genes is not None:
                print("\n<<< PLOTTING GEX ON UMAP >>>")
                try:
                    figs["clustering_gene_expression"] = sc.pl.umap(
                        self.adata[assay] if assay else self.adata, 
                        title=names, return_fig=True, 
                        gene_symbols=gene_symbols, color=names,
                        **kws_clustering)  # UMAP ~ GEX
                except Exception as err:
                    warnings.warn(f"{err}\n\nCould not plot GEX UMAP.")
                    figs["clustering_gene_expression"] = err
            if color is not None:
                print(f"\n<<< PLOTTING {color} on UMAP >>>")
                try:
                    figs[f"clustering_{color}"] = sc.pl.umap(
                        self.adata[assay] if assay else self.adata, 
                        title=title if title else None, return_fig=True, 
                        color=color, frameon=False,
                        **kws_clustering)  # UMAP ~ GEX
                except Exception as err:
                    warnings.warn(f"{err}\n\nCould not plot UMAP ~ {color}.")
                    figs[f"clustering_{color}"] = err
            if cell_types_circle and "X_umap" in (
                self.adata[assay] if assay else self.adata
                ).obsm:  # UMAP(s) with circled cell type(s)?
                fump, axu = plt.subplots(figsize=(3, 3))
                sc.pl.umap(self.adata[assay] if assay else self.adata, 
                           color=[lab_cluster], ax=axu, show=False)  # UMAP
                for h in cell_types_circle:
                    if assay:
                        location_cells = self.adata[assay][self.adata[
                            assay].obs[lab_cluster] == h, :].obsm['X_umap']
                    else:
                        location_cells = self.adata[
                            self.adata.obs[lab_cluster] == h, :].obsm['X_umap']
                    coordinates = [location_cells[:, i].mean() for i in [0, 1]]
                    circle = plt.Circle(tuple(coordinates), 1.5, color="r", 
                                        clip_on=False, fill=False)  # circle
                    axu.add_patch(circle)
                # l_1 = axu.get_legend()  # save original Legend
                # l_1.set_title(lab_cluster)
                # # Make a new Legend for the mark
                # l_2 = axu.legend(handles=[Line2D(
                #     [0],[0],marker="o", color="k", markerfacecolor="none", 
                #     markersize=12, markeredgecolor="r", lw=0, 
                #     label="selected")], frameon=False, 
                #                 bbox_to_anchor=(3,1), title='Annotation')
                #     # Add back the original Legend (was overwritten by new)
                # _ = plt.gca().add_artist(l_1)
                figs["umap_annotated"] = fump
        else:
            print("\n<<< UMAP NOT AVAILABLE TO PLOT. RUN `.cluster()`.>>>")
        return figs
        
    # def save_output(self, directory_path, run_keys="all", overwrite=False):
    #     """Save figures, results, adata object."""
    #     # TODO: FINISH
    #     raise NotImplementedError("Saving output isn't yet fully implemented.")
    #     if isinstance(run_keys, (str, float)):
    #         run_keys = [run_keys]
    #     elif run_keys == "all":
    #         run_keys = list(self.figures.keys())
    #         run_keys += list(self.results.keys())
    #         run_keys = list(pd.unique(self.results))
    #     for r in run_keys:
    #         os.makedirs(os.path.join(directory_path, r), exist_ok=True)
    #         if r in self.figures:
    #             for f in self.figures[r]:
    #                 dirp = os.path.join(directory_path, r, f)
    #                 os.makedirs(dirp, exist_ok=overwrite)
    #                 # self.figures[r][f].savefig(
    #                 #     os.path.join(dirp, f"{f}.pdf"))
    #         if r in self.results:
    #             for f in self.results[r]:
    #                 dirp = os.path.join(directory_path, r)
    #                 if not os.path.exists(dirp)
    #                 os.makedirs(dirp, exist_ok=overwrite)
            
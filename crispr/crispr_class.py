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
# import re
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
                 col_condition="perturbation",
                 col_perturbed="perturbation",
                 col_guide_rna="guide_ids",
                 col_num_umis="num_umis",
                 key_control="NT", 
                 key_treatment="KO", 
                 key_nonperturbed="NP", 
                 kws_process_guide_rna=None,
                 remove_multi_transfected=True,
                 **kwargs):
        """Initialize Crispr class object.

        Args:
            file_path (str, AnnData, or MuData): Path or object 
                containing data. Used in initialization to create 
                the initial `.adata` attribute (an AnnData or 
                MuData object). Either
                    - a path to a 10x directory (with matrix.mtx.gz, 
                    barcodes.tsv.gz, features.tsv.gz),
                    - a path to an .h5ad or .mu file 
                        (Scanpy/AnnData/Muon-compatible), 
                    - an AnnData or MuData object (e.g., already 
                        loaded with Scanpy or Muon), or
                    - a dictionary containing keyword arguments to 
                        pass to `crispr.pp.combine_matrix_protospacer` 
                        (in order to load information about 
                        perturbations from other file(s); see 
                        function documentation), or
                    - to concatenate multiple datasets, a dictionary 
                        (keyed by your desired subject/sample names 
                        to be used in `col_sample_id`) consisting 
                        of whatever objects you would pass to 
                        `create_object()`'s `file` argument for the 
                        individual objects. You must also specify 
                        `col_sample` (a tuple as described in the 
                        documentation below). The other arguments 
                        passed to the `crispr.pp.create_object()` 
                        function (e.g., `col_gene_symbols`) can be 
                        specified as normal if they are common across 
                        samples; otherwise, specify them as lists in 
                        the same order as the `file` dictionary. 
            assay (str, optional): Name of the gene expression assay if 
                loading a multi-modal data object (e.g., "rna"). 
                Defaults to None.
            assay_protein (str, optional):  Name of the assay containing 
                the protein expression modality, if available. 
                For instance, if "adt", `self.adata["adt"]` would be 
                expected to contain the AnnData object for the 
                protein modality. ONLY FOR MULTI-MODAL DATA for certain 
                bonus visualization methods. Defaults to None. 
            col_gene_symbols (str, optional): Column name in `.var` for 
            questionable gene symbols. Defaults to "gene_symbols".
            col_cell_type (str, optional): Column name in `.obs` for 
                cell type. Defaults to "leiden" (anticipating that you 
                will run `self.cluster(...)` with 
                `method_cluster="leiden"`). This column may be
                - pre-existing in data (e.g., pre-run clustering column 
                    or manual annotations), or 
                - expected to be created via `Crispr.cluster()`. 
            col_sample_id (str or tuple, optional): Column in `.obs` 
                with sample IDs. Defaults to "standard_sample_id". 
                If this column does not yet exist in your data and 
                needs to be created by concatenating datasets, 
                you must provide `file_path` as a dictionary keyed
                by desired `col_sample_id` values as well as signal
                that this needs to happen by specifying col_sample_id
                as a tuple, with the second element containing a
                dictionary of keyword arguments to pass to the
                concatenation preprocessing function (e.g., batch
                effect removal).
            col_batch (_type_, optional):  Column in `.obs` with batch 
                IDs. Defaults to None.
            col_condition (str, optional): Either the name of an 
                existing column in `.obs` indicating the experimental 
                condition to which each cell belongs or 
                (for CRISPR designs) the **desired** name of the column 
                that will be created from `col_guide_rna` to indicate 
                the gene(s) targeted in each cell.
                - If there are multiple conditions besides control 
                    (e.g., multiple types of drugs and/or exposure 
                    times, multiple target genes in CRISPR), 
                    this column distinguishes the different conditions, 
                    in contrast to the binary `col_perturbed`.
                - In CRISPR designs (i.e., `col_guide_rna` specified), 
                    this column will be where each guide RNA's target 
                    gene will be stored, whether pre-existing (copied 
                    directly from `col_guide_rna` if 
                    `kws_process_guide_rna` is None) or created during 
                    the Crispr object initialization by passing 
                    `col_guide_rna` and `kws_process_guide_rna` to 
                    `crispr.pp.filter_by_guide_counts()` in order to 
                    convert particular guide RNA IDs to their target(s) 
                    (e.g., STAT1-1|IL6-2-1|NegCtrl32a|IL6-1 => 
                    STAT1|IL6|NT|IL6). 
                    Defaults to None.
                - For non-CRISPR designs (e.g., drug exposure):
                    - This column should exist in the AnnData or MuData 
                        object (either already available upon simply 
                        reading the specified file with no other 
                        needing alterations, or as originally passed to 
                        the initialization method if given instead of a 
                        file path).
                    - It should contain a single `key_control`, but it 
                        can have multiple categories of other entries 
                        that all translate to `key_treatment` in 
                        `col_perturbed`.
                    - If you have multiple control conditions, you 
                        should pass an already-created AnnData object to 
                        the `file_path` argument of the `Crispr` class 
                        initialization method after adding a separate 
                        column with a name different from those 
                        specified in any of the other column arguments. 
                        You can then pass that column name manually to 
                        certain functions' `col_control` arguments and 
                        specify the particular control condition in 
                        `key_control`.
                - In most methods, `key_control` and `key_treatment`, 
                    as well as `col_perturbed` or `col_condition` 
                    (for methods that don't require binary labeling), 
                    can be newly-specified so that you can compare 
                    different conditions within this column. If the 
                    argument is named `col_perturbed`, passing a column 
                    with more than two categories usually results in 
                    subsetting the data to compare only the two 
                    conditions specified in `key_treatment` and 
                    `key_control`. The exception is where there is a 
                    `key_treatment_list` or similarly-named argument.
            col_perturbed (str, optional): Column in `.obs` where class 
                methods will be able to find the binary experimental 
                condition variable. It will be created during `Crispr` 
                object initialization as a binary version of 
                `col_condition`. Defaults to "perturbation". For CRISPR 
                designs, all entries containing the patterns specified 
                in `kws_process_guide_rna["key_control_patterns"]` will 
                be changed to `key_control`, and all cells with 
                targeting guides will be changed to `key_treatment`.
            col_guide_rna (str, optional): Column in `.obs` with guide 
                RNA IDs. Defaults to "guide_ids". This column should 
                always be specified for CRISPR designs and should NOT 
                be specified for other types of experiments. 
                - If only one kind of guide RNA is used, then this 
                    should be a column containing the name of the 
                    gene targeted (for perturbed cells) and the names 
                    of any controls, and `key_treatment` should be the 
                    name of the gene targeted. Then, `col_condition` 
                    will be a direct copy of this column.
                - Entries in this column should be either gene names in 
                    `self.adata.var_names` (or `key_control` or one of
                    the patterns in 
                    `kws_process_guide_rna["key_control_patterns"]`), 
                    plus, optionally, suffixes separating guide #s 
                    (e.g., STAT1-1-2, CTRL-1) and/or with a character 
                    that splits separate guide RNAs within that cell 
                    (if multiply-transfected cells are present). 
                    These characters should be specified in 
                    `kws_process_guide_rna["guide_split"]` and 
                    `kws_process_guide_rna["feature_split"]`, 
                    respectively. 
                    For instance, they would be "-" and "|", if 
                    `col_guide_rna` entries for a cell multiply 
                    transfected by two sgRNAs targeting STAT1, 
                    two control guide RNAs, and a guide targeting CCL5 
                    would look like 
                    "STAT1-1-1|STAT1-1-2|CNTRL-1-1|CCL5-1".
                - Currently, everything after the first dash 
                    (or whatever split character is specified) is 
                    discarded when creating `col_target_genes`, 
                    so keep that in mind.
                - This column will be stored in `.obs` as 
                    `<col_guide_rna>_original` if 
                    `kws_process_guide_rna` is not None, 
                    as that will result in a processed version of this 
                    column being stored under `.obs[<col_guide_rna>]`.
            col_num_umis (str, optional): Name of column in `.obs` with 
                the UMI counts. This should be specified if 
                `kws_process_guide_rna` is not None. For designs with 
                multiply-transfected cells, it should follow the same 
                convention established in 
                `kws_process_guide_rna["feature_split"]`. 
                Defaults to "num_umis".
            key_control (str, optional): The label that is or will be in 
                `col_condition`, `col_guide_rna`, and `col_perturbed` 
                indicating control rows. Defaults to "NT". Either 
                    - exists as entries in pre-existing column(s), or
                    - is the name you want the control entries (detected 
                    using `.obs[<col_guide_rna>]` and 
                    `kws_process_guide_rna["key_control_patterns"]`) 
                    to be categorized as control rows under the new 
                    version(s) of `.obs[<col_guide_rna>]`, 
                    `.obs[<col_target_genes>]`, and/or 
                    `.obs[<col_perturbed>]`. For instance, entries like 
                    "CNTRL-1", "NEGCNTRL", "Control", etc. in 
                    `col_guide_rna` would all be keyed as "Control" in 
                    (the new versions of) `col_target_genes`, 
                    `col_guide_rna`, and `col_perturbed` if you specify 
                    `key_control` as "Control" and 
                    `kws_process_guide_rna` as
                    `dict(key_control_patterns=["CTRL", "Control"])`.
            key_treatment (str, optional): What entries in 
                `col_perturbed` indicate a treatment condition 
                (e.g., drug administration, CRISPR knock-out/down) 
                as opposed to a control condition? This name will also 
                be used for Mixscape classification labeling. 
                Defaults to "KO".
            key_nonperturbed (str, optional): What will be stored in the 
                `mixscape_class_global` and related columns/labels after 
                running Mixscape methods. Indicates cells without a 
                detectible perturbation. Defaults to "NP".
            kws_process_guide_rna (dict, optional): Dictionary of 
                keyword arguments to pass to 
                `crispr.pp.filter_by_guide_counts()`. 
                (See below and crispr.processing.preprocessing 
                documentation). Defaults to None (no processing will 
                take place, in which case BE SURE THAT 
                `col_target_genes` already exists in the data once 
                loaded and contains the already-filtered, summed up, 
                generic gene-named, etc. versions of the guide 
                RNA column). Keys of this dictionary should be:
                    - key_control_patterns (list, optional): List 
                        (or single string) of patterns in guide RNA 
                        column entries that correspond to a control. 
                        For instance, if control entries in the original 
                        `col_guide_rna` column include `NEGCNTRL` and 
                        `Control.D`, you should specify 
                        ['Control', 'CNTRL'] (assuming no non-control 
                        sgRNA names contain those patterns). If blank 
                        entries should be interpreted as control guides, 
                        then include np.nan/numpy.nan in this list. 
                        Defaults to None, which turns to [np.nan].
                    - `max_percent_umis_control_drop` (int, optional): 
                        If control UMI counts are $<=$ this percentage 
                        of the total counts for that cell, and if a 
                        non-control gRNA is also present and meets 
                        other filtering criteria, then consider that 
                        cell pseudo-single-transfected 
                        (non-control gene). Defaults to 75.
                    - `min_percent_umis` (int, optional): sgRNAs with 
                        counts below this percentage will be considered 
                        noise for that guide. Defaults to 40.
                    - `feature_split` (str, optional): For designs with 
                        multiple guides, the character that splits 
                        guide names in `col_guide_rna`. For instance, 
                        "|" for `STAT1-1|CNTRL-1|CDKN1A`. 
                        Defaults to "|". If only single guides, 
                        you should set to None.
                    - `guide_split` (str, optional): The character that 
                        separates guide (rather than 
                        gene target)-specific IDs within gene. 
                        For instance, guides targeting STAT1 may 
                        include STAT1-1, STAT1-2-1, etc.; 
                        the argument would be "-" so the function can 
                        identify all of those as targeting STAT1. 
                        Defaults to "-".
            remove_multi_transfected (bool, optional): In designs with 
                multiple guides per cell, remove multiply-transfected 
                cells (i.e., cells where more than one target 
                guide survived application of any filtering criteria set 
                in `kws_process_guide_rna`). If 
                `kws_process_guide_rna["max_percent_umis_control_drop"]` 
                is greater than 0, then cells with one target guide and 
                control guides which together make up less than 
                `max_percent_umis_control_drop`% of total UMI counts 
                will be considered pseudo-single-transfected for the 
                target guide. Defaults to True. Some functionality may 
                be limited and/or problems occur if set to False and if 
                multiply-transfected cells remain in data. 
        """
        print("\n\n<<<INITIALIZING CRISPR CLASS OBJECT>>>\n")
        self._assay = assay
        self._assay_protein = assay_protein
        self._file_path = file_path
        self._layers = {"layer_original": "original", 
                        "layer_perturbation": "X_pert"}
        if kwargs:
            print(f"\nUnused keyword arguments: {kwargs}.\n")
        
        # Create Attributes to Store Results/Figures
        self.figures = {"main": {}}
        self.results = {"main": {}}
        self.info = {"descriptives": {}, 
                      "guide_rna": {}}  # extra info to store by methods
        
        # Create Object & Store Raw Counts
        if isinstance(col_sample_id, (list, tuple)):  # multi-sample
            kws_process_batches, col_sample_id = col_sample_id
            col_sample_arg = col_sample_id
        else:  # only 1 sample
            kws_process_batches = None
            col_sample_arg = None  # even if col_sample_id is specified...
            # ...we don't want to pass to create_object because it will
            # assume we need that column created through concatenation
            # when it's actually already in the dataset
        self.adata = cr.pp.create_object(
            self._file_path, assay=assay, col_gene_symbols=col_gene_symbols,
            col_sample_id=col_sample_arg, 
            kws_process_guide_rna={
                "col_guide_rna": col_guide_rna, "col_num_umis": col_num_umis, 
                "key_control": key_control, 
                "col_guide_rna_new": col_condition, 
                "remove_multi_transfected": remove_multi_transfected, 
                **kws_process_guide_rna}
            if kws_process_guide_rna else None)  # make AnnData
        self.info["guide_rna"]["keywords"] = kws_process_guide_rna
        self.rna.layers['counts'] = self.rna.X.copy()
        
        # Check Arguments & Data
        if any((x in self.rna.obs for x in [
            col_guide_rna, col_perturbed, col_condition])):
            if col_perturbed in self.rna.obs and (
                col_condition in self.rna.obs):
                raise ValueError("col_perturbed and col_condition cannot "
                                 "both be in `.obs`. col_perturbed must be "
                                 "created from col_condition.")
            if col_perturbed in self.rna.obs:
                warnings.warn(f"col_perturbed {col_perturbed} already "
                              " in `.obs`. Assuming perturbation is binary "
                              "(i.e., only has two conditions, including "
                              "control), so col_condition will be "
                              "equivalent.")
                self.rna.obs.loc[:, col_condition] = self.rna.obs[
                    col_perturbed]
        else:
            raise ValueError("col_condition or "
                             "col_guide_rna must be in `.obs`.")
        print(self.adata.obs, "\n\n") if assay else None
        print("\n\n", self.rna.obs)

        # Binary Perturbation Column
        if (col_perturbed not in 
            self.rna.obs):  # if col_perturbed doesn't exist yet...
            self.rna.obs = self.rna.obs.join(
                self.rna.obs[col_condition].apply(
                    lambda x: x if pd.isnull(x) else key_control if (
                        x == key_control) else key_treatment
                    ).to_frame(col_perturbed), lsuffix="_original"
                )  # create binary form of col_condition
        
        # Store Columns & Keys within Columns as Dictionary Attributes
        self._columns = dict(col_gene_symbols=col_gene_symbols,
                             col_condition=col_condition,
                             col_target_genes=col_condition,
                             col_perturbed=col_perturbed,
                             col_cell_type=col_cell_type,
                             col_sample_id=col_sample_id, 
                             col_batch=col_batch,
                             col_guide_rna=col_guide_rna,
                             col_num_umis=col_num_umis)
        self._keys = dict(key_control=key_control, 
                          key_treatment=key_treatment, 
                          key_nonperturbed=key_nonperturbed)
        print("\n\n")
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)
        print("\n\n", self.rna)
        
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

    @property
    def rna(self):
        """Get RNA modality of AnnData."""
        return self.adata[self._assay] if self._assay else self.adata

    @rna.setter
    def rna(self, value) -> None:
        if self._assay: 
            self.adata[self._assay] = value
        else:
            self.adata = value
            
    @property
    def obs(self):
        """Get `.obs` attribute of AnnData."""
        return self.adata.obs

    @obs.setter
    def obs(self, value) -> None:
        self.adata.obs = value
            
    @property
    def uns(self):
        """Get `.uns` attribute of adata's gene expression modality."""
        return self.adata[self._assay].uns if self._assay else self.adata.uns

    @uns.setter
    def uns(self, value) -> None:
        if self._assay: 
            self.adata[self._assay].uns = value
        else:
            self.adata.uns = value
            
    @property
    def var(self):
        """Get `.var` attribute of .adata's gene expression modality."""
        return self.adata[self._assay].var if self._assay else self.adata.var

    @var.setter
    def var(self, value) -> None:
        if self._assay: 
            self.adata[self._assay].var = value
        else:
            self.adata.var = value
            
    @property
    def obsm(self):
        """Get `.obsm` attribute of .adata's gene expression modality."""
        return self.adata[
            self._assay].obsm if self._assay else self.adata.obsm

    @obsm.setter
    def obsm(self, value) -> None:
        if self._assay: 
            self.adata[self._assay].obsm = value
        else:
            self.adata.obsm = value
    
    def describe(self, group_by=None, plot=False):
        """Describe data."""
        desc, figs = {}, {}
        gbp = [self._columns["col_cell_type"]]
        if group_by:
            gbp += [group_by]
        if "descriptives" not in self.info:
            self.info["descriptives"] = {}
        print("\n\n\n", self.adata.obs.describe().round(2),
              "\n\n\n")
        print(f"\n\n{'=' * 80}\nDESCRIPTIVES\n{'=' * 80}\n\n")
        print(self.adata.obs.describe())
        for g in gbp:
            print(self.adata.obs.groupby(g).describe().round(2), 
                  f"\n\n{'-' * 40}\n\n") 
        try:
            if "guide_rna_counts" not in self.info["descriptives"]:
                self.info["descriptives"][
                    "guide_rna_counts"] = self.count_gRNAs()
            desc["n_grnas"] = self.info["descriptives"][
                "guide_rna_counts"].groupby("gene").describe().rename(
                    {"mean": "mean sgRNA count/cell"})
            print(desc["n_grnas"])
            if plot is True:
                n_grna = self.info["descriptives"][
                    "guide_rna_counts"].to_frame("sgRNA Count")
                if group_by is not None:
                    n_grna = n_grna.join(self.adata.obs[
                        group_by].rename_axis("bc"))  # join group_by variable
                figs["n_grna"] = sns.catplot(data=n_grna.reset_index(),
                    y="gene", x="sgRNA Count", kind="violin", hue=group_by,
                    height=len(self.info["descriptives"][
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
        self.info["descriptives"].update(desc)
        return figs
    
    def get_guide_rna_counts(self, target_gene_idents=None, 
                             group_by=None, **kwargs):
        """Plot guide RNA counts by cell type & 0-2 other variables."""
        if isinstance(group_by, str):
            group_by = [group_by]
        if isinstance(target_gene_idents, str):
            target_gene_idents = [target_gene_idents]
        cols = [self._columns["col_target_genes"]]
        if group_by:  # join group_by variables from adata
            cols += group_by
        cols = list(pd.unique(cols))
        dff = self.info["guide_rna"]["counts_unfiltered"].reset_index(
            "Gene").rename({"Gene": "Guide"}, axis=1).join(self.rna.obs[cols])
        if target_gene_idents:
            dff = dff[dff[self._columns["col_target_genes"]].isin(
                target_gene_idents)]
        kws_plot = dict(
            share_x=True, share_y=False, figsize=(30, 30), split=True, 
            col=group_by[0] if group_by else None,
            hue=group_by[1] if group_by and len(group_by) > 2 else None,
            row=group_by[2] if group_by and len(group_by) > 2 else None,
            col_wrap=cr.pl.square_grid(
                len(dff[group_by[0]].unique()))[1] if group_by and len(
                    group_by) < 3 else None)  # default plot options
        kws_plot.update(kwargs)  # overwrite with any user specifications
        fig = sns.catplot(data=dff.dropna(
            subset=self._columns_created["guide_percent"]), y="Guide", 
                          x=self._columns_created["guide_percent"],
                          kind="violin", **kws_plot)
        fig.fig.suptitle("Guide RNA Counts" + str(
            f" by {', '.join(group_by)}" if group_by else ""))
        fig.fig.tight_layout()
        return self.info["guide_rna"]["counts_unfiltered"], fig
    
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
            remove_doublets=remove_doublets, 
            **self._columns, **self._layers,
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
        if self._columns["col_sample_id"] or self._columns["col_batch"]:
            if colors is None:
                colors = []
                for x in [self._columns["col_sample_id"], 
                          self._columns["col_batch"]]:
                    if x:
                        colors += [x]  # add sample & batch UMAP
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
        preds, ct_dot = cr.ax.perform_celltypist(
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
        return preds, ct_dot
    
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
        
    def run_mixscape(self, assay=None, assay_protein=None,
                     col_cell_type=None,
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
            col_split_by (str, optional): `adata.obs` column name of 
                sample categories to calculate separately 
                (e.g., replicates). Defaults to None.
            assay (str, optional): Assay slot of adata 
                ('rna' for `adata['rna']`). If not provided, will use
                self._assay. Typically, users should not specify this 
                argument.
            assay_protein (str, optional): Protein assay slot name 
                (if available). If True, use `self._assay_protein`. If 
                None, don't run extra protein expression analyses, even 
                if protein expression modality is available.
            protein_of_interest (str, optional): If assay_protein is not 
                None and plot is True, will allow creation of violin 
                plot of protein expression (y) by 
                <target_gene_idents> perturbation category (x),
                split/color-coded by Mixscape classification 
                (`adata.obs['mixscape_class_global']`). 
                Defaults to None.
            target_gene_idents (list or bool, optional): List of names 
                of genes whose perturbations will determine cell 
                grouping for the above-described violin plot and/or
                whose differential expression posterior probabilities 
                will be plotted in a heatmap. Defaults to None.
                True to plot all in `self.adata.uns["mixscape"]`.
            min_de_genes (int, optional): Minimum number of genes a cell 
                must express differentially to be labeled 'perturbed'. 
                For Mixscape and LDA (if applicable). Defaults to 5.
            pval_cutoff (float, optional): Threshold for significance 
                to identify differentially-expressed genes. 
                For Mixscape and LDA (if applicable). Defaults to 5e-2.
            logfc_threshold (float, optional): Will only test genes  
                whose average logfold change across the two cell groups  
                is at least this number. For Mixscape and LDA 
                (if applicable). Defaults to 0.25.
            n_comps_lda (int, optional): Number of principal components
                for PCA. Defaults to None (LDA not performed).
            iter_num (float, optional): Iterations to run 
                in order to converge (if needed).
            plot (bool, optional): Make plots? Defaults to True.
            **kwargs: Additional keyword arguments to be passed to 
                the `crispr.ax.perform_mixscape()` function.
                The arguments that are used vary by method, but for ,
                    example you can pass `key_<control, treatment>` 
                    and/or
                    `col_<perturbation, cell_type, etc.>` to override 
                    those defined by 
                    `self._columns` and `self._keys`.
                    If not specified (which should usually be the case), 
                    the corresponding `._key/column` attribute will be 
                    used. You can pass these extra arguments in rare 
                    case where you want to use different columns/keys 
                    within columns across 
                    different methods or runs of a method. For instance,
                    for experimental conditions that have
                    multiple levels (e.g., control, drug A treatment, 
                    drug B treatment), allowing this argument 
                    (and `col_perturbed`, for instance, 
                    if self._columns["col_perturbed"] 
                    is a binary treatment vs. drug column, and you want 
                    to use the more specific column for Augur)
                    to be specified allows for more flexibility 
                    in analysis.
            
        Returns:
            dict: A dictionary containing figures visualizing
                results, for instance, 
        
        Notes:
            - Classifications are stored in 
            `self.adata.obs['mixscape_class_global']`
            (detectible perturbation (`._keys["key_treatment"]`) vs. 
            un-detectible perturbation (`._keys["key_nonperturbed"]`) 
            vs. control (`self._keys["key_control"]`)) and in
            `self.adata.obs['mixscape_class']` (like the previous, 
                but specific to target genes, e.g., 
                "STAT1 KO" vs. just "KO"). 
            - Posterial p-values are stored in 
                `self.adata.obs['mixscape_class_p_<key_treatment>']`.
            - Other result output will be stored in `self.figures` and 
                `self.results` under the `run_label` key under 
                `mixscape` (e.g., `self.figures["main"]["mixscape"]`).
            
        """
        if assay is None:
            assay = self._assay
        if assay_protein is None:
            assay_protein = self._assay
        for x in [self._columns, self._keys, self._layers]:
            for c in x:  # iterate column/key name attributes
                if c not in kwargs:  # if not passed as argument to method...
                    kwargs.update({c: x[c]})  # & use object attribute
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if "col_cell_type" in kwargs:
            _ = kwargs.pop("col_cell_type")
        figs_mix = cr.ax.perform_mixscape(
            self.adata.copy() if test is True else self.adata, 
            assay=assay, assay_protein=assay_protein,
            col_cell_type=col_cell_type,
            target_gene_idents=target_gene_idents,
            min_de_genes=min_de_genes, col_split_by=col_split_by, 
            plot=plot, **kwargs)
        if test is False:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"mixscape": figs_mix})
        return figs_mix
    
    def run_augur(self, assay=None, classifier="random_forest_classifier", 
                  augur_mode="default", kws_augur_predict=None, n_folds=3,
                  subsample_size=20, n_threads=True, 
                  select_variance_features=False, seed=1618, 
                  plot=True, run_label="main", test=False, **kwargs):        
        """
        Runs the Augur perturbation scoring and prediction analysis.

        Parameters:
            assay (str): The name of the gene expression assay 
                (for multi-modal data objects). 
            augur_mode (str, optional): Augur or permute? 
                Defaults to "default". (See pertpy documentation.)
            classifier (str): The classifier to be used for the analysis. 
                Defaults to "random_forest_classifier".
            kws_augur_predict (dict): Additional keyword arguments to be 
                passed to the `perform_augur()` function.
            n_folds (int): The number of folds to be used for 
                cross-validation. Defaults to 3. 
                Should be >= 3 as a matter of best practices.
            subsample_size (int, optional): Per Pertpy code: 
                "number of cells to subsample randomly per type 
                from each experimental condition." Defaults to 20.
            n_threads (bool): The number of threads to be used for 
                parallel processing. If set to True, the available 
                CPUs minus 1 will be used. Defaults to True.
            select_variance_features (bool, optional): Use Augur 
                to select genes (True), or Scanpy's  
                highly_variable_genes (False). Defaults to False.
            seed (int): The random seed to be used for reproducibility. 
                Defaults to 1618.
            plot (bool): Whether to plot the results. Defaults to True.
            run_label (str): The label for the current run. 
                Defaults to "main".
            test (bool): Whether the function is being called as a 
                test run. If True, self.adata will be copied so 
                that no alterations are made inplace via Augur,
                and results are not stored in object attributes.
                Defaults to False.
            **kwargs: Additional keyword arguments to be passed to 
                the `crispr.ax.perform_mixscape()` function.
                The arguments that are used vary by method, but for ,
                    example you can pass `key_<control, treatment>` 
                    and/or
                    `col_<perturbation, cell_type, etc.>` to override 
                    those defined by 
                    `self._columns` and `self._keys`.
                    If not specified (which should usually be the case), 
                    the corresponding `._key/column` attribute will be 
                    used. You can pass these extra arguments in rare 
                    case where you want to use different columns/keys 
                    within columns across 
                    different methods or runs of a method. For instance,
                    for experimental conditions that have
                    multiple levels (e.g., control, drug A treatment, 
                    drug B treatment), allowing this argument 
                    (and `col_perturbed`, for instance, 
                    if self._columns["col_perturbed"] 
                    is a binary treatment vs. drug column, and you want 
                    to use the more specific column for Augur)
                    to be specified allows for more flexibility 
                    in analysis.

        Returns:
            tuple: A tuple containing the AnnData object, results, and 
                figures generated by the Augur analysis.
        """
        # Column & Key Names + Other Arguments to Pass to Augur
        for x in [self._columns, self._keys]:
            for c in x:  # iterate column/key name attributes
                if c not in kwargs:  # if not passed as argument to method...
                    kwargs.update({c: x[c]})  # & use object attribute
        if n_threads is True:
            n_threads = os.cpu_count() - 1 # use available CPUs - 1
        if assay is None:
            assay = self._assay
        if kws_augur_predict is None:
            kws_augur_predict = {}
        data, results, figs_aug = cr.ax.perform_augur(
            self.adata, assay=assay, classifier=classifier,
            augur_mode=augur_mode, subsample_size=subsample_size,
            select_variance_features=select_variance_features, 
            n_folds=n_folds, kws_augur_predict=kws_augur_predict,
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
                         run_label="main", plot=True, **kwargs):
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
                the real range by setting minimum and maximum color 
                scaling based on properties of the data. 
                Defaults to False.
            run_label (str, optional): The label for the current run. 
                Affects the key under which results and figures are 
                stored in internal attributes. Defaults to "main".
            plot (bool, optional): Whether to create plots. 
                Defaults to True.
            **kwargs: Additional keyword arguments to be passed to 
                the `crispr.ax.perform_mixscape()` function.
                The arguments that are used vary by method, but for ,
                    example you can pass `key_<control, treatment>` 
                    and/or
                    `col_<perturbation, cell_type, etc.>` to override 
                    those defined by 
                    `self._columns` and `self._keys`.
                    If not specified (which should usually be the case), 
                    the corresponding `._key/column` attribute will be 
                    used. You can pass these extra arguments in rare 
                    case where you want to use different columns/keys 
                    within columns across 
                    different methods or runs of a method. For instance,
                    for experimental conditions that have
                    multiple levels (e.g., control, drug A treatment, 
                    drug B treatment), allowing this argument 
                    (and `col_perturbed`, for instance, 
                    if self._columns["col_perturbed"] 
                    is a binary treatment vs. drug column, and you want 
                    to use the more specific column for Augur)
                    to be specified allows for more flexibility 
                    in analysis.
        Returns:
            output: A tuple containing output from 
                `cr.ax.compute_distance()`, 
                including distance matrices, figures, etc. 
                See function documentation.

        """
        for x in [self._columns, self._keys]:
            for c in x:  # iterate column/key name attributes
                if c not in kwargs:  # if not passed as argument to method...
                    kwargs.update({c: x[c]})  # & use object attribute
        output = cr.ax.compute_distance(
            self.adata, distance_type=distance_type, method=method,
            kws_plot=kws_plot, highlight_real_range=highlight_real_range, 
            plot=plot, **kwargs)
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
        for x in [self._columns, self._keys]:
            for c in x:  # iterate column/key name attributes
                if c not in kwargs:  # if not passed as argument to method...
                    kwargs.update({c: x[c]})  # & use object attribute
        output = cr.ax.perform_gsea(
            self.adata, filter_by_highly_variable=filter_by_highly_variable, 
            **kwargs)  # GSEA
        if run_label not in self.results:
            self.results[run_label] = {}
        self.results[run_label]["gsea"] = output
        return output
          
    def run_composition_analysis(self, reference_cell_type, 
                                 assay=None, analysis_type="cell_level", 
                                 est_fdr=0.05, generate_sample_level=False,
                                 plot=True, run_label="main", **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        for x in [self._columns, self._keys]:
            for c in x:  # iterate column/key name attributes
                if c not in kwargs:  # if not passed as argument to method...
                    kwargs.update({c: x[c]})  # & use object attribute
        output = cr.ax.analyze_composition(
            self.adata, reference_cell_type,
            assay=assay if assay else self._assay, analysis_type=analysis_type,
            generate_sample_level=generate_sample_level, 
            est_fdr=est_fdr, plot=plot, **kwargs)
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
            sample_id=self._columns["col_perturbed"], n_mpcs=n_programs,
            celltype_key=col_cell_type, 
            n_counts_key=self._columns["col_num_umis"])
        pdata, mcps, ws, ct_subs = d_l.calculate_multifactor_PMD(
            self.adata.copy(), normalize=True)
        mcp_cols = list(set(pdata.obs.columns).difference(
            self.adata.obs.columns))
        cols = cr.pl.square_grid(len(mcp_cols) + 2)[1]
        fig = sc.pl.umap(
            pdata, color=mcp_cols + [
                self._columns["col_perturbed"], col_cell_type],
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
             run_label="main", **kwargs):
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
            elif i == self._layers["layer_perturbation"]:
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
        if "col_cell_type" in kwargs:
            lab_cluster = kwargs.pop("col_cell_type")
        elif "groupby" in kws_gex_violin or "col_cell_type" in kws_gex_violin:
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
            self.adata[assay] if assay else self.adata, genes, 
            lab_cluster, show=False)
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
            
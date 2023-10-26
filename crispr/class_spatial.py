#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""

import scanpy as sc
import warnings
import seaborn as sns
import pertpy as pt
import crispr as cr
from crispr.class_sc import Omics
import pandas as pd

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Crispr(Omics):
    """A class for CRISPR analysis and visualization."""
    
    _columns_created = dict(guide_percent="Percent of Cell Guides")

    def __init__(
        self, file_path, assay=None, assay_protein=None, 
        col_gene_symbols="gene_symbols", 
        col_cell_type="leiden", col_sample_id="standard_sample_id", 
        col_condition="perturbation",
        col_perturbed="perturbation", col_guide_rna="guide_ids",
        col_num_umis="num_umis", key_control="NT", key_treatment="KO", 
        key_nonperturbed="NP", kws_process_guide_rna=None, 
        kws_multi=None, **kwargs):
        """
        Initialize Crispr class object.

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
                gene symbols. Defaults to "gene_symbols".
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
                that concatenation needs to happen by specifying 
                `col_sample_id` as a tuple, with the second element 
                containing a dictionary of keyword arguments to pass to
                `AnnData.concatenate()` or None (to use defaults).
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
                    - `remove_multi_transfected` (bool, optional): In 
                        designs with multiple guides per cell, remove 
                        multiply-transfected cells (i.e., cells where 
                        more than one target guide survived application 
                        of any filtering criteria set in 
                        `kws_process_guide_rna`). If 
                        `kws_process_guide_rna[
                        "max_percent_umis_control_drop"]` 
                        is greater than 0, then cells with one target 
                        guide and control guides which together make up 
                        less than `max_percent_umis_control_drop`% of 
                        total UMI counts will be considered 
                        pseudo-single-transfected for the 
                        target guide. Defaults to True. 
                        Some functionality may be limited and/or 
                        problems occur if set to False and if 
                        multiply-transfected cells remain in data. 
        """
        print("\n\n<<< INITIALIZING CRISPR CLASS OBJECT >>>\n")
        self._assay = assay
        self._assay_protein = assay_protein
        self._file_path = file_path
        self._layers = {**cr.pp.get_layer_dict(), 
                        "layer_perturbation": "X_pert"}
        if kwargs:
            print(f"\nUnused keyword arguments: {kwargs}.\n")
        
        # Create Attributes to Store Results/Figures
        self.figures = {"main": {}}
        self.results = {"main": {}}
        self.info = {"descriptives": {}, 
                     "guide_rna": {},
                     "methods": {}}  # extra info to store post-use of methods
        
        # Create Object & Store Raw Counts
        super().__init__(
            self._file_path, assay=assay, col_gene_symbols=col_gene_symbols,
            col_sample_id=col_sample_id, col_condition=col_condition,
            key_control=key_control, key_treatment=key_treatment,
            kws_process_guide_rna={
                "col_guide_rna": col_guide_rna, "col_num_umis": col_num_umis, 
                "key_control": key_control, 
                "col_guide_rna_new": col_condition, 
                **kws_process_guide_rna} if kws_process_guide_rna else None, 
            kws_multi=kws_multi)  # make AnnData
        self.info["guide_rna"]["keywords"] = kws_process_guide_rna
        if kws_process_guide_rna and "guide_split" in kws_process_guide_rna:
            self.info["guide_rna"]["guide_split"] = kws_process_guide_rna[
                "guide_split"]
        elif "guide_split" in self.rna.obs:
            self.info["guide_rna"]["guide_split"] = str(
                self.rna.obs["guide_split"].iloc[0])
        else:
            self.info["guide_rna"]["guide_split"] = None
        if kws_process_guide_rna and "feature_split" in kws_process_guide_rna:
            self.info["guide_rna"]["feature_split"] = kws_process_guide_rna[
                "feature_split"]
        elif "feature_split" in self.rna.obs:
            self.info["guide_rna"]["feature_split"] = str(
                self.rna.obs["feature_split"].iloc[0])
        else:
            self.info["guide_rna"]["feature_split"] = None
        print(self.adata, "\n\n") if assay else None
        
        # Check Arguments & Data
        if any((x in self.rna.obs for x in [
            col_guide_rna, col_perturbed, col_condition])):
            if col_perturbed in self.rna.obs and (
                col_condition in self.rna.obs):
                pass
            elif col_perturbed in self.rna.obs:
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
        self._columns = {
            **self._columns,
            **dict(col_gene_symbols=col_gene_symbols, 
                   col_condition=col_condition, 
                   col_target_genes=col_condition, 
                   col_perturbed=col_perturbed, 
                   col_cell_type=col_cell_type, 
                   col_sample_id=col_sample_id, 
                   col_guide_rna=col_guide_rna, col_num_umis=col_num_umis,
                   col_guide_split="guide_split")}
        self._keys = {**self._keys,
                      **dict(key_control=key_control, 
                             key_treatment=key_treatment, 
                             key_nonperturbed=key_nonperturbed)}
        print("\n\n")
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)
        print("\n\n", self.rna)
        if "raw" not in dir(self.rna):
            self.rna.raw = self.rna.copy()  # freeze normalized, filtered data
    
    def describe(self, group_by=None, plot=False):
        """Describe data."""
        desc, figs = {}, {}
        # TODO: DO THIS
        return desc, figs
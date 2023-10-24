#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""

import scanpy as sc
import seaborn as sns
import pertpy as pt
import copy
import crispr as cr
import pandas as pd

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"
        

class Omics(object):
    """A class for single-cell genomics analysis and visualization."""
    
    _columns_created = dict(guide_percent="Percent of Cell Guides")

    def __init__(
        self, file_path, assay=None, assay_protein=None, 
        col_gene_symbols="gene_symbols", col_cell_type="leiden", 
        col_sample_id="standard_sample_id", 
        col_condition=None, key_control=None, key_treatment=None, 
        kws_multi=None, **kwargs):
        """
        Initialize Omics class object.

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
        if kws_multi:
            self.adata = cr.pp.create_object_multi(
                file_path, kws_init=dict(
                    assay=assay, assay_protein=assay_protein, 
                    col_gene_symbols=col_gene_symbols, 
                    col_cell_type=col_cell_type, 
                    col_sample_id=col_sample_id, **kwargs),
                **kws_multi)  # create integrated object
        else:
            self.adata = cr.pp.create_object(
                self._file_path, assay=assay, 
                col_gene_symbols=col_gene_symbols,
                col_sample_id=col_sample_id, **kwargs)  # make AnnData
        print(self.adata.obs, "\n\n") if assay else None
        
        # Store Columns & Keys within Columns as Dictionary Attributes
        self._columns = dict(
            col_gene_symbols=col_gene_symbols, col_cell_type=col_cell_type, 
            col_sample_id=col_sample_id, col_batch=col_sample_id, 
            col_condition=col_condition)
        self._keys = dict(key_control=key_control, key_treatment=key_treatment)
        print("\n\n")
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)
        print("\n\n", self.rna)
        if "raw" not in dir(self.rna):
            self.rna.raw = self.rna.copy()  # freeze normalized, filtered data

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
            
    def print(self):
        print(self.rna.obs.head(), "\n\n")
        print(self.adata, "\n\n")
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)
    
    def describe(self, group_by=None, plot=False):
        """Describe data."""
        desc, figs = {}, {}
        gbp = [self._columns["col_cell_type"]]
        if group_by:
            gbp += [group_by]
        print("\n\n\n", self.adata.obs.describe().round(2),
              "\n\n\n")
        print(f"\n\n{'=' * 80}\nDESCRIPTIVES\n{'=' * 80}\n\n")
        print(self.adata.obs.describe())
        for g in gbp:
            print(self.adata.obs.groupby(g).describe().round(2), 
                  f"\n\n{'-' * 40}\n\n")
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
    
    def preprocess(self, assay_protein=None, layer_in=None, copy=False, 
                   kws_scale=True,  by_batch=None, **kwargs):
        """
        Preprocess (specified layer of) data 
        (defaulting to originally-loaded layer).
        Defaults to preprocessing individual samples, then integrating
        with Harmony, if originally provided as separate datasets.
        
        Include "True" or an integer under `kws_scale` to do normal
        mean-centering and unit_variance standardization of 
        gene expression (if integer, sets max_value), or either the 
        letter "z" or a dictionary (of keyword arguments to 
        override defaults) if you want to z-score gene expression with 
        reference to the control condition.
        """
        if assay_protein is None:
            assay_protein = self._assay_protein
        if layer_in is None:
            layer_in = self._layers["counts"]  # raw counts if not specified
        kws = dict(assay_protein=assay_protein, **self._columns, **kwargs)
        if isinstance(kws_scale, str) and kws_scale.lower() == "z":
            kws_scale = {}
        if isinstance(kws_scale, dict):  # if z-scoring GEX ~ control
            znorm_default = {
                "col_reference": self._columns["col_condition"],
                "key_reference": self._keys["key_control"], 
                "col_batch": self._columns["col_sample_id"]}
            kws_scale = {**znorm_default, **kws_scale}  # add arguments
        self.info["methods"]["scale"] = kws_scale
        kws.update(dict(kws_scale=kws_scale))
        adata = self.rna.copy()
        adata.X = adata.layers[layer_in]  # use specified layer
        adata, figs = cr.pp.process_data(adata, **kws)  # preprocess
        # if assay_protein is not None:  # if includes protein assay
        #     ad_p = muon.prot.pp.clr(adata[assay_protein], inplace=False)
        self.figures["preprocessing"] = figs
        if copy is False:
            self.rna = adata
            # if assay_protein is not None:  # if includes protein assay
            #     self.adata[assay_protein] = ad_p
        return adata, figs
                
    def cluster(self, assay=None, method_cluster="leiden", 
                model_celltypist=None,
                plot=True, colors=None,
                kws_pca=None, kws_neighbors=None, 
                kws_umap=None, kws_cluster=None, 
                copy=False, **kwargs):
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
        if "layer" not in kwargs:
            kwargs.update({"layer": self._layers["log1p"]})
        self.info["methods"]["clustering"] = method_cluster
        adata, figs_cl = cr.ax.cluster(
            self.rna.copy() if copy is True else self.rna, 
            assay=assay, method_cluster=method_cluster,
            **self._columns, **self._keys,
            plot=plot, colors=colors, model_celltypist=model_celltypist,
            kws_pca=kws_pca, kws_neighbors=kws_neighbors,
            kws_umap=kws_umap, kws_cluster=kws_cluster, **kwargs)
        if copy is False:
            self.figures.update({"clustering": figs_cl})
            self.rna = adata
        return figs_cl
    
    def annotate_clusters(self, model, copy=False, **kwargs):
        """Use CellTypist to annotate clusters."""
        preds, ct_dot = cr.ax.perform_celltypist(
            self.rna.copy() if copy is True else self.rna, model, 
            majority_voting=True, **kwargs)  # annotate
        self.results["celltypist"] = preds
        if copy is False:
            self.rna.obs = self.rna.obs.join(preds.predicted_labels,
                                             lsuffix="_last")
        sc.pl.umap(self.rna, color=[self._columns["col_cell_type"]] + list(
            preds.predicted_labels.columns))  # UMAP
        return preds, ct_dot
    
    def find_markers(self, assay=None, n_genes=5, layer="scaled", 
                     method="wilcoxon", key_reference="rest", 
                     plot=True, col_cell_type=None, **kwargs):
        if assay is None:
            assay = self._assay
        if col_cell_type is None:
            col_cell_type = self.info["methods"]["clustering"]
        marks, figs_m = cr.ax.find_markers(
            self.adata, assay=assay, method=method, n_genes=n_genes, 
            layer=layer, key_reference=key_reference, plot=plot,
            col_cell_type=col_cell_type, **kwargs)
        print(marks)
        return marks, figs_m
            
    def plot(self, genes=None, genes_highlight=None,
             marker_genes_dict=None, cell_types_circle=None,
             kws_qc=True, kws_umap=None, kws_heat=None, kws_violin=None, 
             kws_matrix=None, kws_dot=None, **kwargs):
        """Create a variety of plots."""
        figs = {}
        if kws_umap is None:
            kws_umap = {}
        cct = kwargs["col_cell_type"] if (
            "col_cell_type" in kwargs) else self._columns["col_cell_type"] 
        lab_cluster = kws_umap.pop("col_cell_type") if (
            "col_cell_type" in kws_umap) else cct
        if genes_highlight and not isinstance(genes_highlight, list):
            genes_highlight = [genes_highlight] if isinstance(
                genes_highlight, str) else list(genes_highlight)
        if cell_types_circle and not isinstance(cell_types_circle, list):
            cell_types_circle = [cell_types_circle] if isinstance(
                cell_types_circle, str) else list(cell_types_circle)
        cgs = self._columns["col_gene_symbols"] if self._columns[
            "col_gene_symbols"] != self.rna.var.index.names[0] else None
            
        # Pre-Processing/QC
        if kws_qc:
            if kws_qc is True:
                kws_qc = {"hue": self._columns["col_sample_id"]}
            cr.pp.perform_qc(self.adata.copy(), layer=self._layers["counts"], 
                             **kws_qc)  # plot QC
        
        # Gene Expression
        figs["gex"] = cr.pl.plot_gex(
            self.rna, col_cell_type=lab_cluster, genes=genes, kind="all", 
            col_gene_symbols=cgs, marker_genes_dict=marker_genes_dict, 
            kws_violin=kws_violin, kws_heat=kws_heat, 
            kws_matrix=kws_matrix, kws_dot=kws_dot)  # GEX
            
        # UMAP
        if "X_umap" in self.adata.obsm or lab_cluster in self.rna.obs.columns:
            print("\n<<< PLOTTING UMAP >>>")
            figs["umap"] = cr.pl.plot_umap(
                self.rna, col_cell_type=lab_cluster, **kws_umap, 
                genes=genes, col_gene_symbols=cgs)
        else:
            print("\n<<< UMAP NOT AVAILABLE TO PLOT. RUN `.cluster()`.>>>")
        return figs
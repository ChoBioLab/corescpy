#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""

import scanpy as sc
import seaborn as sns
import pertpy as pt
# import copy
import muon
import crispr as cr
# import pandas as pd

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
                    col_condition=col_condition,
                    key_control=key_control, 
                    key_treatment=key_treatment,
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
        self._keys = dict(key_control=key_control, 
                          key_treatment=key_treatment)
        print("\n\n")
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)
        
        # Let Property Setters Run
        self.rna = self.adata[self._assay] if self._assay else self.adata
        self.obs = self.adata.obs
        self.uns = self.rna.uns
        self.var = self.rna.var
        print("\n\n", self.rna)
        if "raw" not in dir(self.rna):
            self.rna.raw = self.rna.copy()  # freeze normalized, filtered data
        self.rna.obs.head()

    @property
    def rna(self):
        """Get RNA modality of AnnData."""
        return self._rna

    @rna.setter
    def rna(self, value) -> None:
        """Set gene expression modality of data."""
        if isinstance(self.adata, muon.MuData):
            self.adata.mod[self._assay] = value
            self.adata.update()
            # self.adata = muon.MuData({**dict(zip(self.adata.mod.keys(), [
            #     self.adata[x] for x in self.adata.mod.keys()])), 
            #                           self._assay: value})
            self._rna = self.adata.mod[self._assay]
        else:
            if self._assay: 
                self.adata[self._assay] = value
            else:
                self.adata = value
            self._rna = value
            
    @property
    def obs(self):
        """Get `.obs` attribute of AnnData."""
        return self._obs

    @obs.setter
    def obs(self, value) -> None:
        self.adata.obs = value
        self._obs = value
        
    @property
    def gex(self):
        """Get gene expression modality with prefix-less index."""
        return self._gex

    @gex.setter
    def gex(self, value) -> None:
        if isinstance(self.adata, muon.MuData):
            ann = self.adata.mod[self._assay].copy()
            ann.index = ann.obs.reset_index()[
                self._columns["col_gene_symbol"]]  # prefix-less gene names
        else:
            ann = self.rna
        self._gex = ann
            
    @property
    def uns(self):
        """Get `.uns` attribute of adata's gene expression modality."""
        return self._uns

    @uns.setter
    def uns(self, value) -> None:
        if self._assay: 
            self.adata[self._assay].uns = value
        else:
            self.adata.uns = value
        self._uns = value
        return self._uns  
                
    @property
    def var(self):
        """Get `.var` attribute of .adata's gene expression modality."""
        return self._var

    @var.setter
    def var(self, value) -> None:
        if isinstance(self.adata, muon.MuData):
            self.adata.mod[self._assay].var = value
            self.adata.update()
        elif self._assay: 
            self.adata[self._assay].var = value
        else:
            self.adata.var = value
        self._var = value
            
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
            
    def plot(self, genes=None, genes_highlight=None,
             marker_genes_dict=None, cell_types_circle=None,
             kws_qc=True, kws_umap=None, kws_heat=None, kws_violin=None, 
             kws_matrix=None, kws_dot=None, kind="all", **kwargs):
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
            self.rna, col_cell_type=lab_cluster, genes=genes, kind=kind, 
            col_gene_symbols=cgs, marker_genes_dict=marker_genes_dict, 
            kws_violin=kws_violin, kws_heat=kws_heat, 
            kws_matrix=kws_matrix, kws_dot=kws_dot)  # GEX
            
        # UMAP
        if "X_umap" in self.rna.obsm or lab_cluster in self.rna.obs.columns:
            print("\n<<< PLOTTING UMAP >>>")
            figs["umap"] = cr.pl.plot_umap(
                self.rna, col_cell_type=lab_cluster, **kws_umap, 
                genes=genes, col_gene_symbols=cgs)
        else:
            print("\n<<< UMAP NOT AVAILABLE TO PLOT. RUN `.cluster()`.>>>")
        return figs
    
    def plot_umap(self, color=None, group=None, **kwargs):
        """Plot UMAP."""
        if color is None:
            color = self._columns["col_cell_type"]
        if group:
            fig = cr.pl.plot_umap_split(self.rna, group, 
                                        color=color, **kwargs)
        else:
            if not isinstance(color, str) and len(color) > 1 and (
                color[0] in self.rna.var_names):  # multi-feature UMAP
                fig = cr.pl.plot_umap_multi(self.rna, color, **kwargs)
            else:  # normal UMAP embedding (categorical or continous)
                fig = sc.pl.umap(self.rna, color=color, 
                                **{"legend_loc": "on data", **kwargs})
        return fig
    
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
        print(type(adata))
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
    
    def annotate_clusters(self, model, mode="best match", 
                          p_threshold=0.5, 
                          over_clustering=None, min_proportion=0, 
                          copy=False, **kwargs):
        """Use CellTypist to annotate clusters."""
        adata = self.rna.copy()
        adata.X = adata.layers[self._layers["log1p"]]
        preds, ct_dot = cr.ax.perform_celltypist(
            adata, model, majority_voting=True, p_threshold=p_threshold,
            mode=mode, over_clustering=over_clustering, 
            min_proportion=min_proportion, **kwargs)  # annotate
        self.results["celltypist"] = preds  # store results
        if copy is False:  # assign if performing inplace
            self.rna.obs = self.rna.obs[self.rna.obs.columns.difference(
                preds.predicted_labels.columns)].join(
                    preds.predicted_labels, lsuffix="_last")
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
          
    def run_composition_analysis(
        self, assay=None, layer=None, col_list_lineage_tree=None,
        covariates=None, reference_cell_type="automatic", 
        analysis_type="cell_level", est_fdr=0.05, generate_sample_level=False,
        plot=True, copy=False, **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        for x in [self._columns, self._keys]:
            for c in x:  # iterate column/key name attributes
                if c not in kwargs:  # if not passed as argument to method...
                    kwargs.update({c: x[c]})  # & use object attribute
        col_condition, col_cell_type = [
            kwargs.pop(x) if x in kwargs else None for x in [
                "col_condition", "col_cell_type"]]  # extract from kwargs
        output = cr.ax.analyze_composition(
            self.adata, col_condition,  col_cell_type, 
            assay=assay, layer=layer, reference_cell_type=reference_cell_type, 
            analysis_type=analysis_type, 
            generate_sample_level=generate_sample_level,
            col_list_lineage_tree=col_list_lineage_tree,  # only for TASCCoda
            covariates=covariates, est_fdr=est_fdr, plot=plot, out_file=None, 
            copy=copy, key_reference_cell_type="automatic", **kwargs)
        if copy is False:
            self.results["composition"] = output
        return output
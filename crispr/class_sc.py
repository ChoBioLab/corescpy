#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""

import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import pertpy as pt
import squidpy as sq
import liana
import os
from copy import deepcopy
import muon
import anndata
import crispr as cr
import pandas as pd

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"
        

class Omics(object):
    """A class for single-cell genomics analysis and visualization."""
    
    _columns_created = dict(guide_percent="Percent of Cell Guides")

    def __init__(
        self, file_path, assay=None, assay_protein=None, raw=False,
        col_gene_symbols="gene_symbols", col_cell_type="leiden", 
        col_sample_id=None, col_subject=None, 
        col_condition=None, col_num_umis=None, 
        key_control=None, key_treatment=None, kws_multi=None, **kwargs):
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
            col_num_umis (str, optional): Name of column in `.obs` with 
                the UMI counts. Defaults to None
        """
        print("\n\n<<< INITIALIZING CRISPR CLASS OBJECT >>>\n")
        self.pdata = None  # for pseudobulk data if ever created
        self._assay = assay
        self._assay_protein = assay_protein
        self._file_path = file_path
        self._layers = {**cr.pp.get_layer_dict(), 
                        "layer_perturbation": "X_pert"}
        self._integrated = kws_multi is not None
        if kwargs:
            print(f"\nUnused keyword arguments: {kwargs}.\n")
        if kws_multi and col_sample_id is None:
            col_sample_id = "unique.idents"
        
        # Create Attributes to Store Results/Figures
        self.figures = {"main": {}}
        self.results = {"main": {}}
        self.info = {"descriptives": {}, 
                     "guide_rna": {},
                     "methods": {}}  # extra info to store post-use of methods
        
        # Store Columns & Keys within Columns as Dictionary Attributes
        self._columns = dict(
            col_gene_symbols=col_gene_symbols, col_cell_type=col_cell_type, 
            col_sample_id=col_sample_id, col_batch=col_sample_id,
            col_subject=col_subject,  # e.g., patient ID rather than sample
            col_condition=col_condition, col_num_umis=col_num_umis)
        self._keys = dict(key_control=key_control, 
                          key_treatment=key_treatment)
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)
        
        # Create Object & Store Raw Counts
        if kws_multi:
            self._integrated = True
            self.adata = cr.pp.create_object_multi(
                file_path, kws_init=dict(
                    assay=assay, assay_protein=assay_protein, 
                    col_gene_symbols=col_gene_symbols, 
                    col_condition=col_condition,
                    key_control=key_control, 
                    key_treatment=key_treatment,
                    col_cell_type=col_cell_type, raw=raw,
                    col_sample_id=col_sample_id, **kwargs),
                **kws_multi)  # create integrated object
        else:
            self.adata = cr.pp.create_object(
                self._file_path, assay=assay, raw=raw,
                col_gene_symbols=col_gene_symbols,
                col_sample_id=col_sample_id, **kwargs)  # make AnnData
        print(self.adata.obs, "\n\n") if assay else None
        
        # Let Property Setters Run
        self.rna = self.adata[self._assay] if self._assay else self.adata
        self.obs = self.adata.obs
        self.uns = self.rna.uns
        self.var = self.rna.var
        if "raw" not in dir(self.rna):
            self.rna.raw = self.rna.copy()  # freeze normalized, filtered data
        if self._columns["col_cell_type"] in self.rna.obs and (
            not isinstance(self.rna.obs[self._columns["col_cell_type"]], 
                           pd.Categorical)):
            self.rna.obs[self._columns["col_cell_type"]] = self.rna.obs[
                self._columns["col_cell_type"]].astype("category")  # object -> category
        print("\n\n", self.rna)
        print(self.rna.var.head())
        print(self.rna.obs.head())

    @property
    def rna(self):
        """Get RNA modality of AnnData."""
        return self.adata[self._assay] if self._assay else self.adata

    @rna.setter
    def rna(self, value) -> None:
        """Set gene expression modality of data."""
        if isinstance(self.adata, muon.MuData):
            self.adata.mod[self._assay] = value
            self.adata.update()
            # self.adata = muon.MuData({**dict(zip(self.adata.mod.keys(), [
            #     self.adata[x] for x in self.adata.mod.keys()])), 
            #                           self._assay: value})
        else:
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
        return (self.adata[self._assay] if self._assay else self.adata).uns

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
        return (self.adata[self._assay] if self._assay else self.adata).var

    @var.setter
    def var(self, value) -> None:
        if isinstance(self.adata, muon.MuData):
            self.adata.mod[self._assay].var = value
            self.adata.update()
        elif self._assay: 
            self.adata[self._assay].var = value
        else:
            self.adata.var = value
            
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

    def map(self, gene=None, col_cell_type=True, **kwargs):
        """Plot GEX &/or cell type(s) on UMAP."""
        if col_cell_type is True:
            col_cell_type = self._columns["col_cell_type"]
        col_cell_type, gene = [[] if x is None else [x] if isinstance(
            x, str) else list(x) for x in [col_cell_type, gene]]
        fig = sc.pl.umap(self.rna, color=gene + col_cell_type, **{
            "legend_loc": "on data", "legend_fontweight": "medium", **kwargs})
        return fig
            
    def plot(self, genes, kind="all", genes_highlight=None, subset=None,
             group=None, layer=None, kws_qc=False, marker_genes_dict=None, 
             kws_umap=None, kws_heat=None, kws_violin=None, kws_dot=None, 
             kws_matrix=None, cell_types_circle=None, **kwargs):
        """Create a variety of plots."""
        figs = {}
        if group is None:  # if unspecified grouping variable...
            group = self._columns["col_cell_type"]  # default cell type column
        if kind == "all":
            kind = ["all", "umap"]
        umap = "umap" in kind or (isinstance(kind, str) and kind == "all")
        if "umap" in kind:  # remove "umap" from "kind" since stored in `umap`
            kind = list(set(kind).difference(set(["umap"])))
        if len(kind) == 1:
            kind = kind[0]  # in case "kind" is length 1 after removing umap
        adata = (self.rna if subset is None else self.rna[subset]).copy()
        if kws_umap is None:
            kws_umap = {}
        if "legend_loc" not in kws_umap:
            kws_umap["legend_loc"] = "on data"
        genes_highlight = cr.tl.to_list(genes_highlight)  # list or None
        cell_types_circle = cr.tl.to_list(cell_types_circle) 
        cgs = self._columns["col_gene_symbols"] if self._columns[
            "col_gene_symbols"] != self.rna.var.index.names[0] else None
            
        # Pre-Processing/QC
        if kws_qc:
            if kws_qc is True:
                kws_qc = {"hue": self._columns["col_sample_id"]}
            cr.pp.perform_qc(adata, layer=self._layers["counts"], **kws_qc)
        
        # Gene Expression
        if "categories_order" in kwargs:  # merge category kws; subset if needed
            kws_violin, kws_heat, kws_matrix, kws_dot = [{
                "categories_order": cr.tl.merge(x, kwargs["categories_order"])
                } for x in [kws_violin, kws_heat, kws_matrix, kws_dot]]
            adata = adata[adata.obs[group].isin(kwargs["categories_order"])]
        if isinstance(kind, str) and (kind != "all"):  # if only 1 plot type
            kws_violin, kws_heat, kws_matrix, kws_dot = [
                kwargs if kind in x[0] and x[1] is None else x[1] 
                for x in zip(["violin", ["heat", "hm"], "matrix", "dot"], [
                    kws_violin, kws_heat, kws_matrix, kws_dot])
                ]  # can specify plot arguments via overall keyword arguments
        print(kws_violin, kws_heat, kws_matrix, kws_dot)
        figs["gex"] = cr.pl.plot_gex(
            adata, col_cell_type=group, genes=genes, kind=kind, layer=layer, 
            col_gene_symbols=cgs, marker_genes_dict=marker_genes_dict,
            kws_violin=kws_violin, kws_heat=kws_heat, kws_matrix=kws_matrix, 
            kws_dot=kws_dot)  # GEX
            
        # UMAP
        if umap is True:
            if "X_umap" in self.rna.obsm or group in self.rna.obs.columns:
                print("\n<<< PLOTTING UMAP >>>")
                figs["umap"] = cr.pl.plot_umap(
                    adata, col_cell_type=group, genes=genes, **kws_umap, 
                    col_gene_symbols=cgs, cell_types_circle=cell_types_circle)
            else:
                print("\n<<< UMAP NOT AVAILABLE. RUN `.cluster()`.>>>")
        return figs
    
    def plot_umap(self, color=None, group=None, cmap="magma", **kwargs):
        """Plot UMAP."""
        if color is None:  # if color-coding column unspecified...
            color = self._columns["col_cell_type"]  # ...color by cell type
        kws = {"frameon": False, "use_raw": False, "cmap": cmap, **kwargs}
        if group:  # if separating by group, return this plot
            return cr.pl.plot_umap_split(self.rna, group, color=color, **kws)
        if not isinstance(color, str) and color[0] in self.rna.var_names:
            fig = cr.pl.plot_umap_multi(self.rna, color, **kws)  # multi-gene
        else:  # ...or single UMAP embedding (categorical or continous)
            fig = sc.pl.umap(self.rna, color=color, **kws)  # UMAP
        return fig
    
    def plot_compare(self, genes, col_condition=None, col_cell_type=None, 
                     layer=None, subset=None, **kwargs):
        """
        Plot gene expression across cell types and 
        condition(s) (string for 1 condition or list for 2, or None
        to default to `self._columns["col_condition"]`).
        """
        ann = (self.rna if subset is None else self.rna[subset]).copy()
        if layer:
            ann.X = (ann.layers[layer] if layer in ann.layers else ann.layers[
                self._layers[layer]]).copy()
        con, cct = [x[1] if x[1] else self._columns[x[0]] 
                    for x in zip(["col_condition", "col_cell_type"], [
                        col_condition, col_cell_type])]  # specs v. default
        con, col = [con, None] if isinstance(con, str) else con  # 1 or 2?
        fig = cr.pl.plot_cat_split(
            ann, con, col_cell_type=cct, genes=genes, 
            **{"columns": col, **kwargs})  # plot by groups
        return fig
    
    def plot_coex(self, genes, use_raw=False, **kwargs):
        """Plot co-expression of a list of genes on a UMAP."""
        ggg = "_".join(genes)  # co-expression variable name (ex: STAT1_ERCC1)
        adata = sc.tl.score_genes(self.rna, genes, score_name=ggg, 
                                  use_raw=use_raw, copy=True)  # score co-GEX
        fig = sc.pl.umap(adata, color=ggg, use_raw=use_raw, **kwargs)  # plot
        return fig
    
    def preprocess(self, assay_protein=None, layer_in="counts", copy=False, 
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
        layer = layer_in if layer_in in self.adata.layers else self._layers[
            layer_in]  # detect layer; raw counts if not specified
        kws = dict(assay_protein=assay_protein, **self._columns, **kwargs)
        kws_scale = {} if isinstance(kws_scale, str) and kws_scale.lower(
            ) == "z" else deepcopy(kws_scale)  # scale kws processing
        if isinstance(kws_scale, dict) and all([x not in kws_scale for x in [
            "max_value", "zero_center"]]):  # if z-scoring GEX ~ control
            znorm_default = {
                "col_reference": self._columns["col_condition"],
                "key_reference": self._keys["key_control"], 
                "col_batch": self._columns["col_sample_id"]}
            kws_scale = {**znorm_default, **kws_scale}  # merge arguments
        self.info["methods"]["scale"] = kws_scale
        kws.update(dict(kws_scale=kws_scale))
        adata = self.rna.copy()
        adata.X = adata.layers[layer]  # use specified layer
        adata, figs = cr.pp.process_data(adata, **kws)  # preprocess
        # if assay_protein is not None:  # if includes protein assay
        #     ad_p = muon.prot.pp.clr(adata[assay_protein], inplace=False)
        for x in kws:
            adata.obs.loc[:, x] = str(kws[x])  # store parameters in `.obs`
        print(type(adata))
        if copy is False:
            self.rna = adata
            self.figures["preprocessing"] = figs
            # if assay_protein is not None:  # if includes protein assay
            #     self.adata[assay_protein] = ad_p
        return adata, figs
    
    def downsample(self, subset=None, assay=None, counts_per_cell=None, 
                   total_counts=None, replace=False, seed=1618, **kwargs):
        """Downsample counts/`.X` (optionally, of a subset). NOT IN-PLACE."""
        adata = self.adata[assay].copy() if assay else self.adata.copy()
        if subset is not None:
            adata = adata[subset]  # subset if desired
        return sc.pp.downsample_counts(
            adata, counts_per_cell=counts_per_cell, total_counts=total_counts, 
            random_state=seed, replace=replace, copy=True)  # downsample
    
    def bulk(self, mode="sum", subset=None, col_cell_type=None, 
             col_sample_id=None, layer="counts", copy=True, **kwargs):
        """Create pseudo-bulk data (col_sample_id=False to ignore)."""
        col_cell_type, col_sample_id = [x[1] if x[1] else self._columns[
            x[0]] for x in zip(["col_cell_type", "col_sample_id"], 
                               [col_cell_type, col_sample_id])]
        if col_sample_id is False:  # False = don't account for sample ID
            col_sample_id = None  # b/c if use None argument -> use ._columns
        kws_def = dict(col_sample_id=self._columns["col_sample_id"],
                       mode="sum", kws_process=True)  # default arguments
        adata = self.rna[subset].copy() if subset else self.rna.copy()
        pdata = cr.tl.create_pseudobulk(
            adata, col_cell_type, col_sample_id=col_sample_id,
            mode=mode, layer=layer, **kwargs)  # create pseudo-bulk data
        if copy is False:
            self.pdata = pdata
        return pdata
                
    def cluster(self, assay=None, method_cluster="leiden", layer="scaled",
                resolution=1, kws_pca=None, kws_neighbors=None, 
                kws_umap=None, kws_cluster=None, kws_celltypist=None, 
                plot=True, colors=None, copy=False, **kwargs):
        """Perform dimensionality reduction and create UMAP."""
        if assay is None:
            assay = self._assay
        if self._integrated is True:
            kws_pca = False  # so will use Harmony-adjusted PCA
        if self._columns["col_sample_id"] or self._columns["col_batch"]:
            if colors is None:
                colors = []
                for x in [self._columns["col_sample_id"], 
                          self._columns["col_batch"]]:
                    if x:
                        colors += [x]  # add sample & batch UMAP
        self.info["methods"]["clustering"] = method_cluster
        ann = self.adata.copy()
        if layer is not None:
            ann.X = ann.layers[layer if (
                layer in self._layers) else self._layers[layer]].copy()
        kws = dict(method_cluster=method_cluster, kws_pca=kws_pca, 
                   kws_neighbors=kws_neighbors, kws_umap=kws_umap, 
                   kws_cluster=kws_cluster, resolution=resolution)
        adata, figs_cl = cr.ax.cluster(
            ann, assay=assay, **self._columns, **self._keys, colors=colors, 
            kws_celltypist=kws_celltypist, **kws, **kwargs)  # cluster data
        for x in kws:
            adata.obs.loc[:, x] = str(kws[x])  # store parameters in `.obs`
        if copy is False:
            self.figures.update({"clustering": figs_cl})
            self.rna = adata
            return figs_cl
        return adata
    
    def annotate_clusters(self, model, mode="best match", layer="log1p", 
                          p_threshold=0.5, over_clustering=None, 
                          min_proportion=0, copy=False, **kwargs):
        """Use CellTypist to annotate clusters."""
        adata = self.rna.copy()
        adata.X = adata.layers[self._layers[layer]]  # log 1 p layer
        c_t = kwargs.pop("col_cell_type", self._columns["col_cell_type"])
        ann, res, figs = cr.ax.perform_celltypist(
            adata, model, majority_voting=True, p_threshold=p_threshold,
            mode=mode, over_clustering=over_clustering, col_cell_type=c_t,
            min_proportion=min_proportion, **kwargs)  # annotate
        self.figures["celltypist"], self.results["celltypist"] = figs, res
        if copy is False:  # assign if performing inplace
            self.rna = ann
        return ann, [res, figs]
    
    def find_markers(self, assay=None, n_genes=10, layer="log1p", 
                     method="wilcoxon", key_reference="rest", 
                     plot=True, col_cell_type=None, **kwargs):
        if assay is None:
            assay = self._assay
        if col_cell_type is None:  # if cell type column not specified...
            if "clustering" in self.info["methods"]:  # use leiden/louvain #s
                col_cell_type = self.info["methods"]["clustering"]
            else:  # otherwise, try getting from _columns attribute
                col_cell_type = self._columns["col_cell_type"]
        marks, figs_m = cr.ax.find_marker_genes(
            self.adata, assay=assay, method=method, n_genes=n_genes, 
            layer=layer, key_reference=key_reference, plot=plot,
            col_cell_type=col_cell_type, **kwargs)  # find marker genes
        marks.groupby(col_cell_type).apply(lambda x: print(x.head(3)))
        return marks, figs_m

    def run_composition_analysis(
        self, assay=None, layer=None, col_list_lineage_tree=None,
        covariates=None, reference_cell_type="automatic", 
        analysis_type="cell_level", est_fdr=0.05, generate_sample_level=True,
        plot=True, copy=False, **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        for x in [self._columns, self._keys]:
            for c in x:  # iterate column/key name attributes
                if c not in kwargs:  # if not passed as argument to method...
                    kwargs.update({c: x[c]})  # & use object attribute
        col_condition, col_cell_type = [kwargs.pop(x, self._columns[
            x]) for x in ["col_condition", "col_cell_type"]]  # extract kwargs
        output = cr.ax.analyze_composition(
            self.adata, col_condition,  col_cell_type, 
            assay=assay, layer=layer, reference_cell_type=reference_cell_type, 
            analysis_type=analysis_type, plot=plot, out_file=None, 
            generate_sample_level=generate_sample_level,
            col_list_lineage_tree=col_list_lineage_tree,  # only for TASCCoda
            covariates=covariates, est_fdr=est_fdr, copy=copy, 
            key_reference_cell_type="automatic", **kwargs)
        if copy is False:
            self.results["composition"] = output
        return output
    
    def run_dialogue(self, n_programs=3, col_confounder=None,
                     col_cell_type=None, cmap="coolwarm", vcenter=0, 
                     layer="log1p", **kws_plot):
        """Analyze <`n_programs`> multicellular programs."""
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        adata = self.rna.copy()
        if layer is not None:
            adata.X = adata.layers[layer]
        col = self._columns["col_perturbed" if (
            "col_perturbed" in self._columns) else "col_condition"] 
        d_l = pt.tl.Dialogue(
            sample_id=col, n_mpcs=n_programs, celltype_key=col_cell_type, 
            n_counts_key=self._columns["col_num_umis"])
        pdata, mcps, w_s, ct_subs = d_l.calculate_multifactor_PMD(
            adata, normalize=True)
        mcp_cols = list(set(pdata.obs.columns).difference(adata.obs.columns))
        cols = cr.pl.square_grid(len(mcp_cols) + 2)[1]
        fig = sc.pl.umap(
            pdata, color=mcp_cols + [col, col_cell_type],
            ncols=cols, cmap=cmap, vcenter=vcenter, **kws_plot)
        if col_confounder:  # correct for a confounding variable?
            res, p_new = d_l.multilevel_modeling(
                ct_subs=ct_subs, mcp_scores=mcps, ws_dict=w_s, 
                confounder=col_confounder)
            self.results[f"dialogue_confounder_{col_confounder}"] = res, p_new
        self.results["dialogue"] = pdata, mcps, w_s, ct_subs
        self.figures["dialogue"] = fig
        return fig
    
    def run_gsea(self, key_condition, col_condition=None, layer="log1p",
                 filter_by_highly_variable=True, copy=False, 
                 pseudobulk=True, **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        if col_condition is None:
            col_condition = self._columns["col_cell_type"]
            if self._columns["col_condition"] is not None:
                col_condition = [col_condition, self._columns[
                    "col_condition"]]
        csid = kwargs.pop("col_sample_id", self._columns["col_sample_id"])
        for x in [self._columns, self._keys]:
            for c in x:  # iterate column/key name attributes
                if c not in kwargs and c not in [
                    "col_condition", "col_sample_id"]:
                    kwargs.update({c: x[c]})  # & use object attribute
        if pseudobulk not in [None, False]:
            if isinstance(pseudobulk, anndata.AnnData):
                data = pseudobulk.copy()  # if pseudobulk provided as data
            elif self.pdata is None:  # if pseudobulk not already an attribute
                if not isinstance(pseudobulk, dict):
                    pseudobulk = {}
                data = self.bulk(**pseudobulk)  # ...create pseudobulk
                if copy is None:
                    self.pdata = data  # store pseudobulk data in `self.pdata`
            else:  # otherwise, use existing pseudobulk data in `self.pdata`
                data = self.pdata.copy()
        else:
            data = self.rna.copy()
        output = cr.ax.perform_gsea(
            data, adata_sc=self.rna.copy(), col_sample_id=csid,
            filter_by_highly_variable=filter_by_highly_variable, 
            col_condition=col_condition, key_condition=key_condition,
            layer=layer if layer in self.rna.layers else self._layers[layer], 
            copy=copy, pseudobulk=pseudobulk, **kwargs)  # GSEA
        if copy is False:
            if pseudobulk is True:
                self.pdata = output[0]
            else:
                self.adata = output[0]
            self.results["gsea"] = output[1]
            self.figures["gsea"] = output[-1]
        return output
    
    
    def plot_gsea(self, ifn_pathways=True, p_threshold=0.0001, **kwargs):
        """
        Plot stored GSEA results (e.g., with different pathways, 
        p-value threshold, or other options than
        you chose when initially running the analysis).
        """
        figs = cr.pl.plot_gsea_results(
            self.rna.copy(), self.results["gsea"]["gsea_results"], 
            p_threshold=p_threshold, **kwargs, ifn_pathways=ifn_pathways)
        return figs
    
    def run_fx_analysis(self, col_covariates=None, layer="counts", 
                        plot_stat="p_adj", uns_key="pca_anova", 
                        copy=False, figsize=30, **kwargs):
        """Perform pseudo-bulk functional analysis."""
        if col_covariates is None:
            if self._columns["col_condition"] is None:
                raise ValueError("Must specify `col_covariates` if "
                                 "`self._columns['col_condition']` is None.")
            col_covariates = [self._columns["col_condition"]]
        cct = kwargs.pop("col_cell_type", self._columns["col_cell_type"])
        layer = layer if layer in self.adata.layers else self._layers[layer]
        out = cr.ax.perform_fx_analysis_pseudobulk(
            self.rna.copy(), cct, col_covariates, layer=layer, 
            figsize=figsize, plot_stat=plot_stat, uns_key=uns_key, 
            col_sample_id=self._columns["col_sample_id"], 
            **kwargs)  # perform functional analysis of pseudo-bulk data
        if copy is False:
            self.rna = out[0]
            self.results["fx_analysis"] = out[1:-1]
            self.figures["fx_analysis"] = out[-1]
        return out
    
    def calculate_receptor_ligand(
        self, method="liana", subset=None, layer="log1p", top_n=20,
        col_cell_type=None, col_condition=None, col_subject=True,
        cmap="magma", kws_plot=None, key_sources=None, key_targets=None, 
        resource="CellPhoneDB", 
        n_perms=10, p_threshold=0.01, remove_ns=True, figsize=None, **kwargs):
        """Calculate receptor-ligand interactions."""
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if col_subject is True:
            col_subject = self._columns["col_cell_type"]
        if col_condition is True:
            col_condition = self._columns["col_condition"]
        if layer and layer not in self.adata.layers:
            layer = self.adata._layers[layer]
        adata = (self.rna[subset] if subset is not None else self.rna).copy()
        res, fig = cr.ax.analyze_receptor_ligand(
            adata, col_condition=col_condition, col_subject=col_subject, 
            method=method, layer=layer, layer_counts=self._layers["counts"],
            key_sources=key_sources, key_targets=key_targets, copy=False, 
            col_cell_type=col_cell_type, top_n=top_n, remove_ns=remove_ns,
            cmap=cmap, p_threshold=p_threshold, figsize=figsize, 
            kws_plot=kws_plot, resource=resource, **kwargs)  # analyze
        self.results["receptor_ligand"] = res
        self.figures["receptor_ligand"] = fig
        self.results["receptor_ligand_info"] = {
            "subset": subset, "col_condition": col_condition}
        return adata, res, fig
    
    def plot_receptor_ligand(self, key_sources=None, key_targets=None,
                             title=None, out_dir=None, **kwargs):
        """Plot previously-run receptorout_dir-ligand analyses."""
        subset = self.results["receptor_ligand_info"]["subset"]
        res = self.results["receptor_ligand"]["liana_res"]
        if isinstance(res, dict):
            figs = {}
            ccc = self.results["receptor_ligand_info"]["col_condition"]
            for c in res:
                if c.lower() == "all":
                    continue
                # res = (self.rna if c.lower() == "overall" else self.rna[
                #     self.rna[ccc] == c]).copy()
                # if subset:
                #     res = res[subset]
                figs[c] = cr.pl.plot_receptor_ligand(
                    liana_res=res[c], title=f"{title} ({c})", **kwargs,
                    # adata=res,
                    key_sources=key_sources, key_targets=key_targets)
                if out_dir:
                    for r in figs[c]:
                        figs[c][r].save(os.path.join(
                            out_dir, f"receptor_ligand_{c}_{r}.png"), 
                                        limitsize=False)
        else:
            # res = (self.rna if subset is None else self.rna[subset]).copy()
            figs = cr.pl.plot_receptor_ligand(
                liana_res=res, 
                # adata=res,
                title=title,
                key_sources=key_sources, key_targets=key_targets, **kwargs)
            if out_dir:
                for r in figs:
                    figs[r].save(os.path.join(
                        out_dir, f"receptor_ligand.png"), limitsize=False)
        return figs
            
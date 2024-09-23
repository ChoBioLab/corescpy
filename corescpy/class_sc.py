#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base, general class from which others (e.g., Spatial, Crispr) inherit.

Use directly for vanilla/uni-modal single-cell RNA-seq data.

@author: E. N. Aslinger
"""

import traceback
import os
from warnings import warn
import functools
import itertools
from copy import deepcopy
import scipy
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import pertpy as pt
# import blitzgsea as blitz
import liana
import spatialdata
import muon
import anndata
import celltypist
import pandas as pd
import numpy as np
import corescpy as cr

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Omics(object):
    """A class for single-cell genomics analysis and visualization."""

    _columns_created = dict(guide_percent="Percent of Cell Guides")

    def __init__(self, file_path, prefix=None, assay=None, assay_protein=None,
                 raw=False, col_gene_symbols="gene_symbols", spatial=False,
                 col_cell_type="leiden", col_sample_id=None, col_subject=None,
                 col_condition=None, key_control=None, key_treatment=None,
                 kws_multi=None, verbose=True, **kwargs):
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
                        pass to `corescpy.pp.combine_matrix_protospacer`
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
                        passed to the `corescpy.pp.create_object()`
                        function (e.g., `col_gene_symbols`) can be
                        specified as normal if they are common across
                        samples; otherwise, specify them as lists in
                        the same order as the `file` dictionary.
            prefix (str, optional):
                As per Scanpy documentation: 'Any prefix before
                matrix.mtx, genes.tsv and barcodes.tsv. For instance,
                if the files are named patientA_matrix.mtx,
                patientA_genes.tsv and patientA_barcodes.tsv the prefix
                is patientA_. (Default: no prefix)' Defaults to None.
            assay (str, optional): Name of the gene expression assay if
                loading a multi-modal data object (e.g., "rna").
                Defaults to None.
            assay_protein (str, optional): Name of the assay containing
                the protein expression modality, if available.
                For instance, if "adt", `self.adata["adt"]` would be
                expected to contain the AnnData object for the
                protein modality. ONLY FOR MULTI-MODAL DATA for certain
                bonus visualization methods. Defaults to None.
            col_gene_symbols (str, optional): Specify "gene_symbols"
                to use gene names or "gene_ids" to use EnsemblIDs) as
                the index of `.var`. Defaults to "gene_symbols".
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
        if verbose is True:
            print("\n\n<<< INITIALIZING OMICS CLASS OBJECT >>>\n")
        if "kws_process_guide_rna" in kwargs and kwargs[
                "kws_process_guide_rna"] in [False, None]:
            _ = kwargs.pop("kws_process_guide_rna")
        kpg = kwargs.pop("kws_process_guide_rna", None)
        key_table = kwargs.pop("key_table", None)
        col_num_umis = kpg["col_num_umis"] if kpg not in [
            None, False] else kwargs["col_num_umis"] if (
                "col_num_umis" in kwargs) else None
        if kws_multi and col_sample_id is None:
            col_sample_id = "unique.idents"
        col_batch = kwargs.pop("col_batch", col_sample_id)
        self.pdata = None  # for pseudobulk data if ever created
        self._assay = assay
        self._assay_protein = assay_protein
        self._file_path = file_path
        self._layers = {**cr.get_layer_dict(), "layer_perturbation": "X_pert"}
        self._integrated = kws_multi is not None

        # Create Attributes to Store Results/Figures/Methods
        self.results, self.figures = {}, {}
        self.info = {"descriptives": {}, "guide_rna": {},
                     "methods": {}}  # extra info to store post-use of methods

        # Store Columns & Keys within Columns as Dictionary Attributes
        self._columns = dict(
            col_gene_symbols=col_gene_symbols, col_cell_type=col_cell_type,
            col_sample_id=col_sample_id, col_batch=col_batch,
            col_subject=col_subject,  # e.g., patient ID rather than sample
            col_condition=col_condition, col_num_umis=col_num_umis)
        self._keys = dict(key_control=key_control,
                          key_treatment=key_treatment, key_table=key_table)
        if verbose is True:
            for q in [self._columns, self._keys]:
                cr.tl.print_pretty_dictionary(q)

        # Create Object & Store Raw Counts
        if kws_multi not in [False, None]:
            self._integrated = True
            self.adata = cr.pp.create_object_multi(
                file_path, kws_init=dict(
                    prefix=prefix, assay=assay, assay_protein=assay_protein,
                    col_gene_symbols=col_gene_symbols,
                    col_condition=col_condition, kws_process_guide_rna=kpg,
                    key_control=key_control, key_treatment=key_treatment,
                    col_cell_type=col_cell_type, raw=raw, key_table=key_table,
                    col_sample_id=col_sample_id, **kwargs), spatial=spatial,
                **kws_multi)  # create integrated object
        else:
            self.adata = cr.pp.create_object(
                self._file_path, prefix=prefix, col_sample_id=col_sample_id,
                col_gene_symbols=col_gene_symbols, kws_process_guide_rna=kpg,
                spatial=spatial, assay=assay, raw=raw,
                key_table=key_table, **kwargs)  # object
        print(f"{self.adata.obs}\n\n" if assay and verbose is True else "")

        # Let Property Setters Run
        self.rna = (self.adata.tables[key_table] if "tables" in dir(
            self.adata) and (
                key_table in self.adata.tables) else self.adata.table) if (
                    isinstance(self.adata, spatialdata.SpatialData)
                    ) else self.adata[self._assay] if (
                        self._assay is not None) else self.adata
        if self._columns["col_cell_type"] in self.rna.obs and not isinstance(
                self.rna.obs[self._columns["col_cell_type"]], pd.Categorical):
            self.rna.obs[self._columns["col_cell_type"]] = self.rna.obs[
                self._columns["col_cell_type"]].astype("category")  # category
        print("\n\n", self.rna, "\n\n", self.rna.var.head(), "\n\n",
              self.rna.obs.head() if verbose is True else "", "\n\n\n\n")

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
        elif isinstance(self.adata, spatialdata.SpatialData):
            if self._keys["key_table"] in self.adata.tables and (
                    self.adata.tables[self._keys["key_table"]] is not None):
                del self.adata.tables[self._keys["key_table"]]
            self.adata.tables[self._keys["key_table"]] = value
            self._rna = self.adata.tables[self._keys["key_table"]]
        else:
            if self._assay:
                self.adata[self._assay] = value
            else:
                self.adata = value
            self._rna = self.adata[self._assay] if self._assay else self.adata

    def __repr__(self) -> str:
        """Represent object."""
        repped = "\n\n\n"
        if "_library_id" in dir(self):
            repped += self._library_id + "\n\n"
        repped += "\n\n".join(["\n".join([
            f"{k} = None" if x[k] is None else f"{k} = \"{x[k]}\""
            for k in x]) for x in [self._columns, self._keys]])
        return repped

    def write(self, out_file, **kwargs):
        """Write object."""
        print(f"\n\n***** Writing object to {out_file}...\n\n")
        if os.path.splitext(out_file)[1] == ".h5ad":
            adata = self.rna.copy()
            try:
                adata.write_h5ad(out_file, **kwargs)
            except TypeError:
                _ = adata.uns.pop("markers")  # may not be able to write
                adata.write_h5ad(out_file, **kwargs)
        elif os.path.splitext(out_file)[1] == ".h5mu":
            self.adata.write(out_file, **kwargs)
        else:
            raise ValueError("File extension must be .h5ad or .h5mu")

    def write_clusters(self, out_directory, col_cell_type="leiden",
                       file_prefix=True, n_top=True,
                       p_threshold=None, overwrite=False, **kwargs):
        """Write clusters (and, if `n_top` != False, markers)."""
        key = kwargs.pop("key_added", f"rank_genes_groups_{col_cell_type}")
        pre = "" if file_prefix is None else f"{file_prefix}_"
        outs = [out_directory, os.path.join(out_directory, n_top) if (
            isinstance(n_top, str)) else out_directory]  # n_top=subdirectory?
        file_grp, file_mks = [os.path.join(
            outs[i], f"{pre}{col_cell_type}{s}.csv") for i, s in enumerate(
                ["", "_markers"])]  # file names
        if overwrite is False:
            for x in [file_grp, file_mks]:
                if x is not None and os.path.exists(x):
                    raise ValueError(f"File {x} already exists.")
        (self.rna.obs.set_index("cell_id") if (
            "cell_id" in self.rna.obs) and isinstance(
                self, cr.Spatial) else self.rna.obs)[
                    col_cell_type].to_frame("group").to_csv(file_grp)  # write
        if n_top is not False and key in self.rna.uns:  # if markers available
            mks = cr.ax.make_marker_genes_df(
                self.rna, col_cell_type, key_added=key)  # marker genes df
            if isinstance(n_top, (int, float)) and n_top is not True:
                if any(mks.groupby(col_cell_type).apply(len) < n_top):
                    warn(f"At least 1 cluster {col_cell_type} has < `n_top` "
                         f"({n_top}) markers stored in `.uns['markers']`.")
                mks = mks.groupby(col_cell_type).apply(lambda x: x.iloc[:min(
                    x.shape[0], n_top)]).reset_index()  # n_top DEGs per type
            mks.to_csv(file_mks)  # write markers
        print(f"Markers File: {file_mks}\nClusters File: {file_grp}\n\n")

    def load(self, file_path, file_path_markers=None,
             method_cluster="leiden", **kwargs):
        """Load AnnData."""
        print(f"\n\n***** Writing object from {file_path}...\n\n")
        if os.path.splitext(file_path)[1] == ".h5ad":
            self.rna = sc.read_h5ad(file_path, **kwargs)
        elif os.path.splitext(file_path)[1] == ".h5mu":
            self.adata = muon.read(file_path, **kwargs)
        else:
            raise ValueError("File extension must be .h5ad or .h5mu")
        if file_path_markers:
            if "markers" not in self.rna.uns:
                self.rna.uns["markers"] = {}
            self.rna.uns["markers"][method_cluster] = pd.read_csv(
                file_path_markers).set_index([method_cluster, "names"])

    def get_layer(self, layer=None, subset=None, inplace=False):
        """Get layer (and optionally, subset)."""
        adata = self.rna.copy() if inplace is False else self.rna
        if layer:
            layer = layer if layer in adata.layers else self._layers[layer]
            adata.X = adata.layers[layer].copy()
        if subset not in [None, False]:
            adata = adata[subset]
        return adata

    def get_variables(self, variables=None):
        """Get `.obs` & `.var` variables/intersection with list."""
        variables = cr.tl.to_list(variables)
        valids = list(self.rna.var_names) + list(self.rna.obs.columns)
        variables = [var for var in variables if var in valids]
        return variables

    def print(self):
        """Print information."""
        print(self.rna.obs.head(), "\n\n")
        print(self.adata, "\n\n")
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)
        if "highly_variable" in self.rna.var:
            print(f"\n\n{'=' * 80}\nHighly Variable Genes\n{'=' * 80}\n\n")
            print(self.rna.var.highly_variable.value_counts().to_markdown())
        print("\n\n\n")

    def describe(self, group_by=None, plot=False):
        """Describe data."""
        desc, figs = {}, {}
        gbp = [self._columns["col_cell_type"]]
        if group_by:
            gbp += [group_by]
        print("\n\n\n", self.adata.obs.describe().round(2), "\n\n\n")
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
        self.info["descriptives"].update(desc)
        return figs

    def print_markers(self, key_cluster, assign=None,
                      col_cell_type=None, col_annotation=None,
                      layer="counts", count_threshold=1,
                      p_threshold=1, lfc_threshold=None,
                      n_top_genes=20, key_comparison=None,
                      show=False, print_threshold=64, **kwargs):
        """
        Print markers for a cluster.

        If `assign` is specified:
        Print frequencies at which a cluster expresses at least
        <count_threshold> transcripts of genes in its top marker list,
        sorted by cell type annotations linked to those genes,
        provided a dataframe (`assign`) with genes as the index and
        linked annotations in <col_annotation> (1st if unspecified).
        If `n_top_genes` is a list of genes, will use those instead of
        top markers. If `key_comparison` is specified, will return
        in the 'Percent_Total' column of the 2nd element of the output
        percentages of total cells in that comparison group
        and `key_cluster` rather than of all total cells.
        """
        c_t = self._columns["col_cell_type"] if (
            col_cell_type is None) else col_cell_type
        kmk = kwargs.pop("key_added", f"rank_genes_groups_{c_t}")  # .uns key
        ann = (self.get_layer(layer=layer, inplace=False) if (
            layer is not None) else self.rna).copy()
        mks = cr.ax.make_marker_genes_df(ann, c_t, key_added=kmk)  # DEGs
        mks = mks[mks.pvals_adj <= p_threshold]  # filter by p-value
        if lfc_threshold is not None:
            mks = mks[mks.logfoldchanges >= lfc_threshold]  # filter by LFC
        if assign is not None:
            percs_exp, percs, n_exp, genes, msg = cr.ax.print_marker_info(
                ann, key_cluster, assign=assign, key_added=kmk,
                col_cell_type=c_t, col_annotation=col_annotation,
                layer=layer, count_threshold=count_threshold,
                p_threshold=p_threshold, lfc_threshold=lfc_threshold,
                n_top_genes=n_top_genes, key_comparison=key_comparison,
                show=show, print_threshold=print_threshold,
                marker_genes_df=mks, **kwargs)
            return mks, percs_exp, n_exp, genes, msg
        return mks.loc[key_cluster]

    def map(self, gene=None, col_cell_type=True, **kwargs):
        """Plot GEX &/or cell type(s) on UMAP."""
        if col_cell_type is True:
            col_cell_type = self._columns["col_cell_type"]
        col_cell_type, gene = [[] if x is None else [x] if isinstance(
            x, str) else list(x) for x in [col_cell_type, gene]]
        fig = sc.pl.umap(self.rna, color=gene + col_cell_type, **{
            "legend_loc": "on data", "legend_fontweight": "medium", **kwargs})
        return fig

    def plot(self, genes=None, kind="all", genes_highlight=None, subset=None,
             group=None, layer=None, kws_qc=False, marker_genes_dict=None,
             kws_umap=None, kws_heat=None, kws_violin=None, kws_dot=None,
             kws_matrix=None, cell_types_circle=None, **kwargs):
        """Create a variety of plots."""
        figs = {}
        adata = (self.rna if subset is None else self.rna[subset]).copy()
        if genes is None and marker_genes_dict:  # marker_genes_dict -> genes
            genes = pd.unique(functools.reduce(lambda i, j: i + j, [
                marker_genes_dict[k] for k in marker_genes_dict]))
        if group is None:  # if unspecified grouping variable...
            group = self._columns["col_cell_type"]  # default cell type column
        if kind == "all":
            kind = ["all", "umap"]
        umap = "umap" in kind or (isinstance(kind, str) and kind == "all")
        if "umap" in kind:  # remove "umap" from "kind" since stored in `umap`
            kind = list(set(kind).difference(set(["umap"])))
        if len(kind) == 1:
            kind = kind[0]  # in case "kind" is length 1 after removing umap
        genes_highlight = cr.tl.to_list(genes_highlight)  # list or None
        cell_types_circle = cr.tl.to_list(cell_types_circle)
        cgs = self._columns["col_gene_symbols"] if self._columns[
            "col_gene_symbols"] != self.rna.var.index.names[0] else None
        kw_def = {
            "col_cell_type": group, "legend_loc": "on data",
            "col_gene_symbols": cgs, "cell_types_circle": cell_types_circle}
        kws_umap = cr.tl.merge(kw_def, kws_umap)  # merge default, specified

        # Pre-Processing/QC
        if kws_qc:
            if kws_qc is True:
                kws_qc = {"hue": self._columns["col_sample_id"]}
            cr.pp.perform_qc(adata, layer=self._layers["counts"], **kws_qc)

        # Gene Expression
        if "categories_order" in kwargs:  # merge category kws; subset if need
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
        figs["gex"] = cr.pl.plot_gex(
            adata, col_cell_type=group, genes=genes, kind=kind, layer=layer,
            col_gene_symbols=cgs, marker_genes_dict=marker_genes_dict,
            kws_violin=kws_violin, kws_heat=kws_heat, kws_matrix=kws_matrix,
            kws_dot=kws_dot)  # GEX

        # UMAP
        if umap is True:
            if "X_umap" in self.rna.obsm or group in self.rna.obs.columns:
                figs["umap"] = cr.pl.plot_umap(adata, genes=genes, **kws_umap)
            else:
                print("\n<<< UMAP NOT AVAILABLE. RUN `.cluster()`.>>>")
        return figs

    def plot_umap(self, color=None, group=None, cmap="magma",
                  multi=False, plot_clusters=True, **kwargs):
        """Plot UMAP."""
        if color is None:  # if color-coding column unspecified...
            color = self._columns["col_cell_type"]  # ...color by cell type
        if plot_clusters is True:
            color = list(pd.unique(list([color] if isinstance(
                color, str) else color) + [self._columns["col_cell_type"]]))
        color_original = color.copy()
        for c in color_original:
            if c not in self.rna.obs and c not in self.rna.var_names:
                warn(f"'{c}' not found in `.obs` or `.var`.")
                color.remove(c)
        kws = {"frameon": False, "use_raw": False, "cmap": cmap, **kwargs}
        if group:  # if separating by group, return this plot
            return cr.pl.plot_umap_split(self.rna, group, color=color, **kws)
        if multi is True and not isinstance(color, str) and color[
                0] in self.rna.var_names:
            fig = cr.pl.plot_umap_multi(self.rna, color, **kws)  # multi-gene
        else:  # ...or single UMAP embedding (categorical or continous)
            figsize = kws.pop("figsize", (10, 10))
            with plt.rc_context({"figure.figsize": figsize}):
                fig = sc.pl.umap(self.rna, color=color, **kws)  # UMAP
        return fig

    def plot_compare(self, genes, col_condition=None, col_cell_type=None,
                     layer=None, subset=None, use_raw=False, **kwargs):
        """
        Plot gene expression across cell types and
        condition(s) (string for 1 condition or list for 2, or None
        to default to `self._columns["col_condition"]`).
        """
        ann = self.get_layer(layer=layer, inplace=False)
        if subset is True:
            ann = ann[subset]
        fig = {}
        genes = self.get_variables(variables=genes)
        con, cct = [x[1] if x[1] else self._columns[x[0]]
                    for x in zip(["col_condition", "col_cell_type"], [
                        col_condition, col_cell_type])]  # specs v. default
        con, col = [con, None] if isinstance(con, str) else con  # 1 or 2?
        fig["categorical"] = cr.pl.plot_cat_split(
            ann, con, col_cell_type=cct, genes=genes,
            **{"columns": col, **kwargs})  # plot by groups
        cgs = self._columns["col_gene_symbols"] if ann.var.index.values[
            0] not in ann.var_names else None  # genes=columns instead of ix?
        for c in [col, con]:
            if c:
                ann.obs = ann.obs.astype({c: "category"})
                sc.pl.tracksplot(
                    ann, genes, c, dendrogram=False, gene_symbols=cgs,
                    use_raw=use_raw, figsize=[
                        len(ann.obs[c].unique()) * 5, len(genes)])
                for g in ann.obs[cct].unique():  # iterate cell type
                    ggg = list(set(genes).intersection(ann[
                        ann.obs[cct] == g].var_names))
                    fig[f"track_{c}_{g}"] = sc.pl.tracksplot(
                        ann[ann.obs[cct] == g], ggg, c, show=False,
                        use_raw=use_raw, gene_symbols=cgs, dendrogram=False,
                        figsize=[len(ann.obs[c].unique()) * 5, len(ggg)])
                    fig[f"track_{c}_{g}"]["groupby_ax"].set_xlabel(g)
                    plt.show()
        return fig

    def plot_coex(self, genes, use_raw=False, copy=True, **kwargs):
        """Plot co-expression of a list of genes on a UMAP."""
        ggg = "_".join(genes)  # co-expression variable name (ex: STAT1_ERCC1)
        adata = sc.tl.score_genes(self.rna, genes, score_name=ggg,
                                  use_raw=use_raw, copy=True)  # score co-GEX
        fig = sc.pl.umap(adata, color=ggg, use_raw=use_raw, **kwargs)  # plot
        if copy is False:
            self.rna = adata
        return fig

    def score_coex(self, genes, layer="counts", n_combos=None,
                   threshold=None, inplace=False, suffix=""):
        """
        Score coexpression.

        Notes:

            Score will be binary (GEX for both genes >= `threshold`) if
            `threshold` is not None; otherwise, score = sum of the GEX.

            Will calculate for subsets if `n_combos` specified (e.g.,
            double-positive for each given set of
            gene pair and triplet
            within the list specified if `n_combos` = [2, 3].)
        """
        print(f"\n***Calculating Coexpression (layer={layer})\n\n")
        ann = self.get_layer(layer, inplace=False)
        if n_combos is None:  # if didn't specify size of subsets
            n_combos = len(genes)  # only do combination of all genes
        n_combos = [int(n_combos)] if isinstance(n_combos, (
            int, float)) else [int(x) for x in n_combos]  # ensure format
        for g in genes:  # loop genes: expression data to .obs
            ann.obs.loc[:, g + suffix] = [i[0] for i in ann[:, g].X.toarray(
                )] if (isinstance(ann[:, g].X, scipy.sparse._csr.csr_matrix)
                       ) else ann[:, g].X  # GEX for gene (to array if sparse)
            if threshold is not None:  # convert to binary >= threshold?
                ann.obs.loc[:, g + suffix] = ann.obs[g + suffix] >= threshold
        for n in n_combos:  # loop subset sizes(e.g., double-+, triple-+)
            for p in itertools.combinations(genes, n):  # loop combinations
                ann.obs.loc[:, "_".join(p) + suffix] = ann.obs[list([
                    i + suffix for i in p])].T.sum()
                if threshold:  # would've coded as 1/0 so + if sum=n
                    ann.obs.loc[:, "_".join(p) + suffix] = ann.obs[
                        "_".join(p) + suffix] >= n
        if inplace is True:
            self.rna = ann
        return ann

    def quantify_transcripts(self, genes, col_cell_type=None, layer="counts"):
        """
        Return transcript counts for genes, optionally, by cell type
        (or any categorical column specified in `col_cell_type`).
        """
        genes = [genes] if isinstance(genes, str) else genes  # ensure list
        cct = col_cell_type if col_cell_type else self._columns[
            "col_cell_type"]  # group by cell type
        ann = self.get_layer(layer, inplace=False)  # get specified layer
        tx_cts, tx_cl = [cr.ax.classify_tx(ann, genes=genes, col_cell_type=x,
                                           layer=None) for x in [None, cct]]
        return tx_cts, tx_cl

    def quantify_cells(self, genes, threshold=0, min_genes="all", subset=None,
                       col_cell_type=None, layer="counts", inplace=True):
        """
        Return cells counts positive (above the specified threshold)
        for (combinations of) genes, optionally (if `col_cell_type` is
        not False), broken down by cluster
        (or any categorical column specified in `col_cell_type`).
        Specify "genes" as a list of lists to calculate coexpression.
        If you only want to test one combination, make it a list of
        length 1 (e.g., [["IL6", "IL6ST"]]). Set `min_genes` to
        relax the constraint that all genes in a combination must be
        coexpressed for a cell to be marked as positive.
        """
        adata = self.get_layer("counts", inplace=False)
        cct = None if col_cell_type is False else col_cell_type if (
            col_cell_type is not None) else self._columns["col_cell_type"]
        genes = [genes] if isinstance(genes, str) else genes  # make iterable
        if subset is not None and inplace is True:
            warn("Can't specify subset & inplace=True. Inplace set to False.")
            inplace = False
        if isinstance(genes[0], str):  # just expression, not co-expression
            cts = cr.ax.classify_gex_cells(
                adata, col_cell_type=cct, genes=genes, layer=None,
                threshold=threshold)
        else:  # score co-expression
            cts = cr.ax.classify_coex_cells(
                adata, col_cell_type=cct, genes=None, layer=None,
                threshold=threshold, min_genes="all")
        return cts

    def preprocess(self, assay_protein=None, layer_in="counts", copy=False,
                   kws_scale=True, col_n_obs_raw="n_obs_raw", **kwargs):
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
        adata = self.get_layer(layer=layer_in, inplace=False)
        if col_n_obs_raw is not None:
            adata.obs.loc[:, col_n_obs_raw] = adata.n_obs
        kws = dict(assay_protein=assay_protein, **self._columns, **kwargs)
        kws_scale = {} if isinstance(kws_scale, str) and kws_scale.lower(
            ) == "z" else deepcopy(kws_scale)  # scale kws processing
        if isinstance(kws_scale, dict) and all([x not in kws_scale for x in [
                "max_value", "zero_center"]]):  # if z-scoring GEX ~ control
            znorm_default = {
                "col_reference": self._columns["col_condition"],
                "key_reference": self._keys["key_control"],
                "col_sample_id": self._columns["col_sample_id"]}
            kws_scale = {**znorm_default, **kws_scale}  # merge arguments
        self.info["methods"]["scale"] = kws_scale
        kws.update(dict(kws_scale=kws_scale))
        adata, figs = cr.pp.process_data(adata, **kws)  # preprocess
        # if assay_protein is not None:  # if includes protein assay
        #     ad_p = muon.prot.pp.clr(adata[assay_protein], inplace=False)
        for x in kws.items():
            adata.obs.loc[:, x[0]] = str(x[1])  # store parameters in `.obs`
        if copy is False:
            self.rna = adata
            # if assay_protein is not None:  # if includes protein assay
            #     self.adata[assay_protein] = ad_p
        return adata, figs

    def downsample(self, subset=None, counts_per_cell=None,
                   total_counts=None, replace=False, seed=1618, **kwargs):
        """Downsample counts/`.X` (optionally, of a subset). NOT IN-PLACE."""
        adata = self.rna.copy()
        if kwargs:
            pass
        if subset is not None:
            adata = adata[subset]  # subset if desired
        kws = dict(counts_per_cell=counts_per_cell, total_counts=total_counts,
                   random_state=seed, replace=replace, copy=True)
        return sc.pp.downsample_counts(adata, **kwargs, **kws)  # downsample

    def bulk(self, mode="sum", subset=None, col_cell_type=None,
             col_sample_id=None, layer="counts", copy=True,
             kws_process=True, **kwargs):
        """Create pseudo-bulk data (col_sample_id=False to ignore)."""
        col_cell_type, col_sample_id = [x[1] if x[1] else self._columns[
            x[0]] for x in zip(["col_cell_type", "col_sample_id"], [
                col_cell_type, col_sample_id])]  # use defaults if not given
        if col_sample_id is False:  # False = don't account for sample ID
            col_sample_id = None  # b/c if use None argument -> use ._columns
        adata = self.rna[subset].copy() if subset else self.rna.copy()
        pdata = cr.tl.create_pseudobulk(
            adata, col_cell_type, col_sample_id=col_sample_id,
            mode=mode, layer=layer, kws_process=kws_process, **kwargs
            )  # pseudo-bulk data; process if kws_process=True or dictionary
        if copy is False:
            self.pdata = pdata
        return pdata

    def cluster(self, assay=None, method_cluster="leiden", layer="scaled",
                resolution=1, kws_pca=None, kws_neighbors=None,
                kws_umap=None, kws_cluster=None, genes_subset=None,
                kws_celltypist=None, colors=None, copy=False, use_gpu=False,
                out_file=None, **kwargs):
        """Perform dimensionality reduction and create UMAP."""
        key_added = kwargs.pop("key_added", method_cluster)
        if self._integrated is True and kws_pca is not False:
            warn("Setting kws_pca to False to use Harmony-adjusted PCA!")
            kws_pca = False  # so will use Harmony-adjusted PCA
        cids = [self._columns[x] for x in [
            "col_sample_id", "col_batch", "col_subject"] if self._columns[x]]
        colors = list(pd.unique(cids + (cr.tl.to_list(colors) if colors else [
            ]))) if colors or len(cids) > 0 else None  # UMAPs ~ ID(s)?
        ann = self.get_layer(layer=layer, inplace=False)
        kws = dict(
            method_cluster=method_cluster, kws_pca=kws_pca, kws_umap=kws_umap,
            kws_neighbors=kws_neighbors, kws_cluster=kws_cluster,
            resolution=resolution, use_gpu=use_gpu,
            genes_subset=genes_subset, key_added=key_added)  # arguments
        ann, figs = cr.ax.cluster(ann, assay=assay, colors=colors,
                                  **kws, **kwargs)  # clustering
        for x in kws.items():  # store specifications in `.uns`
            if "specifications" not in ann.uns:
                ann.uns["specifications"] = {"clustering": {}}
            ann.uns["specifications"]["clustering"][key_added] = x[1]
        if copy is False:
            self.info["methods"]["clustering"] = method_cluster
            self.rna = ann
            if out_file:  # write to file if specified
                self.write(out_file)  # write .rna to h5ad/h5mu
        return ann

    def subcluster(self, restrict_to=None, layer="scaled", resolution=1,
                   method_cluster="leiden", key_added=None,
                   use_gpu=False, copy=False, **kwargs):
        """Perform subclustering."""
        # pkg = rsc.tl if use_gpu is True else sc.tl  # Scanpy or Rapids?
        ann = self.get_layer(layer=layer, inplace=False)
        # pkg = sc.tl if use_gpu is False else rsc.tl
        pkg = sc.tl  # Scanpy, because Rapids not yet implemented
        f_x = pkg.leiden if method_cluster == "leiden" else pkg.louvain
        col_sub, key_sub = restrict_to  # unpack cell type column, keys
        if key_added is None:
            key_added = col_sub + "_sub"
        ann.obs.loc[:, key_added] = ann.obs[col_sub].copy()
        if key_sub is None:  # subcluster all if unspecified
            key_sub = list(ann.obs[col_sub].unique())
        if not isinstance(key_sub, (np.ndarray, list, tuple)) or (
                not isinstance(key_sub[0], (np.ndarray, list, tuple))):
            key_sub = [[key_sub]] if isinstance(key_sub, str) else [
                key_sub]  # in case only subclustering one
        for x in key_sub:
            kwss = cr.tl.merge(kwargs, {
                "key_added": key_added, "restrict_to": (key_added, x)})
            f_x(ann, resolution=resolution, **kwss)  # sub-cluster
            cr.pl.plot_clustering(ann, key_added, title=", ".join(x))  # plot
        if copy is False:
            self.rna = ann
        return ann

    def annotate_clusters(self, model, mode="best match", layer="log1p",
                          p_threshold=0.5, over_clustering=None,
                          col_annotation="Annotation",  # not for CellTypist
                          min_proportion=0, copy=False, lfc_threshold=None,
                          plot_markers=False, out_file=None, **kwargs):
        """
        Use CellTypist or a marker dictionary file to annotate clusters.

        Specify a CellTypist model for `model` to use CellTypist,
        or a path to a file with the first column as gene symbols and
        the second as annotations (i.e., mapping the markers
        to cell types). You can also provide a
        dictionary keyed by/series indexed by the cell type column
        (specify `col_cell_type` in the arguments), with items/series
        entries as the annotation to replace the cluster names.
        """
        if isinstance(model, (dict, pd.Series)):
            flavor = "map"
        elif model is None:
            flavor = "ToppGene"
        elif isinstance(model, pd.DataFrame) or model not in list(
                celltypist.models.models_description().model):
            flavor = "annotations"
        else:
            flavor = "celltypist"
        adata, res, figs = self.rna.copy(), {}, {}  # starting objects
        re_ix = self._assay is not None and ":" in adata.var.index.values[0]
        adata.X = adata.layers[self._layers[layer]]  # log 1 p layer
        m_c = kwargs.pop("method_cluster", "leiden" if (
            "leiden" in adata.uns.keys()) else "louvain")  # cluster method
        c_t = kwargs.pop("col_cell_type", m_c)
        if re_ix is True:  # rename multi-modal index if <assay>:<gene>
            adata.var = adata.var.rename(dict(zip(adata.var.index, ["".join(
                x.split(":")[1:]) for x in adata.var.index])))
        if flavor.lower() == "map":  # mapping provided (e.g., in dictionary)
            res = dict(model) if isinstance(model, pd.Series) else {**model}
            if "leiden" in c_t or "louvain" in c_t:  # start with Leiden (str)
                adata.obs.loc[:, col_annotation] = adata.obs[c_t].astype(
                    int).astype(str)  # integer Leiden/Louvain => string
                res = dict(zip([str(int(i)) for i in res], [
                    res[i] for i in res]))  # integer Leiden/Louvain => string
            else:  # start with original cell type column
                adata.obs.loc[:, col_annotation] = adata.obs[c_t].copy()
            adata.obs.loc[:, col_annotation] = adata.obs.loc[
                :, col_annotation].replace(res).astype("category")  # annotate
        elif flavor.lower() == "celltypist":  # annotate with CellTypist
            adata, res, figs = cr.ax.perform_celltypist(
                adata, model, majority_voting=True, p_threshold=p_threshold,
                mode=mode, over_clustering=over_clustering, col_cell_type=c_t,
                min_proportion=min_proportion, **kwargs)  # annotate
            col_ann = ["majority_voting", "predicted_labels"]  # for writing
        elif "top" in flavor.lower():  # ToppGene method
            types = self.rna.obs[c_t].unique()
            n_top_genes = kwargs.pop("n_top_genes", 10)  # number of top genes
            nta = kwargs.pop("n_top_annotations", 20)  # number of top genes
            sources = kwargs.pop("sources", None)  # to filter Atlas sources
            mks = cr.ax.make_marker_genes_df(
                self.rna, c_t, key_added=f"rank_genes_groups_{c_t}",
                p_threshold=p_threshold, lfc_threshold=lfc_threshold)  # DEGs
            mks = mks.groupby(c_t).apply(lambda x: x.iloc[:min(x.shape[
                0], n_top_genes)]).reset_index(0, drop=True)  # only N top
            markers = [list(mks.loc[x].index.values) for x in types]
            tgdf = [cr.tl.get_topp_gene(
                g, sources=sources, verbose=False, **kwargs).drop(
                    "Source", axis=1).reset_index(0, drop=True)
                for g in markers]  # get ToppGene results for each cluster
            tgdf = pd.concat([x.iloc[:np.min([x.shape[0], nta])] if x.shape[
                0] > 0 else x for x in tgdf], keys=types)  # concatenate
            return tgdf, mks
        else:  # annotate by marker dictionary
            if col_annotation in adata.obs:
                adata.obs = adata.obs.drop(col_annotation, axis=1)
            key_add = kwargs.pop("key_added", f"rank_genes_groups_{c_t}")
            if key_add not in adata.uns:
                sc.tl.rank_genes_groups(adata, c_t, use_raw=False,
                                        key_added=key_add)  # rank DEGs
            _, res = cr.ax.annotate_by_markers(
                adata, model, col_cell_type=c_t, key_added=key_add,
                col_new=col_annotation, **kwargs)  # annotate
            col_ann = [col_annotation]  # to write results (optional)
            print("\n\n\n", res, "\n\n\n")
            annots = dict(zip(res.index, list(res[col_annotation])))
            adata.obs.loc[:, col_annotation] = adata.obs[c_t].replace(
                annots).astype(str)
            if plot_markers not in [None, False]:
                figs["markers"] = cr.pl.plot_markers(
                    adata, n_genes=5, rename=annots)  # replot markers
        if re_ix is True:  # back to original index if needed
            adata.var = adata.var.reset_index().set_index(
                self.rna.var.index.names[0])
        if copy is False:  # assign if performing inplace
            print(adata, "\n\n")
            self.rna = adata
            try:  # plot
                if flavor != "celltypist":  # annotate with CellTypist
                    figs["umap"] = self.plot_umap(color=col_annotation)
                    if "plot_spatial" in dir(self):
                        figs["spatial"] = self.plot_spatial(
                            color=col_annotation)
            except Exception as err:
                print(f"\n\n\n{err}\n\n\nPlotting annotations failed!\n\n")
            self.results[flavor] = res
        if out_file:  # write to file if specified
            self.write(out_file)  # write .adata or .rna, based on extension
            obs = adata.obs.set_index("cell_id") if (
                "cell_id" in adata.obs) else adata.obs  # in case Xenium
            for x in [m_c] + col_ann:  # write clusters & annotations
                obs[x].to_frame("group").to_csv(os.path.splitext(
                    out_file)[0] + f"_{x}.csv")
        return adata, [res, figs]

    def find_markers(self, assay=None, n_genes=20, layer="log1p", copy=False,
                     method="wilcoxon", key_reference="rest", kws_plot=True,
                     col_cell_type=None, use_raw=False,
                     out_file=None, **kwargs):
        """Find gene markers for clusters/cell types."""
        if assay is None:
            assay = self._assay
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        key = kwargs.pop("key_added", f"rank_genes_groups_{col_cell_type}")
        adata = self.rna.copy() if copy is True else self.rna  # copy?
        if col_cell_type is None:  # if cell type column not specified...
            if "clustering" in self.info["methods"]:  # use leiden/louvain #s
                col_cell_type = self.info["methods"]["clustering"]
            else:  # otherwise, try getting from _columns attribute
                col_cell_type = self._columns["col_cell_type"]
        n_clus = adata.obs[col_cell_type].value_counts() > 2
        adata = adata if all(n_clus) else adata[adata.obs[
            col_cell_type].isin(n_clus[n_clus > 1].index.values)].copy()
        marks, figs_m = cr.ax.find_marker_genes(
            adata, assay=assay, method=method, n_genes=n_genes, key_added=key,
            layer=layer, key_reference=key_reference, kws_plot=kws_plot,
            col_cell_type=col_cell_type, use_raw=use_raw, **kwargs)  # markers
        marks = marks.join(adata.obs[col_cell_type].value_counts(
            ).to_frame("n_cells"))  # add per-cluster cell counts
        gex = cr.ax.classify_gex_cells(
            adata, col_cell_type=col_cell_type, layer=self._layers["counts"],
            genes=marks.reset_index()[marks.index.names[1]].unique(
                )).reorder_levels([col_cell_type, "Gene"]).rename_axis(
                    marks.index.names)  # % of cells + for each marker
        marks = marks.join(gex.loc[marks.index.intersection(gex.index)][
            "Percent"].to_frame("cells_gene_pos_percent"))
        if copy is False:
            if all(n_clus) is False:
                warn("Had to run find_markers on subset of adata because some"
                     " clusters had N < 3. Cannot update adata attribute.")
            else:
                self.rna = adata
        marks.groupby(col_cell_type).apply(lambda x: print(x.head(3)))
        if copy is False:
            if "markers" not in self.rna.uns:
                self.rna.uns["markers"] = {}
            self.rna.uns["markers"].update({col_cell_type: marks})
        if out_file:
            print(f"\n\nWriting markers to {out_file}.\n\n")
            marks.to_csv(out_file)
        return marks, figs_m

    def run_composition_analysis(self, assay=None, layer=None,
                                 col_list_lineage_tree=None, covariates=None,
                                 reference_cell_type="automatic",
                                 analysis_type="cell_level", est_fdr=0.05,
                                 generate_sample_level=True, plot=True,
                                 copy=False, **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        col_condition, col_cell_type = [kwargs.pop(x, self._columns[
            x]) for x in ["col_condition", "col_cell_type"]]  # extract kwargs
        output = cr.ax.analyze_composition(
            self.rna, col_condition,  col_cell_type,
            assay=assay, layer=layer, reference_cell_type=reference_cell_type,
            analysis_type=analysis_type, plot=plot, out_file=None,
            generate_sample_level=generate_sample_level,
            col_list_lineage_tree=col_list_lineage_tree,  # only for TASCCoda
            covariates=covariates, est_fdr=est_fdr, copy=copy,
            key_reference_cell_type="automatic", **kwargs)
        if copy is False:
            self.results["composition"] = output
        return output

    def run_dialogue(self, n_programs=3, layer="log1p", col_cell_type=None,
                     col_condition=None, col_confounder=None,
                     cmap="coolwarm", vcenter=0, **kws_plot):
        """Analyze <`n_programs`> multicellular programs."""
        col_cell_type, col_condition = [x[1] if x[1] else self._columns[
            x[0]] for x in zip(["col_cell_type", "col_condition"],
                               [col_cell_type, col_condition])]  # defaults
        col_sample_id = kws_plot.pop("col_sample_id", self._columns[
            "col_sample_id"])
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if col_condition is None:
            col_condition = self._columns["col_condition"]
        figsize = (20, 8)  # per facet for violin plots
        adata, figs = self.get_layer(layer=layer, inplace=False), {}
        col = self._columns["col_perturbed" if (
            "col_perturbed" in self._columns) else "col_condition"]
        print(col_cell_type, col)
        kws = dict(n_counts_key=self._columns["col_num_umis"]) if (
            self._columns["col_num_umis"]) else {}
        d_l = pt.tl.Dialogue(
            sample_id=col, n_mpcs=n_programs, celltype_key=col_cell_type,
            **kws)  # run Dialogue
        pdata, mcps, w_s, ct_subs = d_l.calculate_multifactor_PMD(
            adata, normalize=True)
        mcp_cols = list(set(pdata.obs.columns).difference(adata.obs.columns))
        cols = cr.pl.square_grid(len(mcp_cols) + 2)[1]
        figs["umap"] = sc.pl.umap(
            pdata, color=mcp_cols + [col, col_cell_type],
            ncols=cols, cmap=cmap, vcenter=vcenter, **kws_plot)  # UMAP MCP

        # Correct for Confounding Variable?
        if col_confounder:  # correct for a confounding variable?
            try:
                res, p_n = d_l.multilevel_modeling(
                    ct_subs=ct_subs, mcp_scores=mcps, ws_dict=w_s,
                    confounder=col_confounder)
                self.results[f"dialogue_confound_{col_confounder}"] = res, p_n
            except Exception:
                print(traceback.format_exc(), "\n\nIssue w/ Pertpy Dialogue. "
                      "Can't perform confound correction.")

        # Plot MCP ~ Condition (optional)
        if col_condition is not None:
            dff = pd.concat([pd.concat([sc.get.obs_df(ct_subs[q], [
                col_cell_type, m, col_condition]).rename({m: "MCP"}, axis=1)
                            for m in mcp_cols], keys=mcp_cols, names=[
                                "Program"]) for q in ct_subs]).reset_index(0)
            col_wrap = cr.pl.square_grid(len(dff[col_cell_type].unique()))[0]
            figs["conditions"] = sns.catplot(
                dff, x=col_cell_type, y="MCP", col="Program",  kind="violin",
                hue=col_condition, col_wrap=col_wrap, split=col_condition,
                aspect=figsize[0] / figsize[1], height=figsize[1])  # plot
            for x in mcp_cols:
                figs[f"conditions_pair_{x}"] = d_l.plot_pairplot(
                    adata, celltype_key=col_cell_type, color=col_condition,
                    mcp=x, sample_id=col_sample_id)
        self.results["dialogue"] = pdata, mcps, w_s, ct_subs
        self.figures["dialogue"] = figs
        return figs

    def run_gsea(self, key_condition=None, mode="pertpy", library_blitz=None,
                 layer="log1p", absolute=False, direction="both", copy=False,
                 pseudobulk=True, filter_by_highly_variable=True, **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        cct, csid, ccond = [kwargs.pop(x, self._columns[x]) for x in [
            "col_cell_type", "col_sample_id", "col_condition"]]
        if mode == "pertpy":
            adata = self.get_layer(layer=layer, inplace=False)
            output = cr.ax.perform_gsea_pt(
                adata, ccond, key_condition=key_condition, layer=None,
                correction="benjamini-hochberg", absolute=absolute,
                library_blitz=library_blitz, **kwargs)
            if copy is False:
                adata, self.results["gsea"], self.figures["gsea"] = output
                if adata.n_obs != self.rna.n_obs:
                    warn("Cannot save GSEA results as anndata object",
                         "because Anndata was subsetted during GSEA.")
                else:  # store ranked genes in different RGG key; assign adata
                    self.rna = adata
        else:
            adata = self.get_layer(layer=layer, inplace=False)
            if pseudobulk not in [None, False]:  # use if anndata or create
                if isinstance(pseudobulk, bool) and pseudobulk is True:
                    pseudobulk = dict(col_cell_type=cct, col_sample_id=csid)
                data = pseudobulk.copy() if isinstance(
                    pseudobulk, anndata.AnnData) else self.bulk(**pseudobulk)
            output = cr.ax.perform_gsea(
                data, adata_sc=adata, col_sample_id=csid, col_cell_type=cct,
                layer=None, col_condition=ccond, key_condition=key_condition,
                filter_by_highly_variable=filter_by_highly_variable,
                copy=False, pseudobulk=pseudobulk, **kwargs)  # GSEA
            if copy is False:
                if pseudobulk is True:
                    self.pdata = output[0]
                else:
                    self.rna = output[0]
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
        if col_covariates is None and self._columns["col_condition"] is None:
            con = "`._columns['col_condition']` is None"
            raise ValueError(f"Specify `col_covariates` if {con}.")
        if col_covariates is None:
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

    def calculate_receptor_ligand(self, method="liana", subset=None,
                                  layer="log1p", col_cell_type=None,
                                  col_condition=None, col_subject=True,
                                  col_sample_id=None, key_sources=None,
                                  key_targets=None, resource="CellPhoneDB",
                                  top_n=20, min_prop=0, min_count=0,
                                  min_total_count=0, remove_ns=False,
                                  p_threshold=0.01, n_jobs=None, copy=False,
                                  cmap="magma", kws_plot=None, n_perms=10,
                                  figsize=None, **kwargs):
        """Find receptor-ligand interactions (&, optionally DEA)."""
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        figs = {}
        col_kws = dict(zip(["col_subject", "col_sample_id", "col_condition"
                            ], [col_subject, col_sample_id, col_condition]))
        col_subject, col_sample_id, col_condition = [
            None if col_kws[x] is False else self._columns[x] if col_kws[
                x] in [None, True] else col_kws[x] for x in col_kws
            ]  # ignore ID/condition if argue False; else use default if need
        if isinstance(self.adata, spatialdata.SpatialData):
            adata = self.rna
        else:
            adata = self.get_layer(layer=layer, inplace=False, subset=subset)

        # Differential Expression Analysis (DEA), Optionally
        if col_condition is not None:
            kws_deseq2 = kwargs.pop("kws_deseq2", None)
            if kws_deseq2 is None:
                kws_deseq2 = {}
            figsz_de = kws_deseq2.pop("figsize", np.max(
                figsize) if figsize else None)  # figure size for DEA volcano
            key_control, key_treatment = [
                kwargs[x] if x in kwargs else self._keys[x]
                for x in ["key_control", "key_treatment"]]  # condition labels
            pseudo = cr.tl.create_pseudobulk(
                adata.copy(), col_cell_type, col_sample_id=col_sample_id,
                layer=self._layers["counts"], mode="sum",
                kws_process=True)  # create pseudo-bulk data
            if copy is False:
                self.pdata = pseudo
            cgs = pseudo.var.index.names[0]  # gene symbols column/index name
            res_dea = cr.ax.calculate_dea_deseq2(
                pseudo, col_cell_type, col_condition, key_control,
                key_treatment, top_n=top_n, figsize=figsz_de,
                layer_counts=self._layers["counts"],
                col_subject=col_subject, min_prop=min_prop,
                min_count=min_count, col_gene_symbols=cgs,
                min_total_count=min_total_count, **kws_deseq2)  # run DEA
        else:
            res_dea = None, None, None

        # Ligand-Receptor Analysis
        res, adata, figs["l_r"] = cr.ax.analyze_receptor_ligand(
            adata, col_condition=col_condition, col_subject=col_subject,
            col_sample_id=col_sample_id, resource=resource, n_jobs=n_jobs,
            method=method, layer=layer, layer_counts=self._layers["counts"],
            key_sources=key_sources, key_targets=key_targets, copy=False,
            col_cell_type=col_cell_type, top_n=top_n, remove_ns=remove_ns,
            cmap=cmap, p_threshold=p_threshold, figsize=figsize,
            min_prop=min_prop, min_total_count=min_total_count,
            min_count=min_count, n_perms=n_perms, kws_plot=kws_plot,
            # dea_df=res_dea[1], **kwargs)  # run receptor-ligand
            dea_df=None, plot=res_dea[1] is None, **kwargs)  # receptor-ligand
        res["dea_results"], res["dea_df"], figs["dea"] = res_dea
        if res_dea[1] is not None:
            res["lr_dea_res"] = liana.mu.df_to_lr(
                adata[adata.obs[col_condition] == key_treatment].copy(),
                res_dea[1], col_cell_type, stat_keys=[
                    "stat", "pvalue", "padj"], use_raw=False, layer=layer,
                verbose=True, complex_col="stat", expr_prop=min_prop,
                return_all_lrs=True, resource_name="consensus").sort_values(
                    "interaction_stat", ascending=False)
            kws = cr.tl.merge(
                dict(cmap=cmap, p_threshold=p_threshold, top_n=top_n,
                     figsize=figsize, key_sources=key_sources,
                     key_targets=key_targets), kws_plot)
            figs["lr"] = cr.pl.plot_receptor_ligand(
                adata=adata, lr_dea_res=res["lr_dea_res"], **kws)  # plot
        if copy is False:
            self.rna = adata
            self.results["receptor_ligand"] = res
            self.figures["receptor_ligand"] = figs
            self.results["receptor_ligand_info"] = {
                "subset": subset, "col_condition": col_condition,
                "col_cell_type": col_cell_type}
        return adata, res, figs

    def plot_receptor_ligand(self, key_sources=None, key_targets=None,
                             title=None, out_dir=None, top_n=20,
                             figsize_dea=None, swap_axes=False, **kwargs):
        """Plot previously-run receptorout_dir-ligand analyses."""
        subset = self.results["receptor_ligand_info"]["subset"]
        cct = self.results["receptor_ligand_info"]["col_cell_type"]
        res = self.results["receptor_ligand"].copy()
        if figsize_dea is None:
            figsize_dea = (20, 20)
        kws_dea = dict(sign_thr=0.05, lFCs_thr=0.5,
                       sign_limit=None,  lFCs_limit=None, dpi=200)
        for x in kws_dea:
            kws_dea[x] = kwargs.pop(x, kws_dea[x])
        figs = cr.pl.plot_receptor_ligand(
            adata=(self.adata[subset] if subset else self.adata).copy(),
            lr_dea_res=res["lr_dea_res"], top_n=top_n, swap_axes=swap_axes,
            key_sources=key_sources, key_targets=key_targets, **kwargs)  # p
        if res["dea_df"] is not None:
            from decoupler import plot_volcano_df  # noqa: E402
            dea = res["dea_df"].reset_index()
            p_dims = cr.pl.square_grid(len(dea))
            fig, axs = plt.subplots(p_dims[0], p_dims[1], figsize=figsize_dea)
            for i, x in enumerate(dea[cct].unique()):
                try:
                    plot_volcano_df(
                        dea[dea[cct] == x], x="log2FoldChange", y="padj",
                        **kws_dea, ax=axs.ravel()[i])  # plot
                    axs.ravel()[i].set_title(x)
                except Exception:
                    print(traceback.format_exc())
            fig.tight_layout()
            figs["dea"] = fig
        return figs

    def calculate_causal_network(self, key_source, key_target,
                                 top_n=10, **kwargs):
        """Perform causal network analysis with Corneto."""
        dea_df = self.results["receptor_ligand"]["dea_df"].copy()
        dea_df = self.results["receptor_ligand"]["dea_df"].copy()
        ccd, cct, sub = [self.results["receptor_ligand_info"][x] for x in [
            "col_condition", "col_cell_type", "subset"]]  # columns
        csid = kwargs.pop("col_sample_id", self._columns["col_sample_id"])
        if csid is False:
            csid = None  # ignore sample ID if set to False in kwargs
        key_control, key_treatment = [kwargs.pop(x, self._keys[x]) for x in [
            "key_control", "key_treatment"]]  # key labels
        df_res, problem, fig = cr.ax.analyze_causal_network(
            self.adata[sub] if sub else self.adata, ccd, key_control,
            key_treatment, cct, key_source, key_target, dea_df=dea_df,
            col_sample_id=csid, layer="log1p", solver="scipy", expr_prop=0,
            min_n_ulm=5, node_cutoff=0.1, max_penalty=1, min_penalty=0.01,
            edge_penalty=0.01, max_seconds=60*3, top_n=top_n, verbose=False)
        # fig = corneto.methods.carnival.visualize_network(out[0])
        # fig.view()
        return df_res, problem, fig

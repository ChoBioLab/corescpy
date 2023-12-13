#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""

import os
import squidpy as sq
import scanpy as sc
import matplotlib.pyplot as plt
import crispr as cr
from crispr.class_sc import Omics
import pandas as pd
import numpy as np

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Spatial(Omics):
    """A class for CRISPR analysis and visualization."""
    
    _columns_created = dict(guide_percent="Percent of Cell Guides")

    def __init__(self, file_path, file_path_spatial=None, 
                 visium=False, **kwargs):
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
            file_path_spatial (str, optional): Path to spatial 
                information csv file, if needed.
            visium (bool or dict, optional): File path provided is to 
                10x Visium data? Provide as a dictionary of keyword
                arguments to pass to `scanpy.read_visium()` or simply
                as True to use default arguments. The default is False
                (assumes 10x Xenium data format and 
                `file_path_spatial` provided).
            kwargs (dict, optional): Keyword arguments to pass to the 
                Omics class initialization method.
        """
        print("\n\n<<< INITIALIZING SPATIAL CLASS OBJECT >>>\n")
        if isinstance(visium, dict) or visium is True:
            if not isinstance(visium, dict):
                visium = {}  # unpack file path & arguments
            file_path = sq.read.visium(file_path, **visium)  # read Visium
        super().__init__(file_path, **kwargs)  # Omics initialization
        self._assay_spatial = "spatial"
        if file_path_spatial:  # if need to read in additional spatial data
            self.adata.obs = pd.read_csv(file_path_spatial).set_index(
                self.adata.obs_names).copy()  # read in spatial information
            self.adata.obsm["spatial"] = self.adata.obs[
                ["x_centroid", "y_centroid"]].copy().to_numpy()  # coordinates
        print("\n\n")
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)
        print("\n\n", self.rna)
        if "raw" not in dir(self.rna):
            self.rna.raw = self.rna.copy()  # freeze normalized, filtered data
        self._library_id = None
            
    def plot_spatial(self, color, include_umap=True, 
                     col_sample_id=None, library_id=None, 
                     shape="hex", figsize=30, cmap="magma", **kwargs):
        """Create basic spatial plots."""
        figs = {}
        if isinstance(figsize, (int, float)):
            figsize = (figsize, figsize)
        if "wspace" not in kwargs:
            kwargs["wspace"] = 0.4
        if color is None:
            color = self._columns["col_cell_type"]
        if include_umap is True:
            color = list([color] if isinstance(color, str) else color) + [
                self._columns["col_cell_type"]]
            color = list(pd.unique(color))
        if not library_id:
            library_id = self._library_id
        if "title" not in kwargs:
            kwargs.update({"title": color})
        # libid = col_sample_id if col_sample_id not in [
        #     None, False] else self._columns["col_sample_id"]  # library ID
        figs["spatial"] = sq.pl.spatial_scatter(
            self.adata, library_id=library_id, figsize=figsize, shape=shape, 
            color=[color] if isinstance(color, str) else color, 
            cmap=cmap, alt_var=self._columns["col_gene_symbols"] if (
                self._columns["col_gene_symbols"
                              ] != self.rna.var.index.names[0]) else None,
            **kwargs)
        return figs        
    
    def analyze_spatial(
        self, col_cell_type=None, genes=None, layer="log1p", library_id=None,
        figsize_multiplier=1, dpi=100, palette=None,
        kws_receptor_ligand=None, key_source=None, key_targets=None,
        method_autocorr="moran", alpha=0.005, n_perms=100, 
        seed=1618, cmap="magma", copy=False):
        """Analyze spatial (adapted Squidpy tutorial)."""
        figs = {}
        adata = self.rna if copy is False else self.rna.copy()
        layer = self._layers[layer] if layer in self._layers else layer
        adata.X = adata.layers[layer]  # set data layer
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if not library_id:
            library_id = self._library_id
            
        # Connectivity & Centrality + Interaction Matrix
        print("\n<<< CALCULATING CENTRALITY SCORES >>>")
        self.calculate_graph(col_cell_type=col_cell_type, 
                             figsize_multiplier=figsize_multiplier)  # run
        
        # Co-Occurence
        print("\n<<< QUANTIFYING CELL TYPE CO-OCCURRENCE >>>")
        adata, figs["cooccurrence"] = self.find_cooccurrence(
            col_cell_type=col_cell_type, copy=False)
        
        # Neighbors Enrichment Analysis
        print("\n<<< PERFORMING NEIGHBORHOOD ENRICHMENT ANALYSIS >>>")
        figs["enrichment"] = self.calculate_neighborhood(
            col_cell_type=col_cell_type, copy=False, kws_plot=dict(cmap=cmap))
        
        # Spatially-Variable Genes
        if method_autocorr not in [None, False]:
            figs["svgs_autocorrelation"] = self.find_svgs(
                genes=genes, method=method_autocorr, 
                layer=layer, n_perms=n_perms, palette=palette)
            
        # Receptor-Ligand Interaction
        if kws_receptor_ligand is not False:
            if not kws_receptor_ligand:
                kws_receptor_ligand = {}
            res_rl, figs["receptor_ligand"] = self.calculate_receptor_ligand(
                col_cell_type=col_cell_type, n_perms=n_perms, alpha=alpha, 
                dpi=dpi, key_source=None, key_targets=None,
                **kws_receptor_ligand)  # run receptor-ligand analysis
        
        # Distribution Pattern
        figs["distribution_pattern"] = self.calculate_distribution_pattern()
        
        # Output
        if copy is False:
            self.results["spatial"]["receptor_ligand"] = res_rl
            self.figures["spatial"] = figs
            self.rna = adata
            return figs
        else:
            return adata, figs
    
    def calculate_centrality(self, col_cell_type=None, delaunay=True, 
                             coord_type="generic", figsize=None, 
                             palette="coolwarm", shape="hex", size=None, 
                             title=None, kws_plot=None, 
                             copy=False, jobs=None, **kwargs):
        """
        Characterize connectivity, centrality, and interaction matrix.
        """
        # Connectivity & Centrality
        print("\t*** Building connectivity matrix...")
        adata = self.rna.copy() if copy is True else self.rna
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        if size is None:
            size = int(1 if figsize[0] < 25 else figsize[0] / 15)
        kws_plot = {**dict(figsize=figsize, palette=palette, 
                           shape=shape, size=size), 
                    **dict(kws_plot if kws_plot else {})}
        if jobs is None and jobs is not False:
            jobs = os.cpu_count() - 1  # threads for parallel processing
        sq.gr.spatial_neighbors(
            adata, coord_type=coord_type, delaunay=delaunay,
            spatial_key=self._assay_spatial)  # spatial neighbor calculation
        print("\t*** Computing & plotting centrality scores...")
        sq.gr.centrality_scores(adata, cluster_key=col_cell_type, n_jobs=jobs)
        # adata.uns["spatial"][
        #     "library_id"] = col_sample_id if col_sample_id not in [
        #         None, False] else self._columns["col_sample_id"]  # library ID
        sq.pl.centrality_scores(adata, cluster_key=col_cell_type, 
                                figsize=figsize)
        fig = plt.gcf()
        if title:
            fig.suptitle(title)
        print("\t*** Computing interaction matrix...")
        sq.gr.interaction_matrix(adata, col_cell_type, normalized=False)
        if self._library_id is None and len(list(self.adata.uns[
            self._assay_spatial].keys())) == 1:
            print("<<< UPDATING SELF._LIBRARY_ID >>>")
            self._library_id = list(adata.uns[self._assay_spatial].keys())[0]
        return fig
        
    def calculate_neighborhood(self, col_cell_type=None, library_id=None,
                               figsize=None, palette=None, size=None,
                               shape="hex", seed=1618, 
                               kws_plot=None, copy=False):
        """Perform neighborhood enrichment analysis."""
        adata = self.rna.copy() if copy is True else self.rna
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        if size is None:
            size = int(1 if figsize[0] < 25 else figsize[0] / 15)
        kws_plot = {**dict(palette=palette, shape=shape, size=size), 
                    **dict(kws_plot if kws_plot else {})}
        sq.gr.nhood_enrichment(adata, cluster_key=col_cell_type,
                               n_jobs=None,
                               # n_jobs=jobs,  # not working for some reason?
                               seed=seed)  # neighborhood enrichment
        fig, axs = plt.subplots(1, 2, figsize=figsize)  # set up facet figure
        sq.pl.nhood_enrichment(
            adata, cluster_key=col_cell_type, figsize=(8, 8), 
            title="Neighborhood Enrichment", ax=axs[0])  # heatmap (panel 1)
        sq.pl.spatial_scatter(adata, color=col_cell_type,
                              ax=axs[1], **kws_plot)  # scatterplot (panel 2)
        return fig
        
    def find_cooccurrence(self, col_cell_type=None, key_cell_type=None,
                          layer=None, copy=False, jobs=None,
                          figsize=15, palette=None, title=None,
                          kws_plot=None, shape="hex", size=None, **kwargs):
        """
        Find co-occurrence using spatial data. (similar to neighborhood
        enrichment analysis, but uses original spatial coordinates rather
        than connectivity matrix).
        """
        figs, adata = {}, self.rna.copy() if copy is True else self.rna
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        if size is None:
            size = int(1 if figsize[0] < 25 else figsize[0] / 15)
        kws_plot = {**dict(palette=palette, shape=shape, size=size), 
                    **dict(kws_plot if kws_plot else {})}
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if jobs is None and jobs is not False:
            jobs = os.cpu_count() - 1  # threads for parallel processing
        if layer:
            adata.X = adata.layers[self._layers[layer]].X.copy()
        sq.gr.co_occurrence(adata, cluster_key=col_cell_type, n_jobs=jobs, 
                            spatial_key=self._assay_spatial, **kwargs)
        figs["co_occurrence"] = {}
        figs["spatial_scatter"] = sq.pl.spatial_scatter(
            adata, color=col_cell_type, **kws_plot, 
            figsize=figsize, return_ax=True)  # scatter
        if title:
            figs["spatial_scatter"].suptitle(title)
        figs["co_occurrence"] = sq.pl.co_occurrence(
            adata, cluster_key=col_cell_type,
            clusters=key_cell_type, figsize=figsize)
        if title:
            figs["co_occurrence"].suptitle(title)
        return adata, figs
        
    def find_svgs(self, genes=None, method="moran", n_perms=10,
                  library_id=None, layer=None, copy=False,
                  col_cell_type=None, col_sample_id=None, jobs=None, 
                  figsize=15, title=None, kws_plot=None):
        """Find spatially-variable genes."""
        adata = self.rna.copy() if copy else self.rna
        if not library_id:
            library_id = self._library_id
        if jobs is None and jobs is not False:
            jobs = os.cpu_count() - 1  # threads for parallel processing
        if genes is None:
            genes = 10  # plot top 10 variable genes if un-specified
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        # if size is None:
        #     size = int(1 if figsize[0] < 25 else figsize[0] / 15)
        # kws_plot = {**dict(palette=palette, shape=shape, size=size), 
        #             **dict(kws_plot if kws_plot else {})}
        kws_plot = {"cmap": "magma", **dict(kws_plot if kws_plot else {})}
        print(f"\n<<< QUANTIFYING AUTO-CORRELATION (method = {method}) >>>")
        sq.gr.spatial_autocorr(self.rna, mode=method, layer=layer, 
                               n_perms=n_perms, n_jobs=jobs)  # auto-correlate
        if isinstance(genes, int):
            genes = adata.uns["moranI"].head(genes).index.values
        # libid = col_sample_id if col_sample_id not in [
        #     None, False] else self._columns["col_sample_id"]  # library ID
        # sq.pl.spatial_scatter(adata, color=col_cell_type, 
        #                       library_id=library_id, figsize=figsize,
        #                       **kws_plot)  # cluster plot (panel 1)
        sc.pl.spatial(adata, color=genes + [col_cell_type], 
                      library_id=library_id, **kws_plot)  # GEX plot (panel 2)
        fig = plt.gcf()
        if title:
            fig.suptitle(title)
        return fig
        
    def calculate_distribution_pattern(self, col_cell_type=None, mode="L"):
        """
        Use Ripley's statistics to determine whether distributions are 
        random, dispersed, or clustered.
        """
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        sq.gr.ripley(self.rna, cluster_key=col_cell_type, mode=mode)
        fig = sq.pl.ripley(self.rna, cluster_key=col_cell_type, mode=mode)
        return fig
    
    def calculate_receptor_ligand(self, key_source=None, key_targets=None, 
                                  col_cell_type=None, n_perms=10, 
                                  pvalue_threshold=0.05, 
                                  remove_nonsig=True, alpha=None, copy=False, 
                                  kws_plot=None, **kwargs):
        """Calculate receptor-ligand interactions."""
        if alpha is None:
            alpha = pvalue_threshold
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        adata = self.rna.copy() if copy else self.rna
        res = sq.gr.ligrec(
            adata, n_perms=n_perms, cluster_key=col_cell_type,
            transmitter_params={"categories": "ligand"}, 
            receiver_params={"categories": "receptor"},
            interactions_params={'resources': 'CellPhoneDB'}, 
            copy=True, **kwargs)
        fig = sq.pl.ligrec(res, alpha=alpha, 
                           source_groups=key_source, target_groups=key_targets,
                           remove_nonsig_interactions=remove_nonsig,
                           pvalue_threshold=pvalue_threshold, 
                           **{**dict(kws_plot if kws_plot else {})})  # plot 
        return res, fig
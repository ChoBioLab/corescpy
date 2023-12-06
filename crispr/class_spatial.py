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
            
    def plot(self, color, col_sample_id=None, shape="hex", 
             figsize=30, **kwargs):
        """Create basic plots."""
        figs = {}
        if isinstance(figsize, (int, float)):
            figsize = (figsize, figsize)
        if "wspace" not in kwargs:
            kwargs["wspace"] = 0.4
        if color is None:
            color = self._columns["col_cell_type"]
        libid = col_sample_id if col_sample_id not in [
            None, False] else self._columns["col_sample_id"]  # library ID
        figs["spatial"] = sq.pl.spatial_scatter(
            self.adata, library_id=libid, figsize=figsize, shape=shape, 
            color=[color] if isinstance(color, str) else color, **kwargs)
        return figs        
    
    def analyze_spatial(self, col_cell_type=None, genes=None, layer="log1p",
                        figsize_multiplier=1, dpi=100,
                        kws_receptor_ligand=None,
                        key_source=None, key_targets=None,
                        method_autocorr="moran", alpha=0.005, 
                        col_sample_id=None, n_perms=100, seed=1618, copy=False):
        """Analyze spatial (adapted Squidpy tutorial)."""
        figs = {}
        adata = self.rna if copy is False else self.rna.copy()
        layer = self._layers[layer] if layer in self._layers else layer
        adata.X = adata.layers[layer]  # set data layer
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
            
        # Connectivity & Centrality + Interaction Matrix
        print("\n<<< CALCULATING CENTRALITY SCORES >>>")
        self.calculate_graph(col_cell_type=col_cell_type, 
                             figsize_multiplier=figsize_multiplier)  # run
        self.adata.uns[self._assay_spatial]["library_id"] = list(
            self.adata.uns["spatial"].keys())[0]  # store library_id
        
        # Co-Occurence
        print("\n<<< QUANTIFYING CO-OCCURRENCE >>>")
        adata, figs["cooccurrence"] = self.find_cooccurrence(
            col_cell_type=col_cell_type, copy=False)  # cooccurring cell types
        
        # Neighbors Enrichment Analysis
        print("\n<<< PERFORMING NEIGHBORHOOD ENRICHMENT ANALYSIS >>>")
        figs["enrichment"] = self.calculate_neighborhood(
            col_cell_type=col_cell_type, copy=False)  # analyze enrichment
        
        # Spatially-Variable Genes
        if method_autocorr not in [None, False]:
            figs["svgs_autocorrelation"] = self.find_svgs(
                genes=genes, adata=adata, method=method_autocorr, 
                layer=layer, n_perms=n_perms)  # find SVGs
            
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
                             coord_type="generic", figsize_multiplier=1):
        """
        Characterize connectivity, centrality, and interaction matrix.
        """
        # Connectivity & Centrality
        print("\t*** Building connectivity matrix...")
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        jobs = os.cpu_count() - 1  # threads for parallel processing
        sq.gr.spatial_neighbors(
            self.adata, coord_type=coord_type, delaunay=delaunay,
            spatial_key=self._assay_spatial)  # spatial neighbor calculation
        # adata.uns["spatial"][
        #     "library_id"] = col_sample_id if col_sample_id not in [
        #         None, False] else self._columns["col_sample_id"]  # library ID
        print("\t*** Computing centrality scores...")
        sq.gr.centrality_scores(self.adata, 
                                cluster_key=col_cell_type, n_jobs=jobs)
        sq.pl.centrality_scores(
            self.adata, cluster_key=col_cell_type, figsize=tuple(np.array(
                [16, 5]) * figsize_multiplier))  # plot centrality score
        print("\t*** Computing interaction matrix...")
        sq.gr.interaction_matrix(self.adata, col_cell_type, normalized=False)
        
    def calculate_neighborhood(self, col_cell_type=None, figsize_multiplier=1, 
                               shape="hex", seed=1618, **kwargs):
        """Perform neighborhood enrichment analysis."""
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        sq.gr.nhood_enrichment(self.adata, cluster_key=col_cell_type, 
                               n_jobs=None,
                               # n_jobs=jobs,  # not working for some reason?
                               seed=seed)  # neighborhood enrichment
        fig, axs = plt.subplots(1, 2, figsize=np.array(
            [13, 7]) * figsize_multiplier)  # set up facet grid figure
        sq.pl.nhood_enrichment(
            self.adata, cluster_key=col_cell_type, figsize=(8, 8), 
            title="Neighborhood Enrichment", ax=axs[0])  # heatmap (panel 1)
        sq.pl.spatial_scatter(self.adata, color=col_cell_type, shape=shape, 
                              size=2 * figsize_multiplier, ax=axs[1], 
                              figsize=tuple(np.array(
                                  [2, 2]) * figsize_multiplier), **kwargs
                              )  # scatterplot (panel 2)
        return fig
        
    def find_cooccurrence(self, col_cell_type=None, key_cell_type=None,
                          layer=None, figsize=30, copy=False, 
                          kws_plot=None, **kwargs):
        """
        Find co-occurrence using spatial data. (similar to neighborhood
        enrichment analysis, but uses original spatial coordinates rather
        than connectivity matrix).
        """
        figs = {}
        if isinstance(figsize, (int, float)):
            figsize = (figsize, figsize) 
        if kws_plot is None:
            kws_plot = {}
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        jobs = os.cpu_count() - 1  # threads for parallel processing
        adata = self.rna.copy() if copy is True else self.rna
        if layer:
            adata.X = adata.layers[self._layers[layer]].X.copy()
        sq.gr.co_occurrence(adata, cluster_key=col_cell_type, n_jobs=jobs, 
                            spatial_key=self._assay_spatial, **kwargs)
        figs["co_occurrence"] = {}
        figs["spatial_scatter"] = sq.pl.spatial_scatter(
            adata, color=col_cell_type, shape=None, size=2, figsize=figsize)
        try:
            figs["co_occurrence"] = sq.pl.co_occurrence(
                adata, cluster_key=col_cell_type, 
                clusters=key_cell_type, figsize=figsize, **kws_plot)
        except Exception as err:
            figs["co_occurrence"] = err
            print(f"{err}\n{type(err)}\n{err.args}\n\n"
                  "Failed to plot co-occurrence!")
        return adata, figs
        
    def find_svgs(self, genes=None, method="moran", n_perms=10,
                  layer=None, adata=None, figsize=30,
                  col_cell_type=None, col_sample_id=None):
        """Find spatially-variable genes."""
        fig = {}
        if isinstance(figsize, (int, float)):
            figsize = (figsize, figsize) 
        if genes is None:
            genes = 10  # plot top 10 variable genes if un-specified
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if adata is None:
            adata = self.rna
        jobs = os.cpu_count() - 1  # threads for parallel processing
        print(f"\n<<< QUANTIFYING AUTO-CORRELATION (method = {method}) >>>")
        sq.gr.spatial_autocorr(self.rna, mode=method, layer=layer, 
                               n_perms=n_perms, n_jobs=jobs)  # auto-correlation
        if isinstance(genes, int):
            genes = adata.uns["moranI"].head(genes).index.values
        libid = col_sample_id if col_sample_id not in [
            None, False] else self._columns["col_sample_id"]  # library ID
        fig["scatter"] = sq.pl.spatial_scatter(
            adata, library_id=libid, figsize=figsize,
            color=genes, shape=None, size=2, img=False)
        fig["umap"] = sc.pl.spatial(adata, color=genes, library_id=libid)
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
                                  remove_nonsig=True, alpha=0.005, copy=False, 
                                  kws_plot=None, **kwargs):
        """Calculate receptor-ligand interactions."""
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        adata = self.rna.copy()
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
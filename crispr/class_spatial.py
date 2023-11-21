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

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Spatial(Omics):
    """A class for CRISPR analysis and visualization."""
    
    _columns_created = dict(guide_percent="Percent of Cell Guides")

    def __init__(self, file_path, file_path_spatial, **kwargs):
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
            file_path_spatial (str): Path to spatial information csv 
                file. For instance, .
            kwargs (dict, optional): Keyword arguments to pass to the 
                Omics class initialization method.
        """
        print("\n\n<<< INITIALIZING SPATIAL CLASS OBJECT >>>\n")
        super().__init__(file_path, **kwargs)  # Omics initialization
        self._assay_spatial = "spatial"
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
            
    def plot(self, color):
        """Create basic plots."""
        figs = {}
        if color is None:
            color = self._columns["col_cell_type"]
        figs["spatial"] = sq.pl.spatial_scatter(
            self.adata, library_id=self._assay_spatial,
            shape=None, color=[color], wspace=0.4)
        return figs        
    
    def analyze_spatial(self, col_cell_type=None, genes=None, 
                        method_autocorr="moran", alpha=0.005, layer="log1p",
                        col_sample_id=None, n_perms=100, seed=1618, copy=False):
        """Analyze spatial (adapted Squidpy tutorial)."""
        figs = {}
        adata = self.rna if copy is False else self.rna.copy()
        layer = self._layers[layer] if layer in self._layers else layer
        adata.X = adata.layers[layer]  # set data layer
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        jobs = os.cpu_count() - 1  # threads for parallel processing
            
        # Connectivity & Centrality
        print("\n<<< CALCULATING CONNECTIVITY & CENTRALITY >>>")
        print("\t*** Building connectivity matrix...")
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True,
                                spatial_key=self._assay_spatial)
        adata.uns["spatial"][
            "library_id"] = col_sample_id if col_sample_id not in [
                None, False] else self._columns["col_sample_id"]  # library ID
        print("\t*** Computing centrality scores...")
        sq.gr.centrality_scores(adata, cluster_key=col_cell_type, n_jobs=jobs)
        sq.pl.centrality_scores(adata, cluster_key=col_cell_type, 
                                figsize=(16, 5))  # plot centrality scores
        
        # Interaction Matrix
        sq.gr.interaction_matrix(adata, col_cell_type, normalized=False)
        
        # Co-Occurence
        print("\n<<< QUANTIFYING CO-OCCURRENCE>>>")
        adata, figs["cooccurrence"] = self.find_cooccurrence(
            col_cell_type=col_cell_type, copy=False)
        
        # Neighbors Enrichment Analysis
        print("\n<<< PERFORMING NEIGHBORHOOD ENRICHMENT ANALYSIS >>>")
        sq.gr.nhood_enrichment(adata, cluster_key=col_cell_type, 
                               n_jobs=None,
                               # n_jobs=jobs,  # not working for some reason?
                               seed=seed)
        figs["enrichment"], axs = plt.subplots(1, 2, figsize=(13, 7))
        sq.pl.nhood_enrichment(
            adata, cluster_key=col_cell_type, figsize=(8, 8), 
            title="Neighborhood Enrichment", ax=axs[0])  # heatmap (panel 1)
        sq.pl.spatial_scatter(adata, color=col_cell_type, shape=None, 
                                size=2, ax=axs[1])  # scatterplot (panel 2)
        
        # Spatially-Variable Genes
        if method_autocorr not in [None, False]:
            figs["autocorrelation"] = self.find_svgs(
                genes=genes, adata=adata, method=method_autocorr, 
                layer=layer, n_perms=n_perms)
            
        # Receptor-Ligand Interaction
        adata, figs["receptor_ligand"] = self.calculate_receptor_ligand(
            key_source=None, key_targets=None, col_cell_type=col_cell_type, 
            n_perms=n_perms, alpha=alpha)
        
        # Output
        if copy is False:
            self.figures["spatial"] = figs
            self.rna = adata
            return figs
        else:
            return adata, figs
        
        
    def find_cooccurrence(self, col_cell_type=None, copy=False):
        """
        Find co-occurrence using spatial data. (similar to neighborhood
        enrichment analysis, but uses original spatial coordinates rather
        than connectivity matrix).
        """
        figs = {}
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        jobs = os.cpu_count() - 1  # threads for parallel processing
        adata = self.rna.copy() if copy is True else self.rna
        sq.gr.co_occurrence(adata, cluster_key=col_cell_type, 
                            spatial_key=self._assay_spatial, n_jobs=jobs)
        figs["co_occurrence"] = {}
        figs["spatial_scatter"] = sq.pl.spatial_scatter(
            adata, color=col_cell_type, shape=None, size=2)
        try:
            figs["co_occurrence"] = sq.pl.co_occurrence(
                adata, cluster_key=col_cell_type, figsize=(10, 10))
        except Exception as err:
            figs["co_occurrence"] = err
            print(f"{err}\n{type(err)}\n{err.args}\n\n"
                  "Failed to plot co-occurrence!")
        return adata, figs
        
    def find_svgs(self, genes=None, method="moran", n_perms=10,
                  layer=None, adata=None, col_cell_type=None):
        """Find spatially-variable genes."""
        fig = {}
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
        fig["scatter"] = sq.pl.spatial_scatter(
            adata, library_id=self._assay_spatial,
            color=genes, shape=None, size=2, img=False)
        fig["umap"] = sc.pl.spatial(adata, color=genes) 
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
                                  alpha=0.005, copy=False, **kwargs):
        """Calculate receptor-ligand interactions."""
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        adata = self.rna.copy() if copy is True else self.rna
        res = sq.gr.ligrec(
            adata, n_perms=n_perms, cluster_key=col_cell_type,
            transmitter_params={"categories": "ligand"}, 
            receiver_params={"categories": "receptor"},
            interactions_params={'resources': 'CellPhoneDB'}, **kwargs)
        if key_source is not None and key_targets is not None:
            fig = sq.pl.ligrec(res, alpha=alpha, source_groups=key_source, 
                               target_groups=key_targets)  # plot 
        else:
            fig = None
        return adata, fig
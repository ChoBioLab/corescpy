#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""

import squidpy as sq
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
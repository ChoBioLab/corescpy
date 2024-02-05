#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: E. N. Aslinger
"""

import os
import squidpy as sq
import spatialdata
import scanpy as sc
import traceback
from warnings import warn
import matplotlib
import matplotlib.pyplot as plt
import corescpy as cr
import pandas as pd
import numpy as np

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Spatial(cr.Omics):
    """A class for CRISPR analysis and visualization."""

    def __init__(self, file_path, col_sample_id="Sample", library_id="A",
                 file_path_spatial=None, visium=False, **kwargs):
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
        super().__init__(file_path, spatial=True, col_sample_id=col_sample_id,
                         library_id=library_id, visium=visium, **kwargs)
        self._spatial_key = cr.pp.SPATIAL_KEY
        if self._columns["col_sample_id"] not in self.rna.obs:
            self.rna.obs.loc[:, self._columns["col_sample_id"]] = library_id
        self.figures["spatial"], self.results["spatial"] = {}, {}
        self._library_id = library_id
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)

    def update_from_h5ad(self, file=None):
        """Update SpatialData object `.table` from h5ad file."""
        self.rna = sc.read(os.path.splitext(file)[0] + ".h5ad")
        csid = self._columns["col_sample_id"]
        if isinstance(self.adata, spatialdata.SpatialData) and (
                len(self.rna.obs[csid].unique()) > 1):  # multi-sample
            print("\n\n<<< RESTORING MULTI-SAMPLE FROM h5ad >>>\n")
            for s in self.rna.obs[csid].unique():
                self.rna[self.rna.obs[csid] == s] = cr.pp.update_spatial_uns(
                    self.adata, self._library_id, csid, rna_only=True)
        elif isinstance(self.adata, spatialdata.SpatialData):  # single-sample
            print("\n\n<<< RESTORING SINGLE-SAMPLE FROM h5ad >>>\n")
            self.rna = cr.pp.update_spatial_uns(
                self.adata, self._library_id, csid, rna_only=True)
        else:  # if not SpatialData object
            print("\n\n<<< RESTORED FROM h5ad >>>\n")

    def write(self, file, mode="h5ad", **kwargs):
        """Write AnnData to .h5ad (default) or SpatialData to .zarr."""
        if mode == "h5ad":
            file = os.path.splitext(file)[0] + ".h5ad"
            adata = self.adata.table.copy()  # copy so don't alter self
            if self._spatial_key in adata.uns:  # can't write all SpatialData
                _ = adata.uns.pop(self._spatial_key)  # have to remove .images
            adata.write_h5ad(file, **kwargs)
        else:
            self.adata.write(file, **{"overwrite": True, **kwargs})

    def read_parquet(self, directory=None, kind="transcripts"):
        """
        Read parquet file.

        Specify 'transcripts', 'cell_boundaries', 'nucleus_boundaries'
        or another file stem preceding '.parquet' as `kind`.
        """
        if directory is None:
            directory = self._dir
        for x in self.adata.uns["spatial"]:
            file_parquet = os.path.join(os.path.dirname(self.adata.uns[
                "spatial"][x]["metadata"]["file_path"]), f"{kind}.parquet")
            return pd.read_parquet(file_parquet)

    def print_tiff(self):
        """Print information from tiff file."""
        print(f"\n\n\n{'=' * 80}\nTIFF INFORMATION\n{'=' * 80}\n\n")
        for x in self.adata.uns["spatial"]:
            print(f"\n\t\t{'-' * 40}\n{x}\n{'-' * 40}\n\n")
            cr.pp.describe_tiff(os.path.dirname(self.adata.uns[
                "spatial"][x]["metadata"]["file_path"]))

    def plot_spatial(self, color=None, kind="scatter", key_image=None,
                     col_sample_id=None, library_id=None, title="",
                     shape="hex", figsize=30, cmap="magma", wspace=0.4,
                     **kwargs):
        """Create basic spatial plots."""
        if isinstance(figsize, (int, float)):
            figsize = (figsize, figsize)
        if col_sample_id is None:
            col_sample_id = self._columns["col_sample_id"]
        if isinstance(self.adata, spatialdata.SpatialData) and shape:
            warn("Can't currently use `shape` parameter with SpatialData.")
            shape = None
        cgs = self._columns["col_gene_symbols"] if (self._columns[
            "col_gene_symbols"] != self.rna.var.index.names[0]) else None
        # if library_id is None:  # all libraries if unspecified
        #     library_id = list(self.rna.obs[col_sample_id].unique())
        adata = self.rna.copy()
        _ = adata.uns.pop("leiden_colors", None)
        color = None if color is False else color if color else self._columns[
            "col_cell_type"]  # no color if False; clusters if unspecified
        if color is not None:
            color = list(pd.unique(self.get_variables(color)))
        kws = dict(figsize=figsize, shape=shape, title=title, color=color,
                   # img_res_key=key_image, library_key=col_sample_id,
                   # library_id=library_id,
                   cmap=cmap, alt_var=cgs, wspace=wspace, **kwargs)
        fig = sq.pl.spatial_scatter(adata, **kws) if (
            kind == "scatter") else sq.pl.spatial_segment(adata, **kws)
        return fig

    def calculate_centrality(self, col_cell_type=None, delaunay=True,
                             coord_type="generic", n_jobs=None, figsize=None,
                             palette=None, shape="hex", size=None, title=None,
                             kws_plot=None, copy=False, **kwargs):
        """
        Characterize connectivity, centrality, and interaction matrix.
        """
        # Connectivity & Centrality
        print("\t*** Building connectivity matrix...")
        adata = self.adata
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        if size is None:
            size = int(1 if figsize[0] < 25 else figsize[0] / 15)
        if isinstance(palette, list):
            palette = matplotlib.colors.Colormap(palette)
        kws_plot = {**dict(figsize=figsize, palette=palette, size=size),
                    **dict(kws_plot if kws_plot else {})}
        if n_jobs is None and n_jobs is not False:
            n_jobs = os.cpu_count() - 1  # threads for parallel processing
        sq.gr.spatial_neighbors(
            adata, coord_type=coord_type, delaunay=delaunay,
            spatial_key=self._spatial_key)  # spatial neighbor calculation
        print("\t*** Computing & plotting centrality scores...")
        sq.gr.centrality_scores(adata, cluster_key=col_cell_type,
                                n_jobs=n_jobs)  # centrality scores
        try:
            ann = adata.table.copy()
            _ = ann.uns.pop("leiden_colors", None)  # Squidpy palette bug
            sq.pl.centrality_scores(ann, cluster_key=col_cell_type,
                                    figsize=figsize)
        except Exception:
            traceback.print_exc()
        fig = plt.gcf()
        if title:
            fig.suptitle(title)
        print("\t*** Computing interaction matrix...")
        sq.gr.interaction_matrix(adata, col_cell_type, normalized=False)
        if copy is False:
            self.figures["centrality"] = fig
        return adata, fig

    def calculate_neighborhood(self, col_cell_type=None, mode="zscore",
                               library_id=None, library_key=None, seed=1618,
                               layer=None, palette=None, size=None,
                               key_image=None, shape="hex",
                               title="Neighborhood Enrichment",
                               kws_plot=None, figsize=None, cmap="magma",
                               vcenter=None, cbar_range=None, copy=False):
        """Perform neighborhood enrichment analysis."""
        adata = self.adata
        if library_key is None:
            library_key = self._columns["col_sample_id"]
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if cbar_range is None:
            cbar_range = [None, None]
        if library_id is None:
            library_id = self._library_id
        if key_image is None:
            key_image = list(self.rna.uns[self._spatial_key][library_id][
                "images"].keys())[0]
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        if size is None:
            size = int(1 if figsize[0] < 25 else figsize[0] / 15)
        if isinstance(palette, list):
            palette = matplotlib.colors.Colormap(palette)
        kws_plot = cr.tl.merge(dict(palette=palette, size=size, use_raw=False,
                                    layer=layer), kws_plot)
        sq.gr.nhood_enrichment(adata, cluster_key=col_cell_type,
                               n_jobs=None,
                               # n_jobs=n_jobs,  # not working for some reason
                               seed=seed)  # neighborhood enrichment
        fig, axs = plt.subplots(1, 2, figsize=figsize)  # set up facet figure
        try:
            ann = adata.table.copy()
            _ = ann.uns.pop("leiden_colors", None)  # Squidpy palette bug
            sq.pl.nhood_enrichment(
                ann, cluster_key=col_cell_type, title=title,
                vcenter=vcenter, vmin=cbar_range[0], vmax=cbar_range[1],
                library_id=library_id, library_key=library_key,
                img_res_key=key_image, cmap=cmap, ax=axs[0])  # matrix
        except Exception:
            traceback.print_exc()
        try:
            self.plot_spatial(color=col_cell_type, ax=axs[1], shape=shape,
                              figsize=figsize, cmap=cmap)  # cells (panel 2)
        except Exception:
            traceback.print_exc()
        if copy is False:
            self.figures["neighborhood_enrichment"] = plt.gcf()
        return adata, fig

    def find_cooccurrence(self, col_cell_type=None, key_cell_type=None,
                          layer=None, library_id=None, cmap="magma",
                          n_jobs=None, figsize=15, palette=None, title=None,
                          kws_plot=None, shape="hex", size=None, **kwargs):
        """
        Find co-occurrence using spatial data. (similar to neighborhood
        enrichment analysis, but uses original spatial coordinates rather
        than connectivity matrix).
        """
        adata = self.adata
        if isinstance(key_cell_type, str):
            key_cell_type = [key_cell_type]
        fig = {}
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        if size is None:
            size = int(1 if figsize[0] < 25 else figsize[0] / 15)
        if isinstance(palette, list):
            palette = matplotlib.colors.Colormap(palette)
        kws_plot = cr.tl.merge(dict(palette=palette, size=size), kws_plot)
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if n_jobs is None and n_jobs is not False:
            n_jobs = os.cpu_count() - 1  # threads for parallel processing
        sq.gr.co_occurrence(adata, cluster_key=col_cell_type, n_jobs=n_jobs,
                            spatial_key=self._spatial_key, **kwargs)
        fig["co_occurrence"] = {}
        try:
            fig["spatial_scatter"] = self.plot_spatial(
                color=col_cell_type, groups=key_cell_type,
                shape=shape, figsize=figsize, cmap=cmap,
                library_id=library_id, return_ax=True)  # cell types plot
            if title:
                fig["spatial_scatter"].suptitle(title)
        except Exception:
            traceback.print_exc()
        # fig["co_occurrence"] = sq.pl.co_occurrence(
        #     adata, cluster_key=col_cell_type, legend=False,
        #     clusters=key_cell_type, figsize=figsize)  # plot co-occurrrence
        try:
            ann = adata.table.copy()
            _ = ann.uns.pop("leiden_colors", None)  # Squidpy palette bug
            fig["co_occurrence"] = cr.pl.plot_cooccurrence(
                ann, col_cell_type=col_cell_type, **kws_plot,
                key_cell_type=key_cell_type, figsize=figsize)  # lines plot
        except Exception as err:
            fig["co_occurrence"] = str(traceback.format_exc())
            print(traceback.format_exc())
        if title:
            fig["co_occurrence"].suptitle(title)
        self.figures["co_occurrence"] = fig["co_occurrence"]
        return adata, fig

    def find_svgs(self, genes=10, method="moran", shape="hex", n_perms=10,
                  layer=None, library_id=None, col_cell_type=None, title=None,
                  col_sample_id=None, n_jobs=2, figsize=15,
                  key_image=None, kws_plot=None):
        """Find spatially-variable genes."""
        adata = self.adata
        # adata.table = self.get_layer(layer=layer, subset=None, inplace=True)
        kws_plot = cr.tl.merge({"cmap": "magma", "use_raw": False}, kws_plot)
        if n_jobs == -1:
            n_jobs = os.cpu_count() - 1  # threads for parallel processing
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        # if isinstance(palette, list):
        #     palette = matplotlib.colors.Colormap(palette)
        # if size is None:
        #     size = int(1 if figsize[0] < 25 else figsize[0] / 15)
        # kws_plot = {**dict(palette=palette, shape=shape, size=size),
        #             **dict(kws_plot if kws_plot else {})}
        print(f"\n<<< QUANTIFYING AUTO-CORRELATION (method = {method}) >>>")
        sq.gr.spatial_autocorr(
            adata, mode=method, layer=layer, n_perms=n_perms,
            n_jobs=n_jobs)  # auto-correlate
        if isinstance(genes, int):
            genes = self.rna.uns["moranI"].head(genes).index.values
        ncols = cr.pl.square_grid(len(genes + [col_cell_type]))[1]
        fig = None
        try:
            fig = self.plot_spatial(
                genes + [col_cell_type], shape=shape, figsize=figsize,
                return_ax=True, key_image=key_image, library_id=library_id,
                **kws_plot)
            if title:
                fig.suptitle(title)
        except Exception:
            fig = str(traceback.format_exc())
            traceback.print_exc()
        # sc.pl.spatial(adata, color=genes, library_id=library_id,
        #               figsize=figsize, **kws_plot)  # SVGs GEX plot
        return adata, fig

    def calculate_distribution_pattern(self, col_cell_type=None, mode="L"):
        """
        Use Ripley's statistics to determine whether distributions are
        random, dispersed, or clustered.
        """
        adata = self.adata
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        sq.gr.ripley(adata, cluster_key=col_cell_type, mode=mode)
        fig = sq.pl.ripley(self.adata, cluster_key=col_cell_type, mode=mode)
        return adata, fig

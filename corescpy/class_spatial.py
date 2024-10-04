#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: E. N. Aslinger
"""

import os
import traceback
import json
from itertools import permutations
from warnings import warn
from copy import deepcopy
from dask_image.imread import imread
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.backends.backend_pdf import PdfPages
import shapely
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
# from shapely.geometry.collection import GeometryCollection
import geopandas as gpd
import squidpy as sq
import spatialdata
from spatialdata.models import TableModel
import spatialdata_plot as sdp
import spatialdata_io as sdio
from spatialdata_io._constants._constants import XeniumKeys
import liana
# from liana.method import MistyData, genericMistyData, lrMistyData
# import decoupler as dc
# import plotnine
import scanpy as sc
import pandas as pd
import numpy as np
import corescpy as cr

SEG_CELL_ID_XENIUM = "region"
SEG_CELL_ID_VISIUM = None
COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Spatial(cr.Omics):
    """A class for CRISPR analysis and visualization."""

    def __init__(self, file_path, col_sample_id="Sample", library_id=None,
                 key_table="table", visium=False,
                 col_gene_symbols=None, **kwargs):
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
                    - a dictionary containing keyword arguments to pass
                        to `corescpy.pp.combine_matrix_protospacer`
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
                `file_path_spatial` provided). Pass `visium='hd'` or a
                dictionary:
                {hd': True, **<spatialdata_io.visium_hd arguments>}.
            key_table (str, optional): For `spatialdata` objects,
                `self.adata.tables[key_table]` will be the default
                used for the `self.rna` attribute.
            col_gene_symbols (str, optional): If the index of the
                `.var` attribute of the `AnnData` object
                (or corresponding `SpatialData` table) isn't the
                gene symbols, specify the column name here.
                This default may differ from other classes because of
                how spatial data files tend to be read in.
            kwargs (dict, optional): Keyword arguments to pass to the
                Omics class initialization method.
        """
        print("\n\n<<< INITIALIZING SPATIAL CLASS OBJECT >>>\n")
        _ = kwargs.pop("file_path_spatial", None)
        super().__init__(file_path, spatial=True, col_sample_id=col_sample_id,
                         library_id=library_id, visium=visium,
                         col_gene_symbols=col_gene_symbols,
                         verbose=False, key_table=key_table, **kwargs)
        self._spatial_key = cr.SPATIAL_KEY
        self._kind = "xenium" if visium is False else "visium"
        if library_id is None and visium is True:
            library_id = list(self.rna.uns[self._spatial_key].keys())
            if len(library_id) == 1:
                library_id = library_id[0]
        if "key_table" not in self._keys:
            self._keys["key_table"] = key_table
        self._columns["col_segment"] = SEG_CELL_ID_VISIUM if (
            visium is True) else SEG_CELL_ID_XENIUM
        if self._columns["col_sample_id"] not in self.rna.obs:
            self.rna.obs.loc[:, self._columns["col_sample_id"]] = library_id
        self.figures["spatial"], self.results["spatial"] = {}, {}
        self._library_id = library_id
        if self._kind == "xenium":
            self.rna.ob = self.rna.obs.set_index("cell_id")
        if isinstance(self.adata, spatialdata.SpatialData):
            self.rna = self.adata.tables[self._keys["key_table"]]
            self.adata.pl = sdp.pl.basic.PlotAccessor(self.adata)
            print(self.adata)
        for q in [self._columns, self._keys]:
            cr.tl.print_pretty_dictionary(q)

    def filter_by_quality(self, threshold):
        """Filter `points` by transcript quality score (QV, phred-like)."""
        if self._kind != "xenium":
            raise ValueError("Filtering transcripts by quality"
                             f"not available for {self._kind}.capitalize().")
        self.adata.points["transcripts"] = self.adata.points["transcripts"][
            self.adata.points["transcripts"].qv >= threshold]

    def update_from_h5ad(self, file=None, file_path_markers=None,
                         method_cluster="leiden", key_table=None):
        """Update SpatialData object `.tables` from h5ad file."""
        key_table = key_table if key_table else self._keys["key_table"]
        original_ix = self.rna.uns[
            "original_ix"] if "original_ix" in self.rna.uns else None
        ann = sc.read(os.path.splitext(file)[0] + ".h5ad")
        for x in ann.uns[TableModel.ATTRS_KEY]:
            if isinstance(ann.uns[TableModel.ATTRS_KEY][x], np.ndarray):
                ann.uns[TableModel.ATTRS_KEY][x] = list(
                    ann.uns[TableModel.ATTRS_KEY][x])
        self.rna = ann
        csid = self._columns["col_sample_id"]
        if file_path_markers and not os.path.exists(file_path_markers):
            file_path_markers = None
            warn(f"Marker file {file_path_markers} does not exist; skipping.")
        if isinstance(self.adata, spatialdata.SpatialData) and (
                len(self.rna.obs[csid].unique()) > 1):  # multi-sample
            print("\n\n<<< RESTORING MULTI-SAMPLE FROM h5ad >>>\n")
            for s in self.rna.obs[csid].unique():
                self.rna[self.rna.obs[csid] == s] = cr.pp.update_spatial_uns(
                    self.adata, self._library_id, csid, rna_only=True,
                    spatial_key=self._spatial_key, key_table=key_table)
        elif isinstance(self.adata, spatialdata.SpatialData):  # single-sample
            print("\n\n<<< RESTORING SINGLE-SAMPLE FROM h5ad >>>\n")
            self.rna = cr.pp.update_spatial_uns(
                self.adata, self._library_id, csid, rna_only=True,
                spatial_key=self._spatial_key, key_table=key_table)
        else:  # if not SpatialData object
            print("\n\n<<< RESTORED FROM h5ad >>>\n")
        if original_ix is not None and ("original_ix" not in self.rna.uns or (
                len(self.rna.uns["original_ix"]) <= len(original_ix))):
            self.rna.uns["original_ix"] = original_ix
        if file_path_markers:
            if os.path.exists(file_path_markers) is False:
                raise ValueError(f"{file_path_markers} not found.")
            else:
                if "markers" not in self.rna.uns:
                    self.rna.uns["markers"] = {}
                self.rna.uns["markers"][method_cluster] = pd.read_csv(
                    file_path_markers).set_index([method_cluster, "names"])

    def write(self, file, mode="h5ad", **kwargs):
        """Write AnnData to .h5ad (default) or SpatialData to .zarr."""
        if mode == "h5ad":
            file = os.path.splitext(file)[0] + ".h5ad"
            adata = self.adata.tables[self._keys["key_table"]].copy(
                )  # copy so don't alter self
            if self._spatial_key in adata.uns:  # can't write all SpatialData
                _ = adata.uns.pop(self._spatial_key)  # have to remove .images
            try:
                adata.write_h5ad(file, **kwargs)
            except TypeError:
                _ = adata.uns.pop("markers")  # may not be able to write
                adata.write_h5ad(file, **kwargs)
        else:
            self.adata.write(file, **{"overwrite": True, **kwargs})

    def get_directories(self, library_id):
        dirs = [str(self.rna[self.rna.obs[self._columns[
            "col_sample_id"]] == x].obs["file_path"].iloc[0])
                for x in cr.tl.to_list(library_id)] if (
                    library_id) else self.rna.obs.file_path.unique()  # paths
        return dirs

    def read_panel(self, directory=""):
        """Read gene panel for Xenium data."""
        drop = cr.pp._get_control_probe_names()
        if self._kind != "xenium":
            raise ValueError("Gene panel reading not available for "
                             f"{self._kind.capitalize()}.")
        if os.path.splitext(directory)[1] == ".json":
            file_path = directory
        elif os.path.exists(os.path.join(directory, "gene_panel.json")):
            file_path = os.path.join(directory, "gene_panel.json")
        else:
            if "file_path" in self.rna.obs:
                directory = os.path.join(directory, str(
                    self.rna.obs["file_path"].iloc[0]))
            file_path = os.path.join(directory, "gene_panel.json")
        with open(file_path) as f:
            expect = json.load(f)["payload"]["targets"]
        panel = list(set([g["type"]["data"]["name"] for g in expect]))
        panel = list(set(panel).difference(np.array(panel)[np.where([any((
            i in x for i in drop)) for x in panel])[0]]))
        miss = list(set(panel).difference(self.rna.var_names))
        if len(miss) > 0:
            print(f"\n\nObject {self._library_id} " + str(
                f"is missing genes from panel: \n\n{miss}\n\n"))
            print("This may be normal if you've filtered the data.")
        else:
            print(f"\n\n{self._library_id}: All expected genes present!")
        return expect, panel

    def read_parquet(self, directory=None, kind="transcripts",
                     library_id=None):
        """
        Read parquet file.

        Specify 'transcripts', 'cell_boundaries', 'nucleus_boundaries'
        or another file stem preceding '.parquet' as `kind`.
        """
        dirs = self.get_directories(library_id=library_id)
        parqs = []
        for x in dirs:
            parqs += [pd.read_parquet(os.path.join(x, f"{kind}.parquet"))]
        return dict(zip(dirs, parqs)) if len(parqs) > 1 else parqs[0]

    def print_tiff(self, file=None, library_id=None, plot=True,
                   kind="mip", **kwargs):
        """Print and, optionally, plot information from tiff file."""
        if self._kind != "xenium":
            raise NotImplementedError(
                f"No support for kind {self._kind.capitalize()} TIFF plots.")
        dirs = self.get_directories(library_id=library_id)
        print(f"\n\n\n{'=' * 80}\nTIFF INFORMATION\n{'=' * 80}\n\n")
        for x in dirs:  # iterate sample directories
            fff = os.path.join(x, f"morphology_{kind}.tiff")
            print(f"\n\t\t{'-' * 40}\n{fff}\n{'-' * 40}\n\n")
            cr.pp.describe_tiff(fff)  # describe
            if plot is True:
                cr.pl.plot_tiff(fff)  # plot

    def crop(self, bounds_x, bounds_y=None, bounds_z=None,
             pixel_size=0.2125, allow_make_valid=False, **kwargs):
        """
        Get a copy of the data cropped to specified coordinates.

        Notes
        -----
        Setting `allow_make_valid` to True corrects invalid geometries,
        but you MUST CHECK that this new shape is what you want.

        Specify as a list with the minimum & maximum x, y, &
        (optionally) z coordinates (`bounds_z`). For instance,
        `bounds_x=[2, 400]` and `bounds_y=[50, 200]` will yield the
        region defined by 2-400 on the x-axis and 50-200 on the y-axis.

        Alternatively, specify only `bounds_x` as a path
        (or list of paths) to a file(s) created using the Xenium
        Explorer selection tool to extract the coordinates from there.

        You can also pass a `shapely` polygon or multipolygon object
        to `bounds_x`.
        """
        key_table = kwargs.pop("key_table", self._keys["key_table"])
        if isinstance(bounds_x, (str, Polygon, MultiPolygon)) or isinstance(
                bounds_x[0], str):  # Xenium Explorer selection(s)
            if isinstance(bounds_x, (list, np.ndarray, tuple, str)):  # file
                if isinstance(bounds_x, (list, np.ndarray, tuple)):  # fileS
                    coords = [cr.pp.xenium_explorer_selection(
                         i, pixel_size=pixel_size, as_list=False)
                               for i in bounds_x]
                else:  # single file
                    coords = cr.pp.xenium_explorer_selection(
                        bounds_x, pixel_size=pixel_size, as_list=False)
            else:  # if shapely object
                coords = bounds_x
            if isinstance(coords, list):  # if multiple selections...
                coords = shapely.MultiPolygon(coords)  # ...union of areas
            kws = {"target_coordinate_system": "global",
                   "filter_table": True, **kwargs}
            invalid = "is_valid" in dir(coords) and coords.is_valid is False
            p_coords = gpd.GeoSeries(coords).plot()
            p_coords.set_title("Input")
            plt.show()
            if invalid is True and allow_make_valid is True:
                warn("\n\n*** Invalid geometry! Making geometry valid."
                     "Correcting shape. Check new coordinates!\n\n")
                coords = shapely.validation.make_valid(coords)
                p_coords_new = gpd.GeoSeries(coords).plot()
                p_coords_new.set_title("Corrected")
                plt.show()
            sdata_crop = self.adata.query.polygon(coords, **kws)
            if "tables" not in dir(sdata_crop) or (
                    key_table not in sdata_crop.tables):
                sdata_crop.tables[key_table] = self.rna[self.rna.obs[
                    XeniumKeys.CELL_ID].isin(sdata_crop.shapes[
                        spatialdata.SpatialData.get_annotated_regions(
                            sdata_crop.tables[
                                key_table])].index)].copy()  # restore table
                # needed for now b/c issue if any nuclei radius = NA
                # github.com/scverse/spatialdata-io/
                # issues/173#issuecomment-2231152498
            else:
                i_x = sdata_crop.tables[key_table].obs[
                    XeniumKeys.CELL_ID].copy()  # cropped index
                del sdata_crop.tables[key_table]
                sdata_crop.tables[key_table] = self.rna[
                    self.rna.obs[XeniumKeys.CELL_ID].isin(i_x[
                        i_x.isin(list(self.rna.obs[XeniumKeys.CELL_ID]))])]
        else:  # specified coordinates
            minc, maxc = [[x[i] for x in [bounds_x, bounds_y, bounds_z] if (
                x is not None)] for i in [0, 1]]
            kws_def = dict(axes=("x", "y", "z") if bounds_z else ("x", "y"),
                           target_coordinate_system="global")
            sdata_crop = spatialdata.bounding_box_query(
                self.adata, min_coordinate=minc,
                max_coordinate=maxc, **{**kws_def, **kwargs})
        if "tables" in dir(sdata_crop) and sdata_crop.tables is not None and (
                self._keys["key_table"] in sdata_crop.tables and (
                    sdata_crop.tables[self._keys["key_table"]] is not None)):
            sc.pp.calculate_qc_metrics(sdata_crop.tables[self._keys[
                "key_table"]], percent_top=None, inplace=True)
        return sdata_crop

    def add_image(self, file, name=None, file_align=None, dim="2d"):
        """Add image (optionally, align from Xenium alignment file).

        Args:
            name (str): Desired name of image (used as key in certain
                `.adata`/`.rna` object attributes).
            file (str or PathLike, optional): Path to iamge or
                MultiscaleSpatialImage object. Defaults to None
                (will use base path if `file` is provided as a path).
            file_align (str, dict, or PathLike, optional): Path to
                Xenium image alignment file if image is not aligned.
                Provide as dictionary with the file path keyed by
                'file_align' (for the path to the image alignment file)
                and 'imread_kwargs' and 'image_models_kwargs' for
                arguments to pass to the eponymous arguments in the
                `spatialdata_io.xenium_aligned_image()` function.
                Defaults to None.
            dim (str, optional): "2d" or "3d" image. nDefaults to "2d".

        Raises:
            NotImplementedError: Image alignment for Visium data.
        """
        if name is None:  # if image name not provided, use path as name
            if not isinstance(file, (str, os.PathLike)):
                raise ValueError("Please provide a name for image if `file` "
                                 "is not provided as a file path.")
            name = os.path.splitext(os.path.basename(file))[0]  # name=path
        if file_align is not None:  # if image alignment file exists
            if self._kind.lower() == "visium":  # if Visium
                raise NotImplementedError("Visium alignment not supported.")
            akw = deepcopy(file_align) if isinstance(file_align, dict) else {}
            if isinstance(file_align, dict):  # if keyword arguments provided
                file_align = akw.pop("file_align")
            img = sdio.xenium_aligned_image(file, file_align, **akw)
        else:  # if image already aligned
            data = np.array(file) if isinstance(file, (
                np.ndarray, pd.DataFrame)) else imread(file)  # image -> array
            img = spatialdata.models.Image3DModel.parse(data) if (
                dim == "3d") else spatialdata.models.Image2DModel.parse(data)
        self.adata.images[name] = img

    def show(self, kinds="all", elements="all", color=None, layer=None,
             palette=None, figsize=None, library_id=None, **kwargs):
        """Plot spatial data using the spatialdata-plot framework."""
        fxs, kws = [], []
        print(self.adata)
        libs = library_id if library_id else self._library_id
        if isinstance(kinds, str) and kinds.lower() == "all":
            kinds = ["points", "images", "shapes", "labels"]
        kinds = list(set(cr.to_list(kinds)).intersection(dir(self.adata)))
        plotter = sdp.pl.basic.PlotAccessor(self.adata)
        if isinstance(elements, str) and elements.lower() == "all":
            elements = []
            for x in kinds:
                if x in dir(self.adata):
                    elements += list(getattr(self.adata, x).keys())
        elements = cr.tl.to_list(elements)
        for q in [color, palette]:
            if isinstance(q, dict):  # if different palette ~ data type
                for x in ["points", "images", "shapes", "labels"]:
                    if x not in q:  # if certain type unspecified...
                        q[x] = None  # ...add for consistency
        if not isinstance(color, dict):
            color = cr.to_list(color)
        avail = list(set(self.rna.var_names.union(
            self.rna.var.columns.union(self.rna.obs.columns))))  # variables

        # Construct Function Calls ~ Data Type
        if "points" in kinds and len(self.adata.points) > 0:
            pnts = [v for v in elements if v in self.adata.points]
            ptt = palette["points"] if isinstance(palette, dict) else palette
            clr = color["points"] if isinstance(color, dict) else [
                v for v in color if v in avail]  # genes
            # pt_plter = sdp.pl.basic.PlotAccessor(self.adata) if (
            #     points_fraction is None) else sdp.pl.basic.PlotAccessor(
            #         self.adata[])
            if len(pnts) > 0:
                kws += [dict(elements=pnts[0] if len(pnts) == 1 else pnts if (
                    len(pnts) > 0) else None, palette=ptt, color=clr)]
                fxs += [plotter.render_points]
        if "labels" in kinds and len(self.adata.labels) > 0:
            labs = [v for v in elements if v in self.adata.labels]
            clr = color["labels"] if isinstance(color, dict) else None  # hue
            ptt = palette["labels"] if isinstance(palette, dict) else palette
            if len(labs) > 0:
                kws += [dict(layer=layer, palette=ptt, color=clr)]
                fxs += [plotter.render_labels]
        if "shapes" in kinds and len(self.adata.shapes) > 0:
            shps = [v for v in elements if v in self.adata.shapes]
            ptt = palette["shapes"] if isinstance(palette, dict) else palette
            clr = color["shapes"] if isinstance(color, dict) else None  # hue
            if len(shps) > 0:
                kws += [dict(elements=shps, layer=layer,
                             palette=ptt, color=clr)]
                fxs += [plotter.render_shapes]
        if "images" in kinds and len(self.adata.images) > 0:
            imgs = [v for v in elements if v in self.adata.images]
            ptt = palette["images"] if isinstance(palette, dict) else palette
            clr = color["images"] if isinstance(color, dict) else None  # hue
            if len(imgs) > 0:
                k_i = dict(elements=imgs, channel=kwargs.pop("channel", None),
                           palette=ptt, color=clr)
                for x in ["cmap", "norm", "na_color", "alpha", "scale"]:
                    if x in kwargs:
                        k_i[x] = kwargs.pop(x)
                kws += [k_i]
                fxs += [plotter.render_images]

        # Plot All
        if len(fxs) > 1:  # multi-plot
            fig, axs = plt.subplots(ncols=len(fxs), figsize=figsize, **kwargs)
            for i, f in enumerate(fxs):
                f(**kws[i]).pl.show(coordinate_systems=libs, ax=axs[i])
        else:  # single plot
            fxs[0](**kws[0]).pl.show(coordinate_systems=libs,
                                     figsize=figsize, **kwargs)
        return fig

    def plot_spatial(self, color=None, kind="dot", key_image=None,
                     col_sample_id=None, library_id=None, title=None,
                     shape="hex", figsize=30, cmap="magma", wspace=0,
                     mode="squidpy", layer=None, title_offset=0,
                     out_file=None, **kwargs):
        """Create basic spatial plots."""
        color = None if color is False else color if color else self._columns[
            "col_cell_type"]  # no color if False; clusters if unspecified
        if color is not None:
            color = list(pd.unique(self.get_variables(color)))
        libid = library_id if library_id else self._library_id
        col_sample_id = col_sample_id if col_sample_id else kwargs.pop(
            "libary_key", self._columns["col_sample_id"])
        seg = None if kind.lower() == "dot" else kwargs.pop(
            "seg_cell_id", self._columns["col_segment"])  # segmentation ID
        if isinstance(self.adata, spatialdata.SpatialData) and shape:
            warn("Can't currently use `shape` parameter with SpatialData.")
            shape = None
        cgs = self._columns["col_gene_symbols"] if (self._columns[
            "col_gene_symbols"] != self.rna.var.index.names[0]) else None
        # if library_id is None:  # all libraries if unspecified
        #     library_id = list(self.rna.obs[col_sample_id].unique())
        ann = self.rna.copy()
        kws = cr.tl.merge(dict(figsize=figsize, shape=shape, cmap=cmap,
                               return_ax=True, library_key=col_sample_id,
                               library_id=libid, alt_var=cgs,
                               wspace=wspace), kwargs)  # keyword arguments
        kws["img_res_key"] = key_image if key_image else None if len(
            self.rna.uns[self._spatial_key][libid]["images"]) == 0 else list(
                self.rna.uns[self._spatial_key][libid]["images"].keys())[0]
        if isinstance(color, str) or len(color) == 1:
            color = color if isinstance(color, str) else color[0]
            fig = cr.pl.plot_spatial(ann, col_segment=seg, title=title,
                                     save=out_file, color=color, **kws)
            return fig
        else:
            fig = {}
            for c in color:
                fig[c] = cr.pl.plot_spatial(ann, col_segment=seg, title=title,
                                            save=out_file, color=c, **kws)
        return fig

    def plot_clusters(self, col_cell_type=None, key_cell_type=None,
                      out_dir=None, sep="_", multi_pdf=False,
                      kws_save=None, reorder=False, **kwargs):
        """Plot spatial distributions of clusters; optionally, save

        Args:
            col_cell_type (str, optional): Name of column to plot.
                Defaults to None (`self._columns['col_cell_type']`).
            key_cell_type (str or list, optional): One or more entries
                within the `col_cell_type` column to plot (e.g.,
                specific  clusters). Defaults to None (will plot all).
            out_dir (str, optional): Directory path within which to
                save plots. The files within this directory will be
                named as
                <self._library_id><sep><sep><col_cell_type>\
                    <sep><cluster>
                unless `multi_pdf=True`, in which case, `out_dir`
                should be the file path where you want to save outputs
                as separate pages in the same PDF. Defaults to None
                (won't save ouput).
            sep (str, optional): Separator between `col_cell_type` and
                the cluster name in the save file name as described
                above. (Moot if `multi_pdf=True`.) Defaults to "_".
            kws_save (dict, optional): Additional keyword arguments to
                pass to `matplotlib`'s `figure.savefig()` (optionally).
        """
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        kws_save = {} if kws_save is None else {**kws_save}
        title = kwargs.pop("title", f"{self._library_id}: {col_cell_type}")
        if key_cell_type is None or isinstance(key_cell_type, (
                int, float, str)):
            key_cell_type = [key_cell_type] if isinstance(key_cell_type, (
                int, float, str)) else self.rna.obs[col_cell_type].unique()
        figures = [self.plot_spatial(color=col_cell_type, **kwargs,
                                     title=f"{col_cell_type}")]
        if reorder is True:
            key_cell_type = [str(i) for i in list(pd.Series(
                key_cell_type).astype(int).sort_values())]  # #s in order
        for u, i in enumerate(key_cell_type):  # iterate clusters to plot
            fig = self.plot_spatial(color=col_cell_type, groups=i, **kwargs,
                                    title=f"{col_cell_type}\nCluster {i}")
            if title is not None:
                fig.figure.suptitle(str(title if isinstance(title, (
                    int, float, str)) else title[u]))  # super-title
            if out_dir is not None and multi_pdf is False:  # plot separately
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)  # make directory if needed
                pfp = f"{self._library_id}{sep}{sep}{col_cell_type}{sep}{i}"
                fig.figure.savefig(os.path.join(
                    out_dir, pfp + ".jpg"), **kws_save)
            figures += [fig]
        if multi_pdf is True and out_dir is not None:
            if not os.path.exists(os.path.dirname(out_dir)):
                os.makedirs(os.path.dirname(out_dir))  # make directory
            with PdfPages(out_dir) as pdf:
                for fig in figures:
                    pdf.savefig(fig.figure, **kws_save)
        return figures

    def plot_compare_spatial(self, others, color, cmap="magma",
                             wspace=0.3, layer="log1p", **kwargs):
        """Compare spatial plots to those of other Spatial objects."""
        if isinstance(color, str):
            color = [color]
        selves = [self] + list(others)
        f_s = kwargs.pop("figsize", (5 * len(color), 20 * len(selves)))
        fig, axs = plt.subplots(len(color) + 1, len(selves), figsize=f_s)
        for j, s in enumerate(selves):  # iterate objects
            goi = [s._columns["col_cell_type"]] + color
            for i, g in enumerate(goi):
                s.plot_spatial(ax=axs[i, j], cmap=cmap, layer=layer,
                               color=g, **kwargs)  # plot
                if i == 0:
                    axs[i, j].set_title(s._library_id)
        plt.subplots_adjust(wspace=wspace)
        fig.show()
        return fig, axs

    def plot_segmentation(self, key_image="morphology_focus", figsize=30,
                          kws_crop=None, kws_show=None, channels=None,
                          titles=None, xenium_multimodal=False, **kwargs):
        """Plot spatial segmentation."""
        sdata = self.crop(**kws_crop) if kws_crop else self.adata
        kws_show = {} if kws_show is None else {**kws_show}
        channels = [channels] if channels is None or isinstance(channels, (
            str, float, int)) else list(channels)  # ensure channels iterable
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else figsize  # 1 size given=square figsize -> tuple
        if titles is None:
            titles = [
                "Nuclear (DAPI)", "Interior Protein (AlphaSMA/Vimentin)",
                "Boundary (ATP1A1/CD45/E-Cadherin)", "Interior RNA (18S)"
                ] if xenium_multimodal is True else channels
        if isinstance(titles, str):  # if only one title given...
            titles = [titles] * len(channels)  # ...same for all channels
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for i, x in enumerate(channels):  # iterate channels
            sdata.pl.render_images(key_image, channel=x, **kwargs).pl.show(
                title=titles[i], **kws_show, ax=axes.flatten()[i])  # plot
        return fig, axes

    def plot_image(self, image_name, title="", figsize=None, **kwargs):
        """Plot a `spatialdata` object image(s)."""
        if "coordinate_systems" not in kwargs:
            kwargs["coordinate_systems"] = "global"
        figsize = kwargs.pop("figsize", None)
        if isinstance(image_name, (list, tuple, np.ndarray)):  # multi-image
            if isinstance(title, str) or title is None:
                title = [title] * len(image_name)  # if all same title
            if figsize is None:
                figsize = (20, 6 * len(image_name))
            axes = plt.subplots(1, len(image_name), figsize=figsize)[
                1].flatten()
            for i, x in enumerate(image_name):
                self.plot_image(x, title=title[i], ax=axes[i], **kwargs)
        else:  # single image plot
            self.adata.pl.render_images(image_name).pl.show(
                title=title, **kwargs)

    def impute(self, adata_sc, col_cell_type=None, mode="cells",
               layer="log1p", device="cpu", inplace=True,
               col_annotation="tangram_prediction", out_file=None,  **kwargs):
        """Impute scRNA-seq GEX & annotations onto spatial data."""
        adata_sp = self.get_layer(layer, inplace=inplace)  # get layer
        if col_cell_type is None:  # if unspecified, default cluster column
            col_cell_type = self._columns["col_cell_type"]
        if inplace is False:  # if don't want to modify objects inplace...
            adata_sc = adata_sc.copy()  # ...copy scRNA-seq
        if layer and layer in adata_sc.layers:  # if specified layer in scRNA
            adata_sc.X = adata_sc.layers[layer].copy()  # scRNA-seq layer
        key = kwargs.pop("key_added", f"key_added_{col_cell_type}"
                         )  # key to add (or use, if exists) in .uns for DEGs
        out = cr.pp.impute_spatial(
            adata_sp, adata_sc.copy(), col_cell_type=col_cell_type, mode=mode,
            key_added=key, inplace=True, device=device,
            col_annotation=col_annotation, **kwargs)  # integrate
        if inplace is False:
            self.rna.uns["tangram_object"] = out[0]
            self.rna.obs.loc[:, col_annotation] = out[0].obs[col_annotation]
        else:
            self.rna = out[0]
        if out_file and os.path.splitext(out_file)[1] != ".csv":
            self.write(out_file)
        return out

    def calculate_centrality(self, col_cell_type=None, delaunay=True,
                             coord_type="generic", n_jobs=None, figsize=None,
                             palette=None, copy=False, cmap="magma",
                             title=None, kws_plot=None,
                             out_plot=None, normalized=True, **kwargs):
        """
        Characterize connectivity, centrality, and interaction matrix.
        """
        # Connectivity & Centrality
        print("\n\n<<< QUANTIFYING GRAPH >>>\n\n")
        print("\t*** Building connectivity matrix...")
        out, ext = os.path.splitext(out_plot) if out_plot else (None, None)
        if ext == "":
            ext = ".jpeg"
        cct = col_cell_type if col_cell_type else self._columns[
            "col_cell_type"]
        self.rna.obs = self.rna.obs.astype({cct: "category"})
        adata = self.adata
        f_s = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (30, 7) if figsize is None else figsize
        if isinstance(palette, list):
            palette = matplotlib.colors.Colormap(palette)
        kws_plot = cr.tl.merge(dict(figsize=f_s, palette=palette, cmap=cmap),
                               kws_plot)  # interaction matrix plot arguments
        if n_jobs == -1 or (isinstance(n_jobs, bool) and n_jobs is True):
            n_jobs = os.cpu_count() - 1  # threads for parallel processing
        sq.gr.spatial_neighbors(
            adata, coord_type=coord_type, delaunay=delaunay,
            spatial_key=self._spatial_key)  # spatial neighbor calculation
        print("\t*** Computing & plotting centrality scores...")
        sq.gr.centrality_scores(adata, cluster_key=cct, n_jobs=n_jobs)  # run
        try:
            sq.pl.centrality_scores(
                adata, cluster_key=cct, figsize=f_s,
                save=f"{out}{ext}" if out else None)
            fig = plt.gcf()
        except Exception:
            try:
                ann = adata.tables[self._keys["key_table"]].copy()
                _ = ann.uns.pop(f"{cct}_colors", None)  # Squidpy palette bug
                sq.pl.centrality_scores(
                    ann, cluster_key=cct, figsize=f_s,
                    save=f"{out}{ext}" if out else None)  # plot
                fig = plt.gcf()
            except Exception:
                fig = str(traceback.format_exc())
                traceback.print_exc()
        if not isinstance(fig, str) and title:
            fig.suptitle(title)
        print("\n\t*** Computing interaction matrix...")
        sq.gr.interaction_matrix(adata, cct, normalized=normalized)
        try:
            if "figsize" in kws_plot and kws_plot["figsize"]:
                kws_plot["figsize"] = (kws_plot["figsize"][0] / 3,
                                       kws_plot["figsize"][1])
            sq.pl.interaction_matrix(
                adata, cct, save=f"{out}_interaction{ext}" if out else None,
                **kws_plot)  # plot interaction matrix
        except Exception:
            sq.pl.interaction_matrix(
                adata.tables[self._keys["key_table"]], cct,
                save=f"{out}_interaction{ext}" if out else None,
                **kws_plot)  # plot interaction matrix
        for u, v in zip(["Centrality scores", "Interaction matrix results"], [
                "centrality_scores", "interactions"]):
            print(f"\n\n*** {u} stored in `.rna.uns['{cct}_{v}']`\n\n")
        return adata, fig

    def calculate_neighborhood(self, col_cell_type=None,
                               mode="zscore", normalized=False,
                               library_id=None, library_key=None, seed=1618,
                               layer="log1p", palette=None, size=None,
                               key_image=None, shape="hex", vcenter=None,
                               cbar_range=None, cmap="magma", figsize=None,
                               title="Neighborhood Enrichment", n_jobs=None,
                               kws_plot=None, out_plot=None, copy=False):
        """Perform neighborhood enrichment analysis."""
        adata = self.adata
        if library_key is None:
            library_key = self._columns["col_sample_id"]
        if library_id is None:
            library_id = self._library_id
        if title == "Neighborhood Enrichment":
            title += f" ({mode}{' Normalized by Row' if normalized else ''})"
            title = f"{title} ({library_id})"
        cct = col_cell_type if col_cell_type else self._columns[
            "col_cell_type"]
        if cbar_range is None:
            cbar_range = [None, None]
        if key_image is None and not isinstance(
                self.adata, spatialdata.SpatialData):
            key_image = list(self.rna.uns[self._spatial_key][library_id][
                "images"].keys())[0]
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        kws_plot = {} if kws_plot is None else {**kws_plot}
        if out_plot is not None:
            kws_plot["save"] = out_plot
        if n_jobs == -1 or (isinstance(n_jobs, bool) and n_jobs is True):
            n_jobs = os.cpu_count() - 1  # threads for parallel processing
        print(f"\n\n<<< QUANTIFYING NEIGHBORHOOD ENRICHMENT ({mode}) >>>\n\n")
        sq.gr.nhood_enrichment(adata, cluster_key=cct, n_jobs=n_jobs,
                               seed=seed)  # neighborhood enrichment
        fig, axs = plt.subplots(1, 2, figsize=figsize)  # set up facet figure
        try:
            pkws = dict(cluster_key=cct, title=title, library_id=library_id,
                        library_key=library_key, vcenter=vcenter,
                        vmin=cbar_range[0], vmax=cbar_range[1], ax=axs[0],
                        img_res_key=key_image, cmap=cmap, **kws_plot)
            sq.pl.nhood_enrichment(adata.tables[self._keys["key_table"]] if (
                isinstance(adata, spatialdata.SpatialData)) else adata,
                                   **pkws)  # plot neighborhood enrichment
        except Exception:
            try:
                ann = adata.tables[self._keys["key_table"]].copy()
                _ = ann.uns.pop(f"{cct}_colors", None)  # Squidpy palette bug
                sq.pl.nhood_enrichment(ann, **pkws)  # matrix
            except Exception:
                traceback.print_exc()
        try:
            self.plot_spatial(color=cct, ax=axs[1], shape=shape,
                              figsize=figsize, cmap=cmap)  # cells (panel 2)
        except Exception:
            fig = str(traceback.format_exc())
            traceback.print_exc()
        if copy is False:
            self.figures["neighborhood_enrichment"] = plt.gcf()
        print("\n\n*** Neighborhood results stored in "
              f"`.rna.uns['{cct}_nhood_enrichment']`\n\n")
        return adata, fig

    def cluster_spatial(self, layer="log1p", alpha=0.2,
                        copy=False, **kwargs):
        """Cluster using spatial data (per SC Best Practices)."""
        adata = self.get_layer(layer=layer, inplace=False)
        key_added = kwargs.pop("key_added", "spatial_domains")
        graph_g = adata.obsp["connectivities"]  # gene connectivity
        if "spatial_connectivities" not in adata.obsp:
            sq.gr.spatial_neighbors(adata)  # calculate neighborhood if needed
        graph_s = adata.obsp["spatial_connectivities"]  # spatial connectivity
        joint_graph = (1 - alpha) * graph_g + alpha * graph_s  # gene-spatial
        sc.tl.leiden(adata, adjacency=joint_graph, key_added=key_added)  # run
        fig = self.plot_spatial(color=key_added, **kwargs)  # plot
        if copy is False:
            self.rna = adata
            self._columns.update({"spatial_domains": key_added})
        return adata, fig

    def find_cooccurrence(self, col_cell_type=None, key_cell_type=None,
                          layer=None, library_id=None, cmap="magma",
                          n_jobs=None, figsize=15, palette=None, title=None,
                          kws_plot=None, out_plot=None,
                          shape="hex", size=None, **kwargs):
        """
        Find co-occurrence using spatial data. (similar to neighborhood
        enrichment analysis, but uses original spatial coordinates rather
        than connectivity matrix).
        """
        _ = self.get_layer(layer=layer, inplace=True)
        cct = col_cell_type if col_cell_type else self._columns[
            "col_cell_type"]
        self.rna.obs = self.rna.obs.astype({cct: "category"})
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
        kws_plot = cr.tl.merge(dict(
            palette=palette, size=size, key_cell_type=key_cell_type,
            figsize=figsize, out_file=out_plot), kws_plot)  # plot keywods
        if n_jobs == -1 or (isinstance(n_jobs, bool) and n_jobs is True):
            n_jobs = os.cpu_count() - 1  # threads for parallel processing
        print("\n\n<<< QUANTIFYING CO-OCCURRENCE >>>\n\n")
        sq.gr.co_occurrence(adata, cluster_key=cct, n_jobs=n_jobs,
                            spatial_key=self._spatial_key, **kwargs)
        fig["co_occurrence"] = {}
        if key_cell_type is not None:
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
        #     adata, cluster_key=cct, legend=False,
        #     clusters=key_cell_type, figsize=figsize)  # plot co-occurrrence
        try:
            fig["co_occurrence"] = cr.pl.plot_cooccurrence(
                adata.tables[self._keys["key_table"]] if isinstance(
                    adata, spatialdata.SpatialData) else adata,
                col_cell_type=cct, **kws_plot)  # lines plot
        except Exception:
            try:
                ann = adata.tables[self._keys["key_table"]].copy()
                _ = ann.uns.pop(f"{cct}_colors", None)  # Squidpy palette bug
                fig["co_occurrence"] = cr.pl.plot_cooccurrence(
                    ann, col_cell_type=cct, **kws_plot)  # lines plot
            except Exception:
                fig["co_occurrence"] = str(traceback.format_exc())
                print(traceback.format_exc())
        if title:
            fig["co_occurrence"].suptitle(title)
        print("\n\n*** Co-occurrence results stored in "
              f"`.rna.uns['{cct}_co_occurrence']`\n\n")
        return adata, fig

    def find_svgs(self, genes=10, method="moran", shape="hex", n_perms=10,
                  layer=None, library_id=None, col_cell_type=None, title=None,
                  col_sample_id=None, n_jobs=2, figsize=15,
                  key_image=None, out_plot=None, kws_plot=None):
        """Find spatially-variable genes."""
        adata = self.adata
        kws_plot = cr.tl.merge({"cmap": "magma", "use_raw": False}, kws_plot)
        cct = col_cell_type if col_cell_type else self._columns[
            "col_cell_type"]
        csid = col_sample_id if col_sample_id else self._columns[
            "col_sample_id"]
        figsize = (figsize, figsize) if isinstance(figsize, (
            int, float)) else (15, 7) if figsize is None else figsize
        print(f"\n\n<<< QUANTIFYING AUTO-CORRELATION ({method}) >>>\n\n")
        if n_jobs == -1 or (isinstance(n_jobs, bool) and n_jobs is True):
            n_jobs = os.cpu_count() - 1  # threads for parallel processing
        sq.gr.spatial_autocorr(adata, mode=method, layer=layer, n_jobs=n_jobs,
                               n_perms=n_perms)  # autocorrelation
        if isinstance(genes, int):
            genes = self.rna.uns["moranI"].head(genes).index.values
        # fig, ncols = None, cr.pl.square_grid(len(genes + [cct]))[1]
        if kws_plot is not False:
            try:
                fig = self.plot_spatial(
                    genes + [cct], shape=shape, figsize=figsize, title=title,
                    key_image=key_image, library_id=library_id,
                    library_key=csid, out_file=out_plot, **kws_plot)
            except Exception:
                fig = str(traceback.format_exc())
                traceback.print_exc()
        # sc.pl.spatial(adata, color=genes, library_id=library_id,
        #               figsize=figsize, **kws_plot)  # SVGs GEX plot
        print("\n\n*** SVG results stored in `.rna.uns['moranI']`\n\n")
        print(self.rna.uns["moranI"].head(10), "\n\n\n")
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

    # def run_misty(self, col_cell_type=None, p_threshold=0.05,
    #               organism="human", layer="log1p", top_n=500, seed=1618,
    #               k_fold=10, model="linear", bypass_intra=None, **kwargs):
    #     """Use Misty for receptor-ligand analysis on spatial data."""
    #     fig = {"dot": {}}
    #     if bypass_intra is None:
    #         bypass_intra = model == "linear"
    #     adata = self.get_layer(layer=layer, inplace=False)  # get layer copy
    #     if isinstance(adata, spatialdata.SpatialData):
    #         k_t = kwargs.pop("key_table", self._keys["key_table"])
    #         adata = adata.tables[k_t].copy()
    #     cct = col_cell_type if col_cell_type else self._columns[
    #         "col_cell_type"]  # cell type column
    #     c_cat = dict(zip(adata.obs[cct].unique(), adata.obs[cct].unique()))
    #     adata.obsm[cct].columns = [c_cat.get(c, c) for c in c_cat]
    #     comp = liana.ut.obsm_to_adata(adata, cct)  # types -> adata
    #     prog = dc.get_progeny(organism=organism, top=top_n)  # resource
    #     dc.run_mlm(mat=adata, vnet=prog, source="source", target="target",
    #                weight="weight", verbose=True, use_raw=False)  # MLM
    #     act = liana.ut.obsm_to_adata(adata, "mlm_estimate")  # activity
    #     kws = cr.tl.merge(dict(cutoff=p_threshold, bandwidth=200,
    #                            coord_type="generic", n_rings=1), kwargs)
    #     misty = genericMistyData(intra=comp, extra=act, **kws)  # Misty
    #     misty(verbose=True, model=model, k_cv=k_fold, seed=seed,
    #           bypass_intra=bypass_intra)  # run Misty model
    #     print(misty.uns["target_metrics"].head())
    #     for x in ["intra_R2", "gain_R2"]:
    #         fig["dot"][x] = liana.pl.target_metrics(
    #             misty, stat=x, return_fig=True)
    #     fig["contribute"] = liana.pl.contributions(misty, return_fig=True)
    #     print(misty.uns["interactions"].head())
    #     (
    #         liana.pl.interactions(misty, view="juxta", return_fig=True) +
    #         plotnine.scale_fill_gradient2(low="blue", mid="white",
    #                                       high="red", midpoint=0)
    #     )
    #     return misty, fig

    def calculate_receptor_ligand_spatial(self, col_cell_type=None,
                                          method="cosine", layer="log1p",
                                          genes=None, cmap="coolwarm",
                                          top_n=3, **kwargs):
        """Calculate receptor-ligand information using spatial data."""
        # raise NotImplementedError("Spatially-informed ligand-receptor "
        #                           "analysis not yet implemented.")
        adata = self.get_layer(layer=layer, inplace=False)  # get layer copy
        if isinstance(adata, spatialdata.SpatialData):
            k_t = kwargs.pop("key_table", self._keys["key_table"])
            adata = adata.tables[k_t].copy()  # anndata
        kws = cr.tl.merge(dict(
            n_perms=100, mask_negatives=False, add_categories=True,
            expr_prop=0.2, use_raw=False, verbose=True), kwargs, how="left")
        liana.mt.lr_bivar(adata, function_name=method, **kws)  # run
        lrd = adata.obsm["local_scores"].sort_values(
            "global_mean", ascending=False)
        for x in ["global_mean", "global_sd"]:
            print(x, lrd.var.sort_values(x, ascending=False).head(top_n))
        if genes:
            combs = [f"{g[0]}^{g[1]}" for g in permutations(genes, 2)]
            sc.pl.spatial(lrd.head(top_n), layer="cats", color=combs,
                          cmap=cmap, size=1.4)
        return adata, adata.obsm["local_scores"]

    def calculate_spatial_distance(self, key_reference, col_reference=None,
                                   genes=None, metric="euclidean",
                                   covariates=None, copy=False,
                                   layer="counts", figsize=None, **kwargs):
        """Calculate distance measurements given a reference point."""
        adata = self.get_layer(layer=layer, inplace=False)
        if col_reference is None:
            col_reference = self._columns["col_cell_type"]
        # cid = kwargs.pop("library_key", "cell_id")
        cid = kwargs.pop("col_sample_id", self._columns["col_sample_id"])
        key_added = kwargs.pop("key_added", "design_matrix")
        if covariates is True:  # can use default covariates if specify T
            covariates = []
            for x in ["col_condition", "col_sample_id"]:
                if self._columns[x]:
                    covariates += [self._columns[x]]
        sq.tl.var_by_distance(
            adata, groups=key_reference, cluster_key=col_reference,
            library_key=cid, design_matrix_key=key_added, metric=metric,
            covariates=covariates, spatial_key=self._spatial_key, copy=False)
        print(adata.obsm["design_matrix"].head(20))
        if genes is not None:
            sq.pl.var_by_distance(
                adata=adata, design_matrix_key=key_added, var=genes,
                anchor_key=key_reference,
                covariate=covariates[0] if covariates else None,
                color=covariates[1] if covariates else None,
                show_scatter=False, figsize=figsize, **kwargs)
        if copy is False:
            self.rna = adata
        return adata

    def plot_neighborhood(self, col_cell_type=None, **kwargs):
        """Plot results from neighborhood enrichment analysis."""
        fig = sq.pl.nhood_enrichment(self.rna, col_cell_type if (
            col_cell_type) else self._columns["col_cell_type"], **kwargs)
        return fig

    def plot_cooccurrence(self, col_cell_type=None, groups=None, **kwargs):
        """Plot co-occurrence results."""
        fig = cr.pl.plot_cooccurrence(
            self.rna, col_cell_type if col_cell_type else self._columns[
                "col_cell_type"], cluster_key=groups, **kwargs)
        return fig

    def plot_svgs(self, metrics="all", height=15, aspect=2,
                  x_label_rotation=45, key_added="moranI", **kwargs):
        """Plot SVG results."""
        if isinstance(metrics, str) or metrics is None:
            metrics = list(self.rna.uns[key_added].columns) if (
                metrics == "all") or metrics is None else [metrics]
        fig = {}
        print(self.rna.uns[key_added].head(10), "\n\n\n")
        for x in metrics:
            if x in self.rna.uns[key_added].columns:
                fig[x] = sb.catplot(self.rna.uns[key_added].rename_axis(
                    "Gene")[[x]], kind="bar", x="Gene", hue="Gene", y=x,
                                    height=height, aspect=aspect, **kwargs)
                if x_label_rotation not in [None, False]:
                    fig[x].set_xticklabels(rotation=x_label_rotation)
                fig[x].fig.suptitle("Spatially Variable Genes: "
                                    f"{'Moran I' if x == 'I' else x}")
        return fig

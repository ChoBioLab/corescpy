#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Importing data.

@author: E. N. Aslinger
"""

import os
import re
import warnings
import scanpy as sc
import pandas as pd
import numpy as np
import corescpy as cr

FILE_STRUCTURE = ["matrix", "cells", "genes"]
DEF_FILE_P = "crispr_analysis/protospacer_calls_per_cell.csv"


def name_path_iterative(path):
    """Check for existence of path and, if needed, make path_#
    to avoid overwriting.

    Args:
        path (str): Path to check & correct for redundancy.

    Returns:
        str: Path with iterative suffixes _1, _2, etc. as needed.
    """
    if os.path.exists(path):
        path, ext = os.path.splitext(path)
        warnings.warn(f"{path} already exists.")
        try:  # check if path suffix (after _) coercible to int
            suff = int(path.split("_")[-1])
            path = path.split("_")[:-1]
        except ValueError:
            suff = 0
        path = f"{path}_{suff + 1}{ext}"
    if os.path.exists(path):
        path = name_path_iterative(path)
    return path


def process_paths(files=None, directory_in=None):
    """Construct paths to data and check validity."""

    # Process provided paths
    if directory_in and not os.path.isdir(directory_in):  # directory exists?
        raise ValueError(f"Directory {files} does not exist.")
    if files:
        if not isinstance(files, list):
            if isinstance(files, (set, tuple, np.ndarray)):  # if list-like...
                files = list(files)  # ...convert to list
            else:
                raise ValueError("Files must be list of paths to the files.")
        if directory_in:  # if directory provided too, ensure files relative
            if any((directory_in in f for f in files)):
                warnings.warn("""If directory_in provided,
                              ensure files are RELATIVE paths
                              (i.e., file.ext instead of directory/file.ext)
                              to avoid the code duplicating it
                              (e.g., directory/directory/file.ext).""")
    elif directory_in:  # if directory given in lieu of files...
        files = os.listdir(directory_in)  # list files in directory
    else:
        raise ValueError("Either files or directory must be provided.")
    if directory_in:  # join file paths with directory_in if provided
        files = [os.path.join(directory_in, f) for f in files]
    files_original = files.copy()  # store original paths in case any invalid

    # Ensure valid paths
    files = [f for f in files if os.path.isfile(f)]  # ensure valid paths
    if set(files).intersection(files_original) != set(files):
        invalid_files = set
        raise ValueError(f"Some files invalid! {' , '.join(invalid_files)}")

    return files


def create_subdirectories(files=None, directory_in=None, strip_strings=None,
                          # sep="_",
                          dir_strip=True, unzip=False,
                          directory_out=None, overwrite=False):
    """_summary_

    Args:
        files (_type_, optional): _description_. Defaults to None.
        directory_in (_type_, optional): _description_. Defaults to None.
        strip_strings (_type_, optional): _description_. Defaults to None.
        dir_strip (bool, optional): _description_. Defaults to True.
        unzip (bool, optional): _description_. Defaults to False.
        directory_out (_type_, optional): _description_. Defaults to None.
        overwrite (bool, optional): _description_. Defaults to False.
    """

    # Input paths & output directory
    paths = process_paths(files=files, directory_in=directory_in)
    if not directory_out:  # if directory not provided, assume same as in
        directory_out = directory_in
        if overwrite is False:  # if no overwrite & output=input directory...
            directory_out = name_path_iterative(
                os.path.join(directory_in, "organized"))  # subdirectory
    # DO NOT MODIFY "paths" OR "directory_out" AFTER THIS POINT

    # Create directories
    files_out = dict(zip(paths, [os.path.basename(f) for f in paths]))
    if not os.path.isdir(directory_out):
        os.makedirs(directory_out)  # make output directory if need

    # File stems (without extensions, by person/sample, etc.)
    stems = dict(file=dict(zip(paths, [os.path.splitext(os.path.basename(
        files_out[f] if f[-3:] != ".gz" else files_out[f][:-3]))[0]
                       for f in files_out])))  # w/o extensions

    # Strip any junk strings from unconventional file naming
    if strip_strings:
        if isinstance(strip_strings, str):
            strip_strings = [strip_strings]  # ensure list if only 1
        for i in strip_strings:
            stems["file"] = dict(zip(stems["file"], [re.sub(
                i, "", stems["file"][f]) for f in stems["file"]]))  # rm strs
    # ids = pd.unique([stems["file"][f] for f in stems["file"]])
    # non_redundant_parts = pd.DataFrame(
    #     [os.path.basename(f).split(sep) for f in stems["file"].values()],
    #     index=stems["file"].keys()).apply(
    #         lambda x: pd.Series([np.nan] * len(x), index=x.index,
    #                             name=x.name) if len(x.unique()) == 1
    #         else x).dropna(how="all", axis=1).apply(lambda y: sep.join(
    #             list(y.dropna())), axis=1)

    # Create sub-directories & move or copy files
    dir_sub = dict(zip(files_out, [
        os.path.join(directory_out, stems["file"][i]) for i in files_out]))
    for d in pd.unique(list(dir_sub.values())):
        if not os.path.exists(d):
            os.mkdir(d)
    for f in files_out:
        if dir_strip is True:
            new_path = re.sub("^_", "", re.sub("^[.]", "", re.sub(
                os.path.basename(dir_sub[f]), "", files_out[f])))
            new_path = os.path.join(dir_sub[f], new_path)
        else:
            new_path = os.path.join(dir_sub[f], files_out[f])
        if overwrite is False:
            new_path = name_path_iterative(new_path)  # new path
        new_path = re.sub("cells.tsv", "barcodes.tsv", re.sub(
            "genes.tsv", "features.tsv", new_path))  # fix unconventional
        os.system(f"{'cp' if overwrite is False else 'mv'} {f} {new_path}")
        if os.path.splitext(new_path)[-1] == ".gz" and unzip is True:
            os.system(f"gunzip {new_path}")  # unzip if needed


# def get_matrix_from_h5(file, gex_genes_return=None):
#     """Get matrix from 10X h5 file (modified from 10x code)."""
#     FeatureBCMatrix = collections.namedtuple("FeatureBCMatrix", [
#         "feature_ids", "feature_names", "barcodes", "matrix"])
#     with h5py.File(file) as f:
#         if u"version" in f.attrs:
#             version = f.attrs["version"]
#             if version > 2:
#                 print(f"Version = {version}")
#                 raise ValueError(f"HDF5 format version version too new.")
#         else:
#             raise ValueError(f"HDF5 format version ({version}) too old.")
#         feature_ids = [x.decode("ascii", "ignore")
#                        for x in f["matrix"]["features"]["id"]]
#         feature_names = [x.decode("ascii", "ignore")
#                          for x in f["matrix"]["features"]["name"]]
#         barcodes = list(f["matrix"]["barcodes"][:])
#         matrix = sp_sparse.csr_matrix((f["matrix"]["data"],
#                                        f["matrix"]["indices"],
#                                        f["matrix"]["indptr"]),
#                                       shape=f["matrix"]["shape"])
#         fbm = FeatureBCMatrix(feature_ids, feature_names, barcodes, matrix)
#         if gex_genes_return is not None:
#             gex = {}
#             for g in gex_genes_return:
#                 try:
#                     gene_index = fbm.feature_names.index(g)
#                 except ValueError:
#                     raise Exception(f"{g} not found in list of gene names.")
#                 gex.update({g: fbm.matrix[gene_index, :].toarray(
#                     ).squeeze()})  # gene expression
#         else:
#             gex = None
#         barcodes = [x.tostring().decode() for x in fbm.barcodes]
#         genes = pd.Series(fbm.feature_names).to_frame("gene").join(
#             pd.Series(fbm.feature_ids).to_frame("gene_ids"))
#     return fbm, gex, barcodes, genes


def combine_matrix_protospacer(directory="",
                               subdirectory_mtx="filtered_feature_bc_matrix",
                               file_protospacer=DEF_FILE_P, prefix=None,
                               col_gene_symbols="gene_symbols",
                               col_barcode="cell_barcode", gex_only=False,
                               drop_guide_capture=True, **kwargs):
    """
    Join CellRanger directory-derived AnnData `.obs` & perturbations.

    Example
    -------
    >>> data_dir = "/home/asline01/projects/corescpy/examples/data"
    >>> f_p = "crispr_analysis/protospacer_calls_per_cell.csv"
    >>> adata = combine_matrix_protospacer(
    ... f"{data_dir}/crispr-screening/HH03",
    ... "filtered_feature_bc_matrix", col_gene_symbols="gene_symbols",
    ... file_protospacer=f_p, col_barcode="cell_barcode")

    Or using create_object(), with directory/file-related arguments in
    a dictionary passed to the "file" argument:

    >>> data_dir = "/home/asline01/projects/corescpy/examples/data"
    >>> adata = create_object(
    ... dict(directory=f"{data_dir}/crispr-screening/HH03",
    ... subdirectory_mtx="filtered_feature_bc_matrix",
    ... file_protospacer=f_p, col_barcode="cell_barcode",
    ... col_gene_symbols="gene_symbols")

    """
    adata = sc.read_10x_mtx(
        os.path.join(directory, subdirectory_mtx),
        var_names=col_gene_symbols, gex_only=gex_only, prefix=prefix,
        **kwargs)  # read 10x matrix, barcodes, features files
    dff = pd.read_csv(os.path.join(directory, file_protospacer),
                      index_col=col_barcode)  # perturbation information
    if col_barcode is None:
        dff, col_barcode = dff.set_index(dff.columns[0]), dff.columns[0]
    if drop_guide_capture:
        adata = adata[:, adata.var["feature_types"] != "CRISPR Guide Capture"]
    adata.obs = adata.obs.join(dff.rename_axis(adata.obs.index.names[0]))
    return adata


def construct_file(run=None, panel_id="TUQ97N", directory=None):
    """Construct file path from information."""
    if "outputs" not in directory and os.path.exists(
            os.path.join(directory, "outputs")):
        directory = os.path.join(directory, "outputs")
    run = None if run is None else [run] if isinstance(run, str) else run
    if isinstance(panel_id, str):
        if run is None:
            run = [j for j in os.listdir(os.path.join(
                directory, panel_id)) if os.path.isdir(os.path.join(
                    directory, panel_id, j))]
        panel_id = [panel_id] * len(run)
    else:
        if run is None:
            run = []
            for x in panel_id:
                run += [j for j in os.listdir(os.path.join(
                    directory, x)) if os.path.isdir(j)]
    fff = []
    for i, x in enumerate(run):
        d_x = os.path.join(directory, panel_id[i], x)
        fff += [os.path.join(d_x, y) for y in os.listdir(
            d_x) if os.path.isdir(os.path.join(d_x, y))]
    return fff


def process_multimodal_crispr(adata, assay=None, col_guide_rna="guide_ids",
                              col_num_umis="num_umis", feature_split="|",
                              guide_split="-", keep_extra_columns=True):
    """
    Merge gRNA counts stored in a separate assay into RNA assay.

    Specify the "assay" argument as a list with 1st and 2nd elements
    corresponding to the modality labels (keys in `adata.mod`) for
    gene expression (RNA) and CRISPR guide data, respectively.

    """
    if assay is None:
        assay = ["rna", "gdo"]  # default assay labels
    rna, perturb = assay  # GEX & CRISPR counts assay labels
    gdo = adata.mod[perturb]
    gdo.layers["counts"] = gdo.X.copy()
    umis = pd.DataFrame(adata.mod[perturb].X.toarray(), columns=gdo.var_names,
                        index=gdo.obs.index)  # make gRNA counts dataframe
    guides = umis.apply(
        lambda x: feature_split.join([str(i) for i in np.array(
            umis.columns)[np.where(x != 0)[0]]]) if any(
                x != 0) else np.nan, axis=1)
    nums = umis.apply(
        lambda x: feature_split.join([str(i) for i in np.array(
            x)[np.where(x != 0)[0]]]) if any(x != 0) else np.nan, axis=1)
    umis = guides.to_frame(col_guide_rna).join(nums.to_frame(col_num_umis))
    adata.mod[rna].obs = adata.mod[rna].obs.join(umis)
    if keep_extra_columns is True:  # join columns in gdo not present in adata
        adata.mod[rna].obs = adata.mod[rna].obs.join(gdo.obs[
            gdo.obs.columns.difference(adata.mod[rna].obs.columns)])
    return adata


def get_metadata_cho(directory, file_metadata, panel_id="TUQ97N",
                     run=None, samples=None, capitalize_sample=False,
                     path_col_only=False):
    """Retrieve Xenium metadata."""
    # Get Column & Key Names from Constants Script
    constant_dict = {**cr.get_panel_constants(panel_id=panel_id)}  # constants
    col_sample_id, col_sample_id_o, col_slide, col_condition, col_data_dir = [
        constant_dict[x] if (x in constant_dict) else None for x in [
            "col_sample_id", "col_sample_id_o", "col_slide",
            "col_condition", "col_data_dir"]]

    # Read Metadata
    metadata = (pd.read_excel if os.path.splitext(file_metadata)[
        1] == ".xlsx" else pd.read_csv)(file_metadata, dtype={
            col_slide: str} if col_slide else None)  # read metadata
    if col_sample_id_o != col_sample_id:  # construct <condition>-<block> ID?
        metadata.loc[:, col_sample_id] = metadata[
            col_condition].apply(lambda x: str(x).capitalize() if (
                capitalize_sample is True) else x) + "-" + metadata[
                    col_sample_id_o].astype(str)  # combine condition & block
    metadata = metadata.set_index(col_sample_id)

    # Find File Paths
    if os.path.basename(directory) == panel_id:  # already panel sub-directory
        directory = os.path.dirname(directory)  # get parent directory
    fff = np.array(cr.pp.construct_file(run=run, directory=directory,
                                        panel_id=panel_id))
    bff = np.array([os.path.basename(i) for i in fff])  # base path names
    samps = np.array([i.split("__")[2].split("-")[0] for i in fff])
    for x in metadata[col_sample_id_o]:
        if col_data_dir is not None and col_data_dir in metadata.columns:
            if "outputs" not in directory and os.path.exists(
                    os.path.join(directory, "outputs")):
                direc = os.path.join(directory, "outputs")
            else:
                direc = directory
            m_f = metadata[metadata[col_sample_id_o] == x][col_data_dir].iloc[
                0]  # ...to find manually-defined unconventionally-named files
            if pd.isnull(m_f):
                m_f = None
            else:
                poss = [os.path.join(direc, panel_id, q, m_f)
                        for q in os.listdir(os.path.join(direc, panel_id))]
                poss = np.array(poss)[np.where([os.path.exists(
                    q) for q in poss])[0]] if any([os.path.exists(
                        q) for q in poss]) else None
                if poss is not None and len(poss) > 1:
                    raise ValueError(f"Multiple possible paths ({x})\n{poss}")
                m_f = None if poss is None or len(poss) == 0 else poss[0]
            # in case relative path or other description
        else:
            m_f = np.nan
        locx = np.where(samps == x)[0] if pd.isnull(
            m_f) or m_f is None else np.where(bff == os.path.basename(m_f))[0]
        metadata.loc[metadata[col_sample_id_o] == x, col_data_dir] = fff[
            locx[0]] if (len(locx) > 0) else np.nan  # output file for row
    metadata = metadata.dropna(subset=[col_data_dir]).reset_index(
        ).drop_duplicates().set_index(col_sample_id)
    if samples not in ["all", None]:  # subset by sample ID?
        if isinstance(samples, str):
            samples = [samples]
        if samples[0] in metadata[col_sample_id_o].to_list():
            metadata = metadata.reset_index().set_index(col_sample_id_o).loc[
                samples].reset_index().set_index(col_sample_id)
        else:
            metadata.loc[samples]
    return metadata[col_data_dir] if path_col_only is True else metadata

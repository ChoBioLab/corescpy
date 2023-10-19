#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import os
import re
import warnings
import h5py
import scipy.sparse as sp_sparse
import collections
import scanpy as sc
import pandas as pd
import numpy as np

FILE_STRUCTURE = ["matrix", "cells", "genes"]


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
                raise ValueError("Files must be a list of paths to the files.")
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
                          dir_strip = True,
                          unzip=False,
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
        if overwrite is False:  # if no overwrite & output = input directory...
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
                i, "", stems["file"][f]) for f in stems["file"]]))  # rm strs if need
    # ids = pd.unique([stems["file"][f] for f in stems["file"]])
    # non_redundant_parts = pd.DataFrame(
    #     [os.path.basename(f).split(sep) for f in stems["file"].values()],
    #     index=stems["file"].keys()).apply(
    #         lambda x: pd.Series([np.nan] * len(x), index=x.index,
    #                             name=x.name) if len(x.unique()) == 1 
    #         else x).dropna(how="all", axis=1).apply(lambda y: sep.join(
    #             list(y.dropna())), axis=1)
    
    # Create sub-directories & move or copy files
    dir_sub = dict(zip(files_out, [os.path.join(directory_out, stems["file"][i]) 
                                   for i in files_out]))
    for d in pd.unique(list(dir_sub.values())):
        if not os.path.exists(d):
            os.mkdir(d)
    for f in files_out:
        if dir_strip is True:
            new_path = os.path.join(dir_sub[f], re.sub("^_", "", re.sub(
                "^[.]", "", re.sub(
                os.path.basename(dir_sub[f]), "", files_out[f]))))
        else:
            new_path = os.path.join(dir_sub[f], files_out[f])
        if overwrite is False:
            new_path = name_path_iterative(new_path)  # new path
        new_path = re.sub("cells.tsv", "barcodes.tsv", re.sub(
            "genes.tsv", "features.tsv", new_path))  # fix unconventional naming
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


def combine_matrix_protospacer(
    directory="", subdirectory_mtx="filtered_feature_bc_matrix", 
    col_gene_symbols="gene_symbols", 
    file_protospacer="crispr_analysis/protospacer_calls_per_cell.csv", 
    col_barcode="cell_barcode", 
    **kwargs):
    """
    Combine CellRanger directory-derived AnnData `.obs` & perturbation data.
    
    Example
    -------
    >>> data_dir = "/home/asline01/projects/crispr/examples/data"
    >>> adata = combine_matrix_protospacer(
    ... f"{data_dir}/crispr-screening/HH03", 
    ... "filtered_feature_bc_matrix", col_gene_symbols="gene_symbols", 
    ... file_protospacer="crispr_analysis/protospacer_calls_per_cell.csv", 
    ... col_barcode="cell_barcode")
    
    Or using create_object(), with directory/file-related arguments in 
    a dictionary passed to the "file" argument:
    
    >>> data_dir = "/home/asline01/projects/crispr/examples/data"
    >>> adata = create_object(
    ... dict(directory=f"{data_dir}/crispr-screening/HH03", 
    ... subdirectory_mtx="filtered_feature_bc_matrix", 
    ... file_protospacer="crispr_analysis/protospacer_calls_per_cell.csv"),
    ... col_barcode="cell_barcode", col_gene_symbols="gene_symbols")
    
    """
    adata = sc.read_10x_mtx(
        os.path.join(directory, subdirectory_mtx), 
        var_names=col_gene_symbols, **kwargs)  # 10x matrix, barcodes, features
    dff = pd.read_csv(os.path.join(directory, file_protospacer), 
                      index_col=col_barcode)  # perturbation information
    if col_barcode is None:
        dff, col_barcode = dff.set_index(dff.columns[0]), dff.columns[0]
    adata.obs = adata.obs.join(dff.rename_axis(adata.obs.index.names[0]))
    return adata
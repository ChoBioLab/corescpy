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
import pandas as pd
import numpy as np

strip_strings = [".protospacer_calls", ".genes", ".matrix", ".cells"]
strip_strings = [".protospacer_calls", ".genes", ".matrix", ".cells", "MOLM13"]


def load_data(file):
    """Load CRISPR data."""
    data = pd.read_csv(file)
    return data


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


def create_subdirectories(files=None, directory_in=None, strip_strings=None, sep="_",
                          directory_out=None, overwrite=False):
    
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
    elif overwrite is False:
        raise Warning(f"Directory {directory_out} already exists.")
        # check if any output files already in existing directory_out
        
    # Strip any junk strings from unconventional file naming
    if strip_strings:  
        if isinstance(strip_strings, str): 
            strip_strings = [strip_strings]  # ensure list if only 1
        for i in strip_strings:
            files_out = dict(zip(files_out, [re.sub(
                i, "", files_out[f]) for f in files_out]))  # rm strs if need
    # DO NOT MODIFY "paths_out" AFTER THIS POINT
            
    # File stems (without extensions, by person/sample, etc.)
    stems = dict(file=dict(zip(paths, [os.path.splitext(os.path.basename(
        files_out[f] if f[-3:] != ".gz" else files_out[f][:-3]))[0] 
                       for f in files_out])))  # w/o extensions
    # ids = pd.unique([stems["file"][f] for f in stems["file"]])
    # non_redundant_parts = pd.DataFrame(
    #     [os.path.basename(f).split(sep) for f in stems["file"].values()],
    #     index=stems["file"].keys()).apply(
    #         lambda x: pd.Series([np.nan] * len(x), index=x.index,
    #                             name=x.name) if len(x.unique()) == 1 
    #         else x).dropna(how="all", axis=1).apply(lambda y: sep.join(
    #             list(y.dropna())), axis=1)
    
    # Create sub-directories & move or copy files
    dir_sub = dict(zip(files_out, [os.path.join(directory_out, stems["file"][i]) for i in files_out]))
    for d in pd.unique(list(dir_sub.values())):
        if not os.path.exists(d):
            os.mkdir(d)
    for f in files_out:
        new_path = os.path.join(dir_sub[f], files_out[f])
        if overwrite is False:
            new_path = name_path_iterative(new_path)  # new path
        os.system(f"{'cp' if overwrite is False else 'mv'} {f} {new_path}")
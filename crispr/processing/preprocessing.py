#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import pertpy as pt
import pandas as pd
import numpy as np

OPTS_OBJECT_TYPE = ["augur"]


def create_object(adata, object_type="augur"):
    """Create object(s) from adata."""
    
    # Check validity of object_type argument & modify as needed
    if isinstance(object_type, str):
        object_type = [object_type]
    if isinstance(object_type, (list, tuple, np.ndarray, set)):
        for i, j in enumerate(object_type):
            if not isinstance(j, str):
                raise ValueError(f"""object_type elements must be strings.""")
    else:
        raise ValueError("object_type must be a string or list of strings.")
    object_type = [t.lower() for t in object_type]  # convert to lowercase
    
    # Convert/create objects
    objects = [np.nan] * len(object_type)  # initialize empty list for output
    for i, t in enumerate(object_type):  # iterate object_types
        if t in OPTS_OBJECT_TYPE:  # if valid object type option, convert
            obj = adata
            objects[i] = obj
        else:  # if invalid object type option, warn & leave as nan in list
            raise Warning(f"""object_type {t} invalid or not yet implemented. 
                          Options: {OPTS_OBJECT_TYPE}.""")
    return objects
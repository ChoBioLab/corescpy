#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
@author: E. N. Aslinger
"""

import pandas as pd


def to_list(arg, unique=False):
    """
    Make argument a list whether it's a string or list-like.
    If it's a dictionary, return the items of the dictionary.
    If None, leave as None.
    """
    if arg is None:
        return arg  # leave as None if None
    arg = [arg] if isinstance(arg, str) else [arg[k] for k in arg] if (
        isinstance(arg, dict)) else list(arg)
    if unique is True:
        arg = pd.unique(arg)
    return arg


def merge(arg_override=None, arg_fill_in=None, how="union"):
    """
    Merge two dictionaries, giving precedence to
    arg_override in case of overlap. Use empty dictionary in place of "None."
    """
    arg_override, arg_fill_in = [{**x} if x else {}
                                 for x in [arg_override, arg_fill_in]]
    joined = {**arg_override, **arg_fill_in}
    if how.lower() == "left":  # only include keys in `arg_override`
        joined = dict(zip(arg_override.keys(), [joined[x]
                                                for x in arg_fill_in]))
    elif how.lower() == "right":  # only include keys in `arg_override`
        joined = dict(zip(arg_fill_in.keys(), [joined[x]
                                                for x in arg_fill_in]))
    elif how.lower() == "union":
        pass
    else:
        raise ValueError("`how` must be 'union', 'left', or 'right'")
    return joined

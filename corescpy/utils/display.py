import re
import h5py
import spatialdata
import pandas as pd
import numpy as np


def print_dictionary(dct, show=True):
    """Print a dictionary to allow copy-pasting code for object assignment."""
    text = make_printable_object(print_pretty_dictionary(
        dct, show=False), show=False)
    if "dict(" == text[:5]:
        text = re.sub("^dict.", "", text)[:-1]
    text = "\n".join(text.split(", "))
    if show is True:
        print(text)
    else:
        return text


def make_printable_object(obj, show=False, numpy_alias="np"):
    """Replace `np.nan` in printed output."""
    if isinstance(obj, (np.ndarray, list)):
        text = "[" + ", ".join([make_printable_object(i)
                                for i in obj]) + "]"
        if isinstance(obj, np.ndarray):
            text = f"np.ndarray({text})"
    elif isinstance(obj, (str, float, int)):
        text = "np.nan" if obj == "nan" else f"'{obj}'" if isinstance(
            obj, str) else str(obj)
    elif isinstance(obj, dict):
        if show is True:
            text = dict(zip(obj, [print_pretty_dictionary(text, show=False)
                                  if isinstance(obj[i], dict) else obj[i]
                                  for i in obj]))
            text = print_pretty_dictionary(text, show=False)
        else:
            # text = "dict(" + ", ".join(
            #     [f"{i} = "  + ["", "'"][int(isinstance(
            #         obj[i], str))] + str(obj[i]) + ["", "'"][
            #             int(isinstance(obj[i], str))]
            #         for i in obj]) + ")"
            text = "dict(" + ", ".join(
                [f"{i} = {make_printable_object(obj[i], show=False)}"
                 for i in obj]) + ")"
    elif obj in [None, True, False]:
        text = str(obj)
    elif pd.isnull(obj):
        text = f"{numpy_alias}.nan"
    else:
        raise TypeError(f"Type of object {type(obj)} not supported.")
    if show is True:
        print(text)
    text = re.sub("'", "\"", text)
    text = re.sub(" = ", "=", text)
    return text


def print_pretty_dictionary(dct, show=True, numpy_alias="np"):
    """Print a dictionary to allow copy-pasting code for object assignment."""
    print("\n\n")
    if isinstance(dct, str):
        # items = re.sub(", ", ",", dct).split(",")
        # if "dict(" == dct[:5]:
        #     text = dct
        # else:
        #     if items[0][0] == "{":
        #         items[0] = items[0][1:]  # remove leading brace
        #     if items[-1][-1] == "}":
        #         items[-1] = items[-1][:-1]  # remove clsoing brace
        #     items = [re.sub(": ", ":", i).split(":") for i in items]
        #     items = [[re.sub('"', "", re.sub("'", "", i[0])), i[1]]
        #              for i in items]
        #     text = "dict(" + ", ".join([f"{x[0]} = {x[1]}"
        #                                 for x in items]) + ")"
        text = dct
    else:
        text = "{" + ", ".join([
            f"{i} = {make_printable_object(dct[i], show=False)}"
            if isinstance(dct[i], dict) else
            f"'{i}': {make_printable_object(dct[i], show=False)}"
            for i in dct]) + "}"
    if show is True:
        # text = "\n".join(
        #     [f"{i} = None" if dct[i] is None else str(
        #         f"{i} = "  + ["", "'"][int(isinstance(dct[i], str))] + str(
        #             f"{numpy_alias}.{dct[i]}" if pd.isnull(
        #                 dct[i]) else dct[i]) + ["", "'"][
        #                     int(isinstance(dct[i], str))])
        #      for i in dct])
        text = "\n".join(
            [f"{i}={make_printable_object(dct[i])}" for i in dct])
        print(text)
    else:
        return text


def explore_h5_file(file):
    """Explore an H5 file's format (thanks to ChatGPT)."""
    with h5py.File(file, "r") as h5_file:
        top_level_groups = list(h5_file.keys())
        for group_name in top_level_groups:
            print(f"Group: {group_name}")
            for g in h5_file[group_name]:
                print(f"  Dataset: {g}")


def print_counts(adata, group_by=None, title="Total", **kwargs):
    if kwargs:
        pass
    adata = (adata.table if isinstance(adata, spatialdata.SpatialData
                                       ) else adata).copy()
    print(f"\n\n{'=' * 80}\nCounts: {title}\n{'=' * 80}\n")
    print(f"\n\tObservations: {adata.n_obs}\n")
    if group_by is not None and group_by in adata.obs:
        print(f"{'-' * 40}\nLAYER DIMENSIONS:\n{'-' * 40}")
        for x in adata.layers:
            print(f"{x}: {adata.layers[x].shape}")
        print(f"{'-' * 40}\n")
        if group_by is not None and group_by in adata.obs:
            print("\n", adata.obs[group_by].value_counts().round(2))
        print("\n")
    if "var" in dir(adata):
        print(f"\tGenes: {adata.n_vars}\n")
        des = adata.var.reset_index().describe()
        des = des.loc[list(set(["25%", "50%", "75%"]).intersection(
            des.index))].sort_index()
        if des.empty is False:
            print(des)
    print(f"\n\n{'=' * 80}\n")

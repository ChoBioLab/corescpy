import re
import pandas as pd
import numpy as  np


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
    elif isinstance(obj, str):
        text = "np.nan" if obj == "nan" else f"'{obj}'"
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
    elif pd.isnull(obj):
        text = f"{numpy_alias}.nan"
    elif obj in [None, True, False]:
        text = str(obj)
    else:
        raise TypeError(f"Type of object {type(obj)} not supported.")
    if show is True:
        print(text)
    return text


def print_pretty_dictionary(dct, show=True, numpy_alias="np"):
    """Print a dictionary to allow copy-pasting code for object assignment.""" 
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
            [f"{i}={make_printable_object(dct[i])}"
             for i in dct])
        print(text)
    else:
        return text
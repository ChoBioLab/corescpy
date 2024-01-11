import crispr as cr
import decoupler as dc
import scanpy as sc
import pandas as pd
import numpy as np

layers = cr.pp.get_layer_dict()


def create_pseudobulk(adata, col_cell_type, col_sample_id=None, 
                      layer=layers["counts"], mode="sum", 
                      kws_process=True, **kwargs):
    """Get pseudo-bulk of scRNA-seq data."""
    if kws_process is True:
        kws_process = dict(target_sum=1e6, max_value=10, n_comps=10)
    if layer and layer not in adata.layers:
        raise ValueError(f"{layer} not in adata.layers. Set layer argument to"
                         " None or the name of the layer with count data.")
    pdata = dc.get_pseudobulk(
        adata, sample_col=col_sample_id, groups_col=col_cell_type, 
        layer=layer, mode=mode, **kwargs)
    pdata.layers["counts"] = pdata.X.copy()
    if kws_process is True or isinstance(kws_process, dict):
        sc.pp.normalize_total(pdata, target_sum=kws_process["target_sum"])
        sc.pp.log1p(pdata)
        sc.pp.scale(pdata, max_value=kws_process["max_value"])
        sc.tl.pca(pdata, n_comps=kws_process["n_comps"])
    return pdata
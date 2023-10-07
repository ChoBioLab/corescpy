#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing CRISPR experiment data.

@author: E. N. Aslinger
"""

import os
import re
import scanpy as sc
import pertpy as pt
import muon
import warnings
import scipy
import seaborn
import collections      
import scipy.sparse as sp_sparse
import h5py
import anndata
import pandas as pd
import numpy as np
# from crispr.defaults import names_layers

regress_out_vars = ["total_counts", "pct_counts_mt"]

def create_object(file, col_gene_symbols="gene_symbols", assay=None,
                  col_barcode=None, **kwargs):
    """Create object from Scanpy- or Muon-compatible file."""
    # extension = os.path.splitext(file)[1]
    if isinstance(file, (str, os.PathLike)) and os.path.splitext(
        file)[1] == ".h5mu":  # MuData
        print(f"\n<<< LOADING FILE {file} with muon.read()>>>")
        adata = muon.read(file)
    elif isinstance(file, dict):
        adata = combine_matrix_protospacer(
            **file, col_gene_symbols=col_gene_symbols, col_barcode=col_barcode,
            **kwargs)  # when perturbation info not in mtx
    elif not isinstance(file, (str, os.PathLike)):
        print(f"\n<<< LOADING OBJECT>>>")
        adata = file.copy()
    elif os.path.isdir(file):  # if directory, assume 10x format
        print(f"\n<<< LOADING 10X FILE {file}>>>")
        adata = sc.read_10x_mtx(file, var_names=col_gene_symbols, cache=True,
                                **kwargs)  # 10x matrix, barcodes, features
    elif os.path.splitext(file)[1] == ".h5":
        print(f"\n<<< LOADING 10X .h5 FILE {file}>>>")
        print(f"H5 File Format ({file})\n\n")
        explore_h5_file(file, "\n\n\n")
        adata = sc.read_10x_h5(file, **kwargs)
    else:
        print(f"\n<<< LOADING FILE {file} with sc.read()>>>")
        adata = sc.read(file)
    # TODO: Flesh this out, generalize, test, etc.
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    if col_gene_symbols not in adata.var.columns:
        adata.var = adata.var.rename_axis(col_gene_symbols)
    if assay is not None:
        adata = adata[assay]  # subset by assay if desired 
    print("\n\n", adata)
    return adata


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
    >>> adata = combine_matrix_protospacer(
    ... "/home/asline01/projects/crispr/examples/data/crispr-screening/HH03, 
    ... "filtered_feature_bc_matrix", col_gene_symbols="gene_symbols", 
    ... file_protospacer="crispr_analysis/protospacer_calls_per_cell.csv", 
    ... col_barcode="cell_barcode")
    
    Or using create_object(), with directory/file-related arguments in 
    a dictionary passed to the "file" argument:
    
    >>> adata = create_object(
    ... dict(directory="/home/asline01/projects/crispr/examples/data/crispr-screening/HH03, 
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
    

def process_data(adata, assay=None, assay_protein=None,
                 col_gene_symbols=None,
                 col_cell_type=None,
                 remove_doublets=True,
                 target_sum=1e4,  max_genes_by_counts=2500, max_pct_mt=5,
                 min_genes=200, min_cells=3, 
                 scale=10,  # or scale=True for no clipping
                 regress_out=regress_out_vars, 
                 kws_hvg=None, kws_scale=None, kws_crispr=None,
                 **kwargs):
    """
    Preprocess data (plus CRISPR-specific steps if kws_crispr is not None).
    """
    
    # Initial Information
    print(adata)
    if col_gene_symbols == adata.var.index.names[0]:
        col_gene_symbols = None
    if col_cell_type is not None and col_cell_type in adata.obs:
        print(f"\n\n{'=' * 80}\nCell Counts\n{'=' * 80}\n\n")
        print(adata.obs[col_cell_type].value_counts())
    figs = {}
    n_top = kwargs.pop("n_top") if "n_top" in kwargs else 20
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    print(col_gene_symbols, assay, n_top)
        
    # Doublets
    # doublets = detect_doublets(adata[assay] if assay else adata)
    # if remove_doublets is True:
    #     adata
    #     # TODO: doublets
        
    
    # Highly-Expressed Genes
    figs["highly_expressed_genes"] = sc.pl.highest_expr_genes(
        adata[assay] if assay else adata, n_top=n_top,
        gene_symbols=col_gene_symbols)
    if kws_scale is None:
        kws_scale = {}

    # Filtering
    print("\n<<< FILTERING >>>")
    sc.pp.filter_cells(adata[assay] if assay else adata, min_genes=min_genes)
    sc.pp.filter_genes(adata[assay] if assay else adata, min_cells=min_cells)
    
    # Mitochondrial Count QC
    print("\n<<< DETECTING MITOCHONDRIAL GENES >>>")
    no_mt = False
    if assay is None:
        adata.var["mt"] = adata.var_names.str.startswith(
            "MT-")  # annotate mitochondrial genes
    else:
        try:
            adata[assay].var["mt"] = adata[assay].var_names.str.startswith(
                "MT-")  # annotate mitochondrial genes
        except TypeError as err_mt:
            warnings.warn(f"\n\n{'=' * 80}\n\nCould not assign MT: {err_mt}")
            no_mt = True
    
    # Quality Control
    print("\n<<< PERFORMING QUALITY CONTROL >>>")
    if no_mt is False:
        sc.pp.calculate_qc_metrics(adata[assay] if assay else adata, 
                                   qc_vars=["mt"], percent_top=None, 
                                   log1p=True, inplace=True)
        try:
            figs["qc_pct_counts_mt_hist"] = seaborn.histplot(
                adata[assay].obs["pct_counts_mt"] if assay else adata.obs[
                    "pct_counts_mt"])
        except Exception as err:
            print(err)
        try:
            figs["qc_metrics_violin"] = sc.pl.violin(
                adata[assay] if assay else adata, 
                ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                jitter=0.4, multi_panel=True)
        except Exception as err:
            print(err)
        for v in ["pct_counts_mt", "n_genes_by_counts"]:
            try:
                figs[f"qc_{v}_scatter"] = sc.pl.scatter(
                    adata[assay] if assay else adata, x="total_counts", y=v)
            except Exception as err:
                print(err)
    figs["qc_log"] = seaborn.jointplot(
        data=adata[assay].obs if assay else adata.obs,
        x="log1p_total_counts", y="log1p_n_genes_by_counts", kind="hex")
        
    # More Filtering
    try:
        if assay is None:
            adata = adata[adata.obs.n_genes_by_counts < max_genes_by_counts, :]
            adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]
            adata.raw = adata  # freeze normalized & filtered adata
        else:
            adata[assay] = adata[assay][
                adata[assay].obs.n_genes_by_counts < max_genes_by_counts, :]
            adata[assay] = adata[assay][
                adata[assay].obs.pct_counts_mt < max_pct_mt, :]  # MT counts
            adata[assay].raw = adata[assay]  # freeze normalized, filtered data
    except TypeError as err_f:
            warnings.warn(f"\n\n{'=' * 80}\n\nCould not filter: {err_f}")
    
    # Normalize
    print("\n<<< NORMALIZING >>>")
    sc.pp.normalize_total(adata[assay] if assay else adata, 
                          target_sum=target_sum)  # count-normalize
    sc.pp.log1p(adata[assay] if assay else adata)  # log-normalize
    if assay_protein is not None:  # if includes protein assay
        muon.prot.pp.clr(adata[assay_protein])
        
    # Variable Genes
    if kws_hvg is not None:
        print("\n<<< DETECTING VARIABLE GENES >>>")
        if kws_hvg is True:
            kws_hvg = {}
        figs["highly_variable_genes"] = sc.pp.highly_variable_genes(
            adata[assay] if assay else adata, 
            **kws_hvg)  # highly variable genes 
        try:
            if assay is None:
                adata= adata[:, adata.var.highly_variable]  # filter by HVGs
            else:
                adata[assay]= adata[:, adata[
                    assay].var.highly_variable]  # filter by HVGs
        except (TypeError, IndexError) as err_h:
                warnings.warn(f"""\n\n{'=' * 80}\n\n Could not subset 
                              by highly variable genes: {err_h}""")
    
    # Regress Confounds
    if regress_out: 
        print("\n<<< REGRESSING OUT CONFOUNDS >>>")
        sc.pp.regress_out(adata[assay] if assay else adata, regress_out)
    
    # Scaling Genes
    if scale is not None:
        print("\n<<< SCALING >>>")
        if scale is True:  # if True, just scale to unit variance
            sc.pp.scale(adata[assay] if assay else adata, 
                        **kws_scale)  # scale
        else:  # if scale provided as an integer...
            sc.pp.scale(adata[assay] if assay else adata, max_value=scale,
                        **kws_scale)  # ...also clip values > "scale" SDs
            
    # Cell Counts (Post-Processing)
    if col_cell_type is not None and col_cell_type in adata.obs:
        print(f"\n\n{'=' * 80}\nCell Counts (Post-Processing)\n{'=' * 80}\n\n")
        print(adata.obs[col_cell_type].value_counts())
        
    # Gene Expression Heatmap
    # if assay:
    #     labels = adata[assay].var.reset_index()[col_gene_symbols]  # genes
    #     if labels[0] != adata[assay].var.index.values[0]:
    #         labels = dict(zip(list(adata[assay].var.index.values), list(
    #             adata[assay].var.reset_index()[col_gene_symbols])))
    # else:
    #     labels = adata.var.reset_index()[col_gene_symbols]  # gene names
    #     if labels[0] != adata.var.index.values[0]:
    #         labels = dict(zip(list(adata.var.index.values), list(
    #             adata.var.reset_index()[col_gene_symbols])))
    # figs["gene_expression"] = sc.pl.heatmap(
    #     adata[assay] if assay else adata, labels,
    #     col_cell_type, show_gene_labels=True, dendrogram=True)
    # if layer is not None:
    #     labels = adata.var.reset_index()[col_gene_symbols]  # gene names 
    #     figs[f"gene_expression_{layer}"] = sc.pl.heatmap(
    #         adata[assay] if assay else adata, labels,
    #         layer=layer, show_gene_labels=True, 
    #         dendrogram=True)
            
    print("\n\n")
    return adata, figs


def explore_h5_file(file):
    """Explore an H5 file's format (thanks to ChatGPT)."""
    with h5py.File(file, "r") as h5_file:
        # List all top-level groups in the HDF5 file
        top_level_groups = list(h5_file.keys())
        # Explore the groups and datasets within each group
        for group_name in top_level_groups:
            group = h5_file[group_name]
            print(f"Group: {group_name}")
            # List datasets within the group
            datasets = list(group.keys())
            for dataset_name in datasets:
                print(f"  Dataset: {dataset_name}")


def get_matrix_from_h5(file, gex_genes_return=None):
    """Get matrix from 10X h5 file (modified from 10x code)."""
    FeatureBCMatrix = collections.namedtuple("FeatureBCMatrix", [
        "feature_ids", "feature_names", "barcodes", "matrix"])
    with h5py.File(file) as f:
        if u"version" in f.attrs:
            version = f.attrs["version"]
            if version > 2:
                print(f"Version = {version}")
                raise ValueError(f"HDF5 format version version too new.")
        else:
            raise ValueError(f"HDF5 format version ({version}) too old.")
        feature_ids = [x.decode("ascii", "ignore") 
                       for x in f["matrix"]["features"]["id"]]
        feature_names = [x.decode("ascii", "ignore") 
                         for x in f["matrix"]["features"]["name"]]        
        barcodes = list(f["matrix"]["barcodes"][:])
        matrix = sp_sparse.csr_matrix((f["matrix"]["data"], 
                                       f["matrix"]["indices"], 
                                       f["matrix"]["indptr"]), 
                                      shape=f["matrix"]["shape"])
        fbm = FeatureBCMatrix(feature_ids, feature_names, barcodes, matrix)
        if gex_genes_return is not None:
            gex = {}
            for g in gex_genes_return:
                try:
                    gene_index = fbm.feature_names.index(g)
                except ValueError:
                    raise Exception(f"{g} not found in list of gene names.")
                gex.update({g: fbm.matrix[gene_index, :].toarray(
                    ).squeeze()})  # gene expression
        else:
            gex = None
        barcodes = [x.tostring().decode() for x in fbm.barcodes]
        genes = pd.Series(fbm.feature_names).to_frame("gene").join(
            pd.Series(fbm.feature_ids).to_frame("gene_ids"))
    return fbm, gex, barcodes, genes


def assign_guide_rna(adata, assignment_threshold=5, layer="counts",
                     plot=False, **kwargs):
    """Assign guide RNAs to cells (based on pertpy tutorial notebook)."""
    gdo = adata.mod["gdo"]
    gdo.layers["counts"] = gdo.X.copy()  # save original counts
    sc.pp.log1p(gdo)  # log-transform data
    if plot is True:
        pt.pl.guide.heatmap(gdo, key_to_save_order="plot_order", 
                            **kwargs)  # heatmap
    g_a = pt.pp.GuideAssignment()  # guide assignment
    g_a.assign_by_threshold(gdo, assignment_threshold=assignment_threshold, 
                            layer=layer)  # assignment thresholding
    g_a.assign_to_max_guide(gdo, assignment_threshold=assignment_threshold, 
                            layer=layer)  # assignment thresholding
    print(gdo.obs["assigned_guide"])  # creates layer "assigned_guides"
    return gdo


def remove_batch_effects(adata, col_cell_type="leiden", 
                         col_batch="batch", plot=True):
    """Remove batch effects (IN PROGRESS)."""
    if plot is True:
        sc.pl.umap(adata, color=[col_batch, col_cell_type], 
                   wspace=.5, frameon=False)
    pt.tl.SCGEN.setup_anndata(adata, batch_key=col_batch, 
                              labels_key=col_cell_type)
    model = pt.tl.SCGEN(adata)
    model.train(max_epochs=100, batch_size=32, 
                early_stopping=True, early_stopping_patience=25)
    corrected_adata = model.batch_removal()
    if plot is True:
        sc.pp.neighbors(corrected_adata)
        sc.tl.umap(corrected_adata)
        sc.pl.umap(corrected_adata, color=[col_batch, col_cell_type], 
                   wspace=0.4, frameon=False)
    return corrected_adata


def detect_guide_targets(adata, col_guide_rna="guide_ID", 
                         feature_split="|", guide_split="-",
                         key_control_patterns=None,
                         key_control="Control", **kwargs):
    """Detect guide RNA gene targets."""
    if kwargs:
        print(f"\nUn-used Keyword Arguments: {kwargs}")
    if key_control_patterns is None:
        key_control_patterns = [
            key_control]  # if already converted, pattern=key itself
    if isinstance(key_control_patterns, str) or pd.isnull(
        key_control_patterns):
        key_control_patterns = [key_control_patterns]
    targets = adata.obs[col_guide_rna].str.strip(" ").replace("", np.nan)
    if np.nan in key_control_patterns:  # if NAs mean control sgRNAs
        key_control_patterns = list(pd.Series(key_control_patterns).dropna())
        targets = targets.replace(
            np.nan, key_control)  # NaNs replaced w/ control key
    else:  # if NAs mean unperturbed cells
        if any(pd.isnull(targets)):
            warnings.warn(
                f"Dropping rows with NaNs in `col_guide_rna`.")
        targets = targets.dropna()
    keys_leave = [key_control]  # entries to leave alone
    targets, nums = [targets.apply(
        lambda x: [re.sub(p, ["", r"\1"][j], str(i)) for i in x.split(
            feature_split)]) for j, p in enumerate([
                f"{guide_split}.*", rf'^.*?{re.escape(guide_split)}(.*)$'])
                        ]  # each entry -> list of target genes
    targets = targets.apply(
        lambda x: [i if i in keys_leave else key_control if any(
                (k in i for k in key_control_patterns)) else i 
            for i in x])  # find control keys among targets
    grnas = targets.to_frame("t").join(nums.to_frame("n")).apply(
        lambda x: [i + guide_split + "_".join(np.array(
            x["n"])[np.where(np.array(x["t"]) == i)[0]]) 
                    for i in pd.unique(x["t"])], 
        axis=1).apply(lambda x: feature_split.join(x)).to_frame(
            col_guide_rna
            )  # e.g., STAT1-1|STAT1-2|NT-1-2 => STAT1-1_2
    return targets, grnas


def find_guide_info(adata, col_guide_rna, col_num_umis=None, 
                    feature_split="|", guide_split="-",
                    key_control_patterns=None,
                    key_control="Control", **kwargs):
    if isinstance(key_control_patterns, str):
        key_control_patterns = [key_control_patterns]
    targets, grnas = detect_guide_targets(
        adata, col_guide_rna=col_guide_rna,
        feature_split=feature_split, guide_split=guide_split, 
        key_control_patterns=key_control_patterns, 
        key_control=key_control, **kwargs)  # target genes
    tg_info = grnas.iloc[:, 0].to_frame(
        col_guide_rna + "_flat_ix").join(
            targets.to_frame(col_guide_rna + "_list"))
    if col_num_umis is not None:
        tg_info = tg_info.join(
            adata.obs[[col_num_umis]].apply(
                lambda x: [float(i) for i in str(x[col_num_umis]).split(
                    feature_split)], axis=1).to_frame(
                        col_num_umis + "_list"))
    tg_info = adata.obs[col_guide_rna].to_frame(col_guide_rna).join(tg_info)
    tg_info = tg_info.join(tg_info[col_num_umis + "_list"].dropna().apply(
        sum).to_frame(col_num_umis + "_total"))  # total UMIs/cell
    return tg_info


def filter_by_guide_counts(adata, col_guide_rna, col_num_umis=None, 
                           max_percent_umis_control_drop=75,
                           min_percent_umis=40,
                           feature_split="|", guide_split="-",
                           key_control_patterns=None,
                           key_control="Control", **kwargs):
    # Extract Guide RNA Information
    tg_info = find_guide_info(
        adata, col_guide_rna, col_num_umis=col_num_umis, 
        feature_split=feature_split, guide_split=guide_split, 
        key_control_patterns=key_control_patterns, 
        key_control=key_control, **kwargs)

    # Sum Up gRNA UMIs
    cols = [col_guide_rna + "_list", col_num_umis + "_list"]
    feats_n = tg_info[cols].dropna().apply(lambda x: pd.Series(
        dict(zip(pd.unique(x[cols[0]]), [sum(np.array(x[cols[1]])[
            np.where(np.array(x[cols[0]]) == i)[0]]) for i in pd.unique(
                x[cols[0]])]))), axis=1).stack().rename_axis(["bc", "g"])

    # Actual Filtering
    feats_n = feats_n.to_frame("n").join(feats_n.groupby("bc").sum().to_frame("t"))
    feats_n = feats_n.assign(p=feats_n.n / feats_n.t * 100)
    filt = feats_n.groupby(["bc", "g"]).apply(
        lambda x: np.nan if x.name[1] == key_control and float(
            x["p"]) <= max_percent_umis_control_drop else np.nan if float(
                x["p"]) < min_percent_umis else x.stack())
    filt = pd.concat(list(filt.dropna())).unstack(2)
    filt = filt.n.to_frame("u").groupby("bc").apply(
        lambda x: pd.Series({cols[0]: list(x.reset_index("g")["g"]), 
                            cols[1]: list(x.reset_index("g")["u"])}))
    tg_info = tg_info.join(filt, lsuffix="_all", rsuffix="_count_filtered")
    tg_info = tg_info.dropna().loc[adata.obs.index.intersection(
        tg_info.dropna().index)]  # re-order according to adata index
    # # Determine Exclusion Criteria Met per gRNA
    # tg_info = tg_info.join(feats_n.to_frame("u").groupby("bc").apply(
    #     lambda x: pd.Series({cols[0]: list(x.reset_index("g")["g"]), 
    #                          cols[1]: list(x.reset_index("g")["u"])})), 
    #                        rsuffix="_unique", lsuffix="_all")
    # ecol, grc, nuc = [f"{col_guide_rna}_exclusions", f"{col_guide_rna}_list_unique", 
    #                   f"{col_num_umis}_percent"]  # column names
    # tg_info.loc[:, ecol] = tg_info[grc].apply(lambda x: [""] * len(x))  # placehold
    # tg_info.loc[:, ecol] = tg_info.apply(
    #     lambda x: [x[ecol][i] + " " + "low_umis" if x[
    #         nuc][i] < min_percent_umis else x[ecol][i] 
    #                for i, q in enumerate(x[grc])], axis=1)  # low UMI count
    # tg_info.loc[:, ecol] = tg_info.apply(
    #     lambda x: [x[ecol][i] + " " + "tolerable_control_umis" if (
    #         q == key_control and any(np.array(x[grc]) != key_control) and x[
    #             nuc][i] <= max_percent_umis_control_drop) else x[ecol][i] 
    #                for i, q in enumerate(x[grc])], axis=1
    #     )  # control guides w/i tolerable % => drop control guide
    # tg_info.loc[:, ecol] = tg_info[ecol].apply(lambda x: [
    #     np.nan if i == "" else i for i in x])
    for q in [col_guide_rna, col_num_umis]:
        tg_info.loc[:, q] = tg_info[q + "_list_count_filtered"].apply(
            lambda x: feature_split.join(str(i) for i in x))
    return tg_info
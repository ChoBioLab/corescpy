#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Preprocessing single-cell data.

@author: E. N. Aslinger
"""

import os
import scanpy as sc
import pertpy as pt
import muon
from  warnings import warn
import seaborn
import matplotlib.pyplot as plt 
# import anndata
from anndata import AnnData
import crispr as cr
from crispr.class_sc import Omics
import pandas as pd
import numpy as np

pd.DataFrame.iteritems = pd.DataFrame.items  # back-compatibility
regress_out_vars = ["total_counts", "pct_counts_mt"]


def get_layer_dict():
    """Retrieve layer name conventions."""
    lay =  {"preprocessing": "preprocessing", 
            "perturbation": "X_pert",
            "unnormalized": "unnormalized",
            "norm_total_counts": "norm_total_counts",
            "log1p": "log1p",
            "unscaled": "unscaled", 
            "scaled": "scaled", 
            "unregressed": "unregressed",
            "counts": "counts"}
    return lay


def create_object_multi(file_path, kws_init=None, kws_pp=None, 
                        kws_cluster=None, kws_harmony=True):
    """Create objects, then preprocess, cluster, & integrate them."""
    # Arguments
    ids = list(file_path.keys())
    [kws_init, kws_pp, kws_cluster] = [dict(zip(ids, x)) if isinstance(
        x, list) else dict(zip(ids, [x] * len(ids))) for x in [
            kws_init, kws_pp, kws_cluster]]  # dictionaries for each

    # Create AnnData Objects
    selves = dict(zip(file_path, [
        Omics(file_path[f], **kws_init[f]) if (
            "kws_process_guide_rna" not in kws_init) else cr.Crispr(
                file_path[f], **kws_init[f]) for f in file_path])
                  )  # create individual objects

    # Preprocessing & Clustering
    for x in selves:  # preprocess & cluster each object
        if kws_pp is not None:
            print(f"\n<<< PREPROCESSING {x} >>>")
            selves[x].preprocess(**kws_pp[x])
        if kws_cluster is not None:
            print(f"\n<<< CLUSTERING {x} >>>")
            selves[x].cluster()
    print(f"\n<<< CONCATENATING OBJECTS: {', '.join(ids)} >>>")
    col_id = selves[ids[0]]._columns["col_sample_id"]
    if col_id is None:
        col_id = "unique.idents"
    adata = AnnData.concatenate(
        *[selves[x].adata for x in selves], join="outer", batch_key=col_id,
        batch_categories=ids, uns_merge="same", 
        index_unique="-", fill_value=None)  # concatenate AnnData objects
        
    # Integrate
    if kws_harmony is not None:
        print(f"\n<<< INTEGRATING WITH HARMONY >>>")
        sc.tl.pca(adata)  # PCA
        sc.external.pp.harmony_integrate(
            adata, col_id, basis="X_pca", 
            adjusted_basis="X_pca_harmony", **kws_harmony)  # harmony
        adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]  # assign new basis
        adata.uns["harmony"] = True
    return adata


def create_object(file, col_gene_symbols="gene_symbols", assay=None,
                  kws_process_guide_rna=None, **kwargs):
    """
    Create object from Scanpy- or Muon-compatible file(s) or object.
    """
    # Load Object (or Copy if Already AnnData or MuData)
    _ = kwargs.pop("col_sample_id") if "col_sample_id" in kwargs else None
    if isinstance(file, (str, os.PathLike)) and os.path.splitext(
        file)[1] == ".h5mu":  # MuData
        print(f"\n<<< LOADING FILE {file} with muon.read() >>>")
        adata = muon.read(file)
    elif isinstance(file, dict):  # metadata in protospacer files
        print(f"\n<<< LOADING PROTOSPACER METADATA >>>")
        adata = cr.pp.combine_matrix_protospacer(
            **file, col_gene_symbols=col_gene_symbols, 
            **kwargs)  # + metadata from protospacer
    elif not isinstance(file, (str, os.PathLike)):  # if already AnnData
        print(f"\n<<< LOADING OBJECT >>>")
        adata = file.copy()
    elif os.path.isdir(file):  # if directory, assume 10x format
        print(f"\n<<< LOADING 10X FILE {file} >>>")
        adata = sc.read_10x_mtx(
            file, var_names=col_gene_symbols, cache=True, **kwargs)
    elif os.path.splitext(file)[1] == ".h5":  # .h5 file
        print(f"\n<<< LOADING 10X .h5 FILE {file} >>>")
        print(f"H5 File Format ({file})\n\n")
        # cr.tl.explore_h5_file(file, "\n\n\n")
        adata = sc.read_10x_h5(file, **kwargs)
    else:
        print(f"\n<<< LOADING FILE {file} with sc.read() >>>")
        adata = sc.read(file)
    
    # Formatting & Initial QC Visualization
    adata.var_names_make_unique()
    try: 
        adata.obs_names_make_unique()
    except Exception as err:
        warn(f"{err}\n\n\nCoult not make obs names unique.")
    cr.tl.print_counts(adata, title="Raw")
    if col_gene_symbols not in adata.var.columns:
        # if assay: 
        #     adata[assay] = adata[assay].var.rename_axis(col_gene_symbols) 
        # else:
        adata.var = adata.var.rename_axis(col_gene_symbols)
    # cr.pp.perform_qc(adata.copy(), hue=col_sample_id)  # plot QC
    
    # Process Guide RNA
    if kws_process_guide_rna:
        if assay:
            adata[assay]  = cr.pp.process_guide_rna(
                adata[assay], **kws_process_guide_rna)
        else:
            adata = cr.pp.process_guide_rna(adata, **kws_process_guide_rna)
        cct = kws_process_guide_rna["col_cell_type"] if "col_cell_type" in (
            kws_process_guide_rna) else None
        cr.tl.print_counts(adata[assay] if assay else adata, 
                           title="Post-Guide RNA Processing", group_by=cct)
        # TODO: FIGURE
    
    # Layers & QC
    layers = cr.pp.get_layer_dict()  # standard layer names
    if assay: 
        adata[assay].layers[layers["counts"]] = adata[assay].X.copy()
    else:
        adata.layers[layers["counts"]] = adata.X.copy()
    print("\n\n", adata)
    return adata
    
    
def process_data(adata, 
                 col_gene_symbols=None,
                 col_cell_type=None,
                 # remove_doublets=True,
                 outlier_mads=None,
                 cell_filter_pmt=None,
                 cell_filter_ncounts=None, 
                 cell_filter_ngene=None,
                 gene_filter_ncell=None,
                 target_sum=1e4,
                 kws_hvg=True,
                 kws_scale=None,
                 regress_out=regress_out_vars, 
                 **kwargs):
    """
    Perform various data processing steps.

    Args:
        adata (AnnData or MuData): The input data object.
        col_gene_symbols (str, optional): The name of the column or 
            index in `.var` containing gene symbols. Defaults to None.
        col_cell_type (str, optional): The name of the column 
            in `.obs` containing cell types. Defaults to None.
        target_sum (float, optional): Total-count normalize to
            <target_sum> reads per cell, allowing between-cell 
            comparability of counts. If None, total count-normalization
            will not be performed. Defaults to 1e4.
        outlier_mads (float or int or dict): To calculate outliers
            based on MADs (see SC Best Practices). If a dictionary,
            key based on names of columns added by QC. Filtering
            will be performed based on outlier status rather than
            other arguments to this function if not None.
            Defaults to None.
        cell_filter_pmt (list, optional): The range of percentage of 
            mitochondrial genes per cell allowed. Will filter out cells
            that have outside the range [minimum % mt, maximum % mt].
            Defaults to 5.
        cell_filter_ncounts (list, optional): Retain only cells
            that have a number of reads within the 
            range specified: [minimum reads, maximum].
        cell_filter_ngene (list, optional): Retain only cells that 
            express a number of genes within the 
            range specified: 
            [minimum genes expressed within a cell, maximum].
            Specify 0 as the minimum in order still to calculate
            certain metrics but without filtering.
            Defaults to None (no filtering on this property performed).
        gene_filter_ncell (list, optional): Retain only genes that 
            are expressed by a number of cells within the 
            range specified: [minimum cells expressing gene, maximum].
            If either element is None, filtering is not performed
            according to that property.
            If True, filtering criteria are derived from the data
                (i.e., outlier calculation).
            Defaults to None (no filtering on this property performed).
        kws_hvg (dict or bool, optional): The keyword arguments for 
            detecting variable genes or True for default arguments. 
            To filter by HVGs, include an 
            extra argument in this dictionary: {"filter": True}.
            Otherwise, a highly_variable genes column is created and
            can be used in clustering, but non-HVGs will be retained in
            the data for other uses. Defaults to True.
        kws_scale (int, bool, or dict, optional): Specify True to 
            center on average gene expression and scale to unit 
            variance, as an integer to additionally clip maximum 
            standard deviation, or as a dictionary with keyword 
            arguments in order to normalize with reference to some
            category in a column in `.obs`.
            If a dictionary, under the key "col_reference", specify the 
            name of the columns containing reference vs. other, and
            under the key "key_reference", the label of the reference 
            category within that column.
            Note that using a dictionary (z-scoring) will result in
            the `.X` attribute being reset to the 
            un-log-normalized data, then scaled and set to 
            the scaled data. To avoid this behavior, pass "layer" in
            the dictionary to specify a layer to set before scaling.
        regress_out (list or None, optional): The variables to 
            regress out. Defaults to regress_out_vars.
        **kwargs: Additional keyword arguments.

    Returns:
        adata (AnnData): The processed data object.
        figs (dict): A dictionary of generated figures.
    """
    # Setup Object
    layers = cr.pp.get_layer_dict()  # layer names
    ann = adata.copy()  # copy so passed AnnData object not altered inplace
    ann.layers[layers["counts"]] = ann.X.copy()  # original counts in layer
    # ann.layers[layers["preprocessing"]] = ann.X.copy() # set original layer
    ann.obs["n_counts"] = ann.X.sum(1)
    ann.obs["log_counts"] = np.log(ann.obs["n_counts"])
    ann.obs["n_genes"] = (ann.X > 0).sum(1)
    print(ann)
    
    # Initial Information/Arguments
    if col_gene_symbols == ann.var.index.names[0]:  # if symbols=index...
        col_gene_symbols = None  # ...so functions will refer to index name
    figs = {}
    kws_hvg = {} if kws_hvg is True else kws_hvg
    filter_hvgs = kws_hvg.pop("filter") if "filter" in kws_hvg else False
    n_top = kwargs.pop("n_top") if "n_top" in kwargs else 10
    # if kws_scale:
    #     sid = kws_scale["col_batch"] if "col_batch" in kws_scale else None
    if "col_batch" in kwargs or "col_sample_id" in kwargs:
        sids = []
        for x in ["col_batch", "col_sample_id"]:
            if x in kwargs and x not in sids and kwargs[x] is not None: 
                sss = kwargs.pop("col_sample_id")
                sids += [sss]
        if len(sids) == 0:
            sids = None
    if kwargs:
        print(f"\nUn-Used Keyword Arguments: {kwargs}\n\n")
        
    # Set Up Layer & Variables
    if outlier_mads is not None:  # if filtering based on calculating outliers 
        if isinstance(outlier_mads, (int, float)):  # same MADs, all metrics
            qc_mets = ["log1p_total_counts", "log1p_n_genes_by_counts",
                       "pct_counts_in_top_20_genes"]
            outlier_mads = dict(zip(qc_mets, [outlier_mads] * len(qc_mets)))
    cr.tl.print_counts(ann, title="Initial", group_by=col_cell_type)
    print(col_gene_symbols, "\n\n", n_top, "\n\n", ann.var.describe(
        ) if "var" in dir(ann) else None, "\n\n")
    
    # Exploration & QC Metrics
    print("\n<<< PERFORMING QUALITY CONTROL ANALYSIS>>>")
    figs["qc_metrics"] = cr.pp.perform_qc(
        ann, n_top=n_top, col_gene_symbols=col_gene_symbols,
        hue=sids)  # QC metric calculation & plotting

    # Basic Filtering (DO FIRST...ALL INPLACE)
    print("\n<<< FILTERING CELLS (TOO FEW GENES) & GENES (TOO FEW CELLS) >>>") 
    if cell_filter_ngene:
        sc.pp.filter_cells(ann, min_genes=cell_filter_ngene[0])
    if gene_filter_ncell:
        sc.pp.filter_genes(ann, min_cells=gene_filter_ncell[0])
    cr.tl.print_counts(ann, title="Post-Basic Filter", group_by=col_cell_type)
    
    # Further Filtering
    print("\n<<< FURTHER CELL & GENE FILTERING >>>")
    ann = cr.pp.filter_qc(ann, outlier_mads=outlier_mads,
                          cell_filter_pmt=cell_filter_pmt,
                          cell_filter_ncounts=cell_filter_ncounts,
                          cell_filter_ngene=cell_filter_ngene, 
                          gene_filter_ncell=gene_filter_ncell)
    cr.tl.print_counts(ann, title="Post-Filter", group_by=col_cell_type)
    
    # Doublets
    # doublets = detect_doublets(adata)
    # if remove_doublets is True:
    #     adata
    #     # TODO: doublets
    
    # Normalization
    print("\n<<< NORMALIZING CELL COUNTS >>>")
    if target_sum is not None:  # total-count normalization INPLACE
        print("\n\t*** Total-count normalizing...")
        sc.pp.normalize_total(ann, target_sum=target_sum, copy=False)
    print(f"\n\t*** Log-normalizing => `.X` & {layers['log1p']} layer...")
    sc.pp.log1p(ann)  # log-transformed; INPLACE
    ann.layers[layers["log1p"]] = ann.X.copy()  # also keep in layer
    ann.raw = ann.copy()  # before HVG, batch correction, etc.
        
    # Gene Variability (Detection, Optional Filtering)
    print("\n<<< DETECTING VARIABLE GENES >>>")
    sc.pp.highly_variable_genes(ann, **kws_hvg)
    if filter_hvgs is True:
        print("\n<<< FILTERING BY HIGHLY VARIABLE GENES >>>")
        ann = ann[:, ann.var.highly_variable]  # filter by HVGs
        cr.tl.print_counts(ann, title="HVGs", group_by=col_cell_type)
    
    # Regress Out Confounds
    if regress_out: 
        print(f"\n<<< REGRESSING OUT CONFOUNDS >>>\n\n\t{regress_out}")
        ann.layers[layers["unregressed"]] = ann.X.copy()
        sc.pp.regress_out(ann, regress_out, copy=False)
        warn("Confound regression doesn't yet properly use layers.")
    
    # Gene Expression Normalization
    if kws_scale is not None:
        print("\n<<< NORMALIZING RAW GENE EXPRESSION >>>")
        normalize_genes(ann, kws_reference=kws_scale if isinstance(
            kws_scale, dict) else None)  # scale by overall or reference
        print(f"\n\t*** Scaling => `.X` & {layers['scaled']} layer...")
        ann.layers[layers["scaled"]] = ann.X.copy()  # also keep in layer
            
    # Final Data Examination
    cr.tl.print_counts(ann, title="Post-Processing", group_by=col_cell_type)
    figs["qc_metrics_post"] = cr.pp.perform_qc(
        ann, n_top=n_top, col_gene_symbols=col_gene_symbols,
        hue=sids)  # QC metric calculation & plot
    return ann, figs


def normalize_genes(adata, max_value=None, kws_reference=None, **kwargs):
    """Normalize gene expression relative to overall or reference."""
    if kws_reference is not None:  # with reference to some control
        print(f"\n\t*** Resetting to raw counts before scaling...")
        if any((x not in kws_reference for x in [
            "col_reference", "key_reference"])):
            raise ValueError(
                "'col_reference' and 'key_reference' must be "
                "in `normalization` argument if method = 'z'.")
        print("\n\t*** Z-scoring (relative to "
              f"{kws_reference['key_reference']})")
        adata = z_normalize_by_reference(adata, **kws_reference)
    else:  # with reference to whole sample
        print(f"\n\t*** Scaling gene expression...")
        if max_value:
            print(f"\n\t*** Clipping maximum GEX SD to {max_value}...")
            kwargs.update({"max_value": max_value})
        sc.pp.scale(adata, copy=False, **kwargs)  # center/standardize


def check_normalization(adata, n_genes=1000):
    """Check if an AnnData object is log-normalized."""
    return (adata.X[:1000].min() >= 0) or (adata.X[:n_genes].max() <= 9.22)


def z_normalize_by_reference(adata, col_reference="Perturbation", 
                             key_reference="Control", 
                             retain_zero_variance=True, 
                             layer=None, **kwargs):
    """
    Mean-center & standardize by reference condition 
    (within-batch option).
    If `retain_zero_variance` is True, genes with 
    zero variance are retained.
    """
    if kwargs:
        print(f"\nUn-Used Keyword Arguments: {kwargs}\n\n")
    if layer is None:
        layer = cr.pp.get_layer_dict()["counts"]  # raw counts layer
    if layer is not None:
        adata.X = adata.layers[layer].copy()  # reset to raw counts
    if layer:
        adata.X = adata.layers[layer].copy()
    gex = adata.X.copy()  #  gene expression matrix (full)
    gex_ctrl = adata[adata.obs[
        col_reference] == key_reference].X.A.copy()  # reference condition
    gex, gex_ctrl = [q.A if "A" in dir(q) else q 
                        for q in [gex, gex_ctrl]]  # sparse -> dense matrix
    mus, sds = np.nanmean(gex_ctrl, axis=0), np.nanstd(
        gex_ctrl, axis=0)  # means & SDs of reference condition genes
    if retain_zero_variance is True:
        sds[sds == 0] = 1   # retain zero-variance genes at unit variance
    adata.X = (gex - mus) / sds  # z-score gene expression
    return adata


def perform_qc(adata, n_top=20, col_gene_symbols=None, 
               hue=None, patterns=None, layer=None):
    """Calculate & plot quality control metrics."""
    figs = {}
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    try:
        figs["highly_expressed_genes"] = sc.pl.highest_expr_genes(
            adata, n_top=n_top, gene_symbols=col_gene_symbols)  # high GEX genes
    except Exception as err:
        warn(f"{err}\n\n{'=' * 80}\n\nCouldn't plot highly expressed genes!")
        figs["highly_expressed_genes"] = err
    if patterns is None:
        patterns = dict(zip(["mt", "ribo", "hb"], 
                            [("MT-", "mt-"), ("RPS", "RPL", "rps", "rpl"), (
                                "^HB[^(P)]", "^hb[^(p)]")]))
    names = dict(zip(["mt", "ribo", "hb"], 
                     ["Mitochondrial", "Ribosomal", "Hemoglobin"]))
    p_names = [names[k] if k in names else k for k in patterns]  # "pretty"
    patterns_names = dict(zip(patterns, p_names))  # map abbreviated to pretty
    print(f"\n\t*** Detecting {', '.join(p_names)} genes...") 
    for k in patterns:
        try:
            gvars = adata.var_names.str.startswith(patterns[k])
            if len(gvars) > 0:
                adata.var[k] = gvars
        except Exception as err:
            warn(f"\n\n{'=' * 80}\n\nCouldn't assign {k}: {err}")
    qc_vars = list(set(patterns.keys()).intersection(
        adata.var.keys()))  # available QC metrics 
    pct_n = [f"pct_counts_{k}" for k in qc_vars]  # "% counts" variables
    print("\n\t*** Calculating & plotting QC metrics...\n\n")
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, percent_top=None, 
                               log1p=True, inplace=True)  # QC metrics
    hhh, yes = [hue] if isinstance(hue, str) else hue, True
    if hhh:  # plot by hue variables (e.g., batch, sample)
        for h in hhh:
            if h not in adata.obs and h not in adata.var:
                warn(f"\n\t{h} not found in adata.obs or adata.var; "
                    "skipping color-coding")
                if yes is None:
                    continue  # skip if already did "None" hue
                yes = None
            rrs, ccs = cr.pl.square_grid(len(pct_n + ["n_genes_by_counts"])
                                        )  # dimensions for subplot grid
            fff, axs = plt.subplots(rrs, ccs, figsize=(
                5 * rrs, 5 * ccs))  # subplot figure & axes
            for a, v in zip(axs.flat, pct_n + ["n_genes_by_counts"]):
                try:  # facet "v" of scatterplot
                    sc.pl.scatter(adata, x="total_counts", y=v, ax=a, 
                                  show=False, color=h if yes else None)
                except Exception as err:
                    print(err)
            plt.show()
            figs[f"qc_scatter_by_{h}" if yes else "qc_scatter"] = fff
            try:
                vam = pct_n + ["n_genes_by_counts"] + list([h] if yes else [])
                fff = seaborn.pairplot(
                    adata.obs[vam].rename_axis("Metric", axis=1).rename({
                        "total_counts": "Total Counts", **patterns_names
                        }, axis=1), diag_kind="kde", hue=h if yes else None, 
                    diag_kws=dict(fill=True, cut=0))  # pairplot
            except Exception as err:
                fff = err
                print(err)
            figs[f"pairplot_by_{h}" if yes else "pairpolot"] = fff
    try:
        figs["pct_counts_kde"] = seaborn.displot(
            adata.obs[pct_n].rename_axis("Metric", axis=1).rename(
                patterns_names, axis=1).stack().to_frame(
                    "Percent Counts"), x="Percent Counts", col="Metric", 
            kind="kde", cut=0, fill=True)  # KDE of pct_counts
    except Exception as err:
        figs["pct_counts_kde"] = err
        print(err)
    try:
        figs["metrics_violin"] = sc.pl.violin(
            adata, ["n_genes_by_counts", "total_counts"] + pct_n,
            jitter=0.4, multi_panel=True)  # violin of counts, genes
    except Exception as err:
        figs["qc_metrics_violin"] = err
        print(err)
    try:
        figs["qc_log"] = seaborn.jointplot(
            data=adata.obs, x="log1p_total_counts", 
            y="log1p_n_genes_by_counts", kind="hex")  # jointplot
    except Exception as err:
        figs["qc_log"] = err
        print(err)
    print(adata.var.describe())
    return figs


def filter_qc(adata, outlier_mads=None, cell_filter_pmt=None, 
              cell_filter_ncounts=None, cell_filter_ngene=None, 
              gene_filter_ncell=None):
    """Filter low-quality/outlier cells & genes."""
    ann = adata.copy()
    if isinstance(cell_filter_pmt, (int, float)):  # if just 1 # for MT %...
        cell_filter_pmt = [0, cell_filter_pmt]  # ...assume it's for maximum %
    if cell_filter_pmt is None:  # None = no MT filter but calculates metrics
        cell_filter_pmt = [0, 
                           100]
    min_pct_mt, max_pct_mt = cell_filter_pmt
    if outlier_mads is not None:  # automatic filtering using outlier stats
        outliers = ann.obs[outlier_mads.keys()]
        print(f"\n<<< DETECTING OUTLIERS {outliers.columns} >>>") 
        for x in outlier_mads:
            outliers.loc[:, f"outlier_{x}"] = cr.tl.is_outlier(
                ann.obs, outlier_mads[x])  # separate metric outlier columns
        cols_outlier = list(set(
            outliers.columns.difference(ann.obs.columns)))
        outliers.loc[:, "outlier"] = outliers[cols_outlier].any()  # binary
        print(f"\n<<< FILTERING OUTLIERS ({cols_outlier}) >>>") 
        ann.obs = ann.obs.join(outliers[["outlier"]])  # + outlier column
        print(f"Total Cell Count: {ann.n_obs}")
        ann = ann[(~ann.obs.outlier) & (~ann.obs.mt_outlier)].copy()  # drop
        print(f"Cell Count (Outliers Dropped): {ann.n_obs}")
    else:  # manual filtering
        print("\n<<< PERFORING THRESHOLD-BASED FILTERING >>>") 
        print(f"\nTotal Cell Count: {ann.n_obs}")
        print("\n\t*** Filtering cells by mitochondrial gene percentage...") 
        print(f"\n\tMinimum={min_pct_mt}\n\tMaximum={max_pct_mt}")
        ann = ann[(ann.obs.pct_counts_mt <= max_pct_mt) * (
            ann.obs.pct_counts_mt >= min_pct_mt)]  # filter by MT %
        print(f"\tNew Count: {ann.n_obs}")
        print("\n\t*** Filtering genes based on # of genes expressed...")
        cell_filter_ngene, cell_filter_ncounts, gene_filter_ncell, \
            gene_filter_ncell = [x if x else [None, None] for x in [
                cell_filter_ngene, cell_filter_ncounts, 
                gene_filter_ncell, gene_filter_ncell]]  # -> iterable if None
        if cell_filter_ngene[0] is not None:
            sc.pp.filter_cells(ann, min_genes=cell_filter_ngene[0])
            print(f"\n\tMinimum={cell_filter_ngene[0]}\tCount: {ann.n_obs}")
        else:
            print("\n\tNo Minimum")
        if cell_filter_ngene[1] is not None:
            sc.pp.filter_cells(ann, max_genes=cell_filter_ngene[1])
            print(f"\n\tMaximum={cell_filter_ngene[1]}\tCount: {ann.n_obs}")
        print("\n\t*** Filtering cells based on # of reads...")
        if cell_filter_ncounts[0] is not None:
            sc.pp.filter_cells(ann, min_counts=cell_filter_ncounts[0])
            print(f"\n\tMinimum={cell_filter_ncounts[0]}\tCount: {ann.n_obs}")
        else:
            print("\n\tNo Minimum")
        if cell_filter_ncounts[1] is not None:
            sc.pp.filter_cells(ann,
                # min_genes=None, max_genes=None, min_counts=None,
                max_counts=cell_filter_ncounts[1])
            print(f"\n\tMaximum={cell_filter_ncounts[1]}\tCount: {ann.n_obs}")
        else:
            print("\n\tNo Maximum")
        print("\n\t*** Filtering genes based on # of cells in which they "
              "are expressed...")
        if gene_filter_ncell[0] is not None:
            sc.pp.filter_genes(ann, min_cells=gene_filter_ncell[0])
            print(f"\n\tMinimum={gene_filter_ncell[0]}\tCount: {ann.n_obs}")
        else:
            print("\n\tNo minimum")
        if gene_filter_ncell[1] is not None:
            sc.pp.filter_genes(ann, max_cells=gene_filter_ncell[1])
            print(f"\n\tMaximum={gene_filter_ncell[1]}\tCount: {ann.n_obs}")
        else:
            print("\n\tNo maximum")
        print(f"\nPost-Filtering Cell Count: {ann.n_obs}")
    return ann


def remove_batch_effects(adata, col_cell_type="leiden", 
                         col_batch="orig.ident", plot=True, **kws_train):
    """Remove batch effects (IN PROGRESS)."""
    if not kws_train:
        kws_train = dict(max_epochs=100, batch_size=32, 
                         early_stopping=True, early_stopping_patience=25)
    train = adata.copy()
    train.obs["cell_type"] = train.obs[col_cell_type].tolist()
    train.obs["batch"] = train.obs[col_batch].tolist()
    if plot is True:
        sc.pl.umap(train, color=[col_batch, col_cell_type], 
                   wspace=.5, frameon=False)
    print(f"\n<<< PREPARING DATA >>>") 
    pt.tl.SCGEN.setup_anndata(train, batch_key="batch", 
                              labels_key="cell_type")  # prepare AnnData
    model = pt.tl.SCGEN(train)
    print(f"\n<<< TRAINING >>>") 
    model.train(**kws_train)  # training
    print(f"\n<<< CORRECTING FOR BATCH EFFECTS >>>") 
    corrected_adata = model.batch_removal()  # batch correction
    if plot is True:
        sc.pp.neighbors(corrected_adata)
        sc.tl.umap(corrected_adata)
        sc.pl.umap(corrected_adata, color=[col_batch, col_cell_type], 
                   wspace=0.4, frameon=False)
    return corrected_adata
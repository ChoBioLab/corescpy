import squidpy as sq
import liana
from liana.method import cellphonedb
import decoupler as dc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import scanpy as sc
import omnipath
import corneto
import traceback
import matplotlib.pyplot as plt
from warnings import warn
from crispr.visualization import plot_receptor_ligand, square_grid
from crispr.utils import create_pseudobulk, merge
import pandas as pd
import numpy as np


def analyze_receptor_ligand(
    adata, method="liana", n_jobs=4, seed=1618, col_condition=None, 
    col_cell_type=None, col_sample_id=None, col_subject=None, 
    key_control=None, key_treatment=None,
    key_sources=None, key_targets=None, layer="log1p", layer_counts="counts",
    min_prop=0, min_count=0, min_total_count=0, kws_deseq2=None,
    n_perms=10, p_threshold=0.05, figsize=None, remove_ns=True, top_n=20,
    cmap="magma", kws_plot=None, resource="CellPhoneDB", copy=True, **kwargs):
    """Perform receptor-ligand analysis."""
    if copy is True:
        adata = adata.copy()
    kws_plot = {} if kws_plot is None else {**kws_plot}
    figs = {}
    kws_deseq2 = merge({"n_jobs": n_jobs, "refit_cooks": True, 
                        "p_threshold": p_threshold}, kws_deseq2)  # deseq2 kws
    if layer is not None:
        adata.X = adata.layers[layer].copy()
        
    # Squidpy Method
    if method == "squidpy":
        res = sq.gr.ligrec(
            adata, n_perms=n_perms, cluster_key=col_cell_type, copy=True,
            transmitter_params={"categories": "ligand"}, 
            receiver_params={"categories": "receptor"}, kws_plot=None, 
            interactions_params={"resources": resource}, **kwargs)
        fig = sq.pl.ligrec(
            res, alpha=p_threshold, source_groups=key_sources, **kws_plot,
            target_groups=key_targets, remove_nonsig_interactions=remove_ns,
            # pvalue_threshold=p_threshold, 
            **kws_plot)  # plot 
        
    # Liana Method
    else:
        resource = resource.lower()  # all Liana resources are lowercase
        kwargs = {**dict(use_raw=False, return_all_lrs=True, 
                         verbose=True, key_added="liana_res"), **kwargs}
        cellphonedb(adata, groupby=col_cell_type, n_jobs=n_jobs,
                    resource_name=resource, seed=seed, **kwargs)  # run l-r
        res = {"liana_res": adata.uns[kwargs["key_added"]]}  # store results
        kws = {**dict(
            cmap=cmap, p_threshold=p_threshold, top_n=top_n, figsize=figsize,
            key_sources=key_sources, key_targets=key_targets), **kws_plot}
        if col_condition is not None:
            # Differential Expression Analysis (DEA)
            pdata = create_pseudobulk(
                adata.copy(), col_cell_type, col_sample_id=col_sample_id, 
                layer=layer_counts, mode="sum", kws_process=True)  # bulk
            cgs = pdata.var.index.names[0]  # gene symbols column/index name
            res["dea_results"], figs["dea"] = calculate_dea_deseq2(
                pdata, col_cell_type, col_condition, key_control, 
                key_treatment, col_subject=col_subject, min_prop=min_prop, 
                min_count=min_count, min_total_count=min_total_count,
                layer_counts=layer_counts, **kws_deseq2)  # run DEA
            res["dea_df"] = pd.concat([res["dea_results"][t].results_df if (
                res) else None for t in res["dea_results"]], keys=res[
                    "dea_results"], names=[col_cell_type, cgs]).reset_index(
                        ).set_index(cgs)  # merge results dfs of cell types
            res["lr_res"] = liana.mu.df_to_lr(
                adata[adata.obs[col_condition] == key_treatment].copy(),  # tx
                res["dea_df"], col_cell_type, stat_keys=[
                    "stat", "pvalue", "padj"], use_raw=False, layer=layer, 
                verbose=True, complex_col="stat", expr_prop=min_prop, 
                return_all_lrs=True, resource_name="consensus").sort_values(
                    "interaction_stat", ascending=False
                    )  # merge DEA results & scRNA treatment group data
        else:
            res["lr_res"] = None
        try:
            figs["lr"] = plot_receptor_ligand(
                adata=adata, lr_res=res["lr_res"], **kws)  # plots
        except Exception as err:
            print(traceback.format_exc())
            figs["lr"] = err
        for x in figs["lr"]:
            print(figs["lr"][x])
    return res, figs
    

def calculate_dea_deseq2(
    pdata, col_cell_type, col_condition, key_control, key_treatment, top_n=20,
    n_jobs=4, layer_counts="counts", col_subject=None, col_gene_symbols=None, 
    min_prop=0, min_count=0, min_total_count=0, p_threshold=0.05, **kwargs):
    """
    Calculate DEA based on Liana tutorial usage of DESeq2.
    
    Extra keyword arguments are passed to DeseqDataset, except for
    "alt_hypothesis" and "lfc_null," which are passed to DeseqStats.
    """
    res, quiet, figsize = {}, True, kwargs.pop("figsize", (20, 20))
    lfc_null = kwargs.pop("lfc_null", 0.0)  # DESeqStats argument
    alt_hypothesis = kwargs.pop("alt_hypothesis", None)  # DESeqStats argument
    filt_c, filt_i = [kwargs.pop(x, True) for x in [
        "cooks_filter", "independent_filter"]]  # DESeqStats filter arguments
    if col_gene_symbols is None:  # if gene name column unspecified...
        col_gene_symbols = pdata.var.index.names[0]  # ...index=gene names
    facs = col_condition if not col_subject else [col_condition, col_subject]
    pdata = pdata[~pdata.obs[col_condition].isna()].copy()  # no condition NAs 
    cts = pdata.obs.groupby(col_cell_type).apply(
            lambda x: all((k in list(x[col_condition]) for k in [
                key_treatment, key_control])) and x.shape[0] > 3).replace(
                    False, np.nan).dropna().index.values  # cell types 
    
    # Run DEA for Each Cell Type
    for t in cts:  # iterate cell types from above: w/ both conditions & n > 3
        
        # Set Up Pseudo-Bulk Data
        psub = pdata[pdata.obs[col_cell_type] == t]  # subset to cell type
        genes = dc.filter_by_expr(  # edgeR-based filtering function
            psub, group=col_condition, min_count=min_count, min_prop=min_prop,
            min_total_count=min_total_count)  # genes with enough counts/reads
        psub = psub[:, genes].copy()  # filter data ~ those genes
        
        # Skip Cell Type if Not Enough Data or Doesn't Contain Both Conditions
        if any(psub[psub.obs[col_condition].isin([key_control, key_treatment])
                    ].obs.value_counts() < 2):
            res[t] = None
            warn(f"Skipping {t} DEA: levels missing in {col_condition}")
            continue  # skip if doesn't contain both conditions or n < 4
        
        # Perform DESeq2
        psub.X = psub.layers[layer_counts].copy()  # counts layer
        print(psub)
        # dds = DeseqDataSet(
        #     adata=psub, design_factors=facs, quiet=quiet, n_cpus=n_jobs,
        #     ref_level=[col_condition, key_control], **kwargs)  # DESeq adata
        dds = DeseqDataSet(
            adata=psub, design_factors=facs, quiet=quiet,
            ref_level=[col_condition, key_control], **kwargs)  # DESeq adata
        dds.deseq2()  # estimate dispersion & logfold change
        res[t] = DeseqStats(dds, alpha=p_threshold, contrast=[
            col_condition, key_treatment, key_control], quiet=quiet, 
                            cooks_filter=filt_c, independent_filter=filt_i)
        res[t].quiet = quiet
        res[t].summary()  # Wald test summary (print)
        res[t].lfc_shrink(
            coeff=f"{col_condition}_{key_treatment}_vs_{key_control}")
    print(res)
    p_dims = square_grid(len(res))
    fig, axs = plt.subplots(p_dims[0], p_dims[1], figsize=figsize)
    for i, x in enumerate(res):
        dc.plot_volcano_df(
            res[x].results_df, x="log2FoldChange", y="padj", top=top_n, 
            sign_thr=0.05, 
            lFCs_thr=0.5, sign_limit=None, lFCs_limit=None, dpi=200, 
            ax=axs.ravel()[i])
        axs.ravel()[i].set_title(x)
    fig.tight_layout()
    return res, fig


def analyze_causal_network(adata, col_condition, key_control, key_treatment,
                           col_cell_type, key_source, key_target, 
                           dea_df=None, col_sample_id=None,
                           resource_name="cellphonedb",
                           layer="log1p", layer_counts="counts",
                           expr_prop=0.1, min_n_ulm=5, col_gene_symbols=None,
                           node_cutoff=0.1, max_penalty=1, min_penalty=0.01,
                           edge_penalty=0.01, max_seconds=60*3, 
                           solver="scipy", top_n=10, verbose=False):
    """
    Analyze causal network (adapted from Liana tutorial).
    """
    figs = {}
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    if col_gene_symbols is None:
        col_gene_symbols = adata.var.index.names[0]
    
    # Pseudo-Bulk -> DEA
    if dea_df is None:
        pdata = dc.get_pseudobulk(
            adata, sample_col=col_sample_id, groups_col=col_cell_type,
            mode="sum", layer=layer_counts)  # pseudobulk
        dea_df = calculate_dea_deseq2(
            pdata, col_cell_type, col_condition, key_control, key_treatment,
            min_count=5, min_total_count=10, quiet=True)  # DEA results
        dea_df = pd.concat(dea_df).reset_index().rename(
            columns={"level_0": col_cell_type}).set_index(
                col_gene_symbols)  # format results into dataframe
    dea_df = dea_df.rename_axis(col_gene_symbols).copy()
    lr_res = liana.mu.df_to_lr(
        adata, dea_df=dea_df.copy(), resource_name="consensus",
        return_all_lrs=False, expr_prop=expr_prop, # filter interactions
        groupby=col_cell_type, stat_keys=["stat", "pvalue", "padj"], 
        use_raw=False, complex_col="stat", verbose=True).sort_values(
            "interaction_stat", ascending=False)
    
    # Subset to Treatment Condition
    adata = adata[adata.obs[col_condition] == key_treatment].copy()

    # Select Cell Types of Interest
    lr_stats = lr_res[lr_res["source"].isin([key_source]) & lr_res[
        "target"].isin([key_target])].copy()  # subset to chosen source-target
    lr_stats = lr_stats.sort_values(
        "interaction_stat", ascending=False, key=abs)  # sort by statistic
    
    # Select Starting Nodes (Receptors) for the Network ~ Interaction Effects
    lr_dict = lr_stats.set_index("receptor")["interaction_stat"].to_dict()
    lr_n = dict(sorted({**lr_dict}.items(), key=lambda item: abs(
        item[1]), reverse=True))  # sort
    scores_in = {k: v for i, (k, v) in enumerate(lr_n.items()) if i < top_n}
    
    # Top Transcription Factor Selection
    dea_wide = dea_df[[col_cell_type, "stat"]].reset_index().set_index(
        col_gene_symbols).pivot(index=col_cell_type, columns=col_gene_symbols, 
                                values="stat").fillna(0)  # long to wide
    net = dc.get_collectri()  # get TF regulons
    ests, pvals = dc.run_ulm(
        mat=dea_wide, net=net, min_n=min_n_ulm)  # enrichment analysis
    tfs = ests.copy().loc[key_target].to_dict()  # target TFs
    scores_out = {k: v for i, (k, v) in enumerate(tfs.items()) if i < top_n}

    # PPI Network
    ppis = omnipath.interactions.OmniPath().get(genesymbols=True)
    ppis["mor"] = ppis["is_stimulation"].astype(int) - ppis[
        "is_inhibition"].astype(int)  # dummy-code directionality (I think?)
    ppis = ppis[(ppis["mor"] != 0) & (ppis["curation_effort"] >= 3) & ppis[
        "consensus_direction"]]  # keep only high quality
    input_pkn = ppis[["source_genesymbol", "mor", "target_genesymbol"]]
    input_pkn.columns = ["source", "mor", "target"]
    scores_in = {k: v for i, (k, v) in enumerate(lr_n.items()) if i < top_n}
    
    # Prior Knowledge Network
    prior_graph = liana.mt.build_prior_network(
        input_pkn, scores_in, scores_out, verbose=True)
    
    # Node Weights = GEX Proportions w/i Target Cell Type
    temp = adata[adata.obs[col_cell_type] == key_target].copy()
    node_weights = pd.DataFrame(temp.X.getnnz(axis=0) / temp.n_obs, 
                                index=temp.var_names)
    node_weights = node_weights.rename(columns={0: 'props'})
    node_weights = node_weights["props"].to_dict()
    
    # Find Causal Network
    df_res, problem = liana.mt.find_causalnet(
        prior_graph, scores_in, scores_out, node_weights,
        node_cutoff=node_cutoff,  # max_penalty to any node < this of cells
        max_penalty=max_penalty, min_penalty=min_penalty, solver=solver,
        max_seconds=max_seconds, edge_penalty=edge_penalty, verbose=verbose)
    fig = corneto.methods.carnival.visualize_network(df_res)
    fig.view()
    return df_res, problem, fig
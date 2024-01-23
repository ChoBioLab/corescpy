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
import warnings
from crispr.visualization import plot_receptor_ligand
from crispr.utils import create_pseudobulk
import pandas as pd
import numpy as np


def analyze_receptor_ligand(
    adata, method="liana", col_condition=None, 
    key_control=None, key_treatment=None,
    col_sample_id=None, col_subject=None, 
    min_prop=0, min_count=0, min_total_count=0,
    layer="log1p", layer_counts="counts",
    key_sources=None, key_targets=None, col_cell_type=None, 
    n_perms=10, p_threshold=0.05, figsize=None, remove_ns=True, top_n=20,
    cmap="magma", kws_plot=None, resource="CellPhoneDB", copy=True, **kwargs):
    """Perform receptor-ligand analysis."""
    if copy is True:
        adata = adata.copy()
    if kws_plot is None:
        kws_plot = {}
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    if any((any(adata.obs[col_condition] == x) for x in ["Overall", "all"])):
        raise ValueError("'Overall' and 'all' are reserved by this function "
                         f"but exist in `.obs['{col_condition}']. "
                         "Please rename those entries.")
        
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
            **{**dict(kws_plot if kws_plot else {})})  # plot 
        
    # Liana Method
    else:
        kwargs = {**dict(use_raw=False, return_all_lrs=True, 
                         verbose=True, key_added="liana_res"), **kwargs}
        cellphonedb(adata, groupby=col_cell_type, **kwargs)  # run cellphonedb
        kws = {**dict(
            cmap=cmap, p_threshold=p_threshold, top_n=top_n, figsize=figsize,
            key_sources=key_sources, key_targets=key_targets), **kws_plot}
        res = {"liana_res": adata.uns[kwargs["key_added"]]}
        if col_condition is not None:
            # Differential Expression Analysis
            pdata = create_pseudobulk(
                adata, col_cell_type, col_sample_id=col_sample_id, 
                layer=layer_counts, mode="sum")  # pseudo-bulk data
            res["dea_results"] = calculate_dea_deseq2(
                pdata, col_cell_type, col_condition, 
                key_control, key_treatment, col_subject=col_subject,
                min_prop=min_prop, min_count=min_count, layer=layer_counts,
                min_total_count=min_total_count)  # perform DEA
            atx = adata[adata.obs[col_condition] == key_treatment
                        ].copy()  # subset to treatment group
            res["lr_res"] = liana.mu.df_to_lr(
                atx, dea_df=res["dea_results"], resource_name="consensus", 
                expr_prop=min_prop, groupby=col_cell_type, verbose=True, 
                stat_keys=["stat", "pvalue", "padj"], use_raw=False, 
                complex_col="stat", return_all_lrs=False).sort_values(
                    "interaction_stat", ascending=False)  # merge DEA results
        try:
            fig = plot_receptor_ligand(
                adata=adata, lr_res=res["lr_res"], **kws)  # plot
        except Exception as err:
            print(traceback.format_exc())
            fig = err
        for x in fig:
            print(fig[x])
    return res, fig
    

def calculate_dea_deseq2(pdata, col_cell_type, col_condition,
                         key_control, key_treatment, layer="counts", 
                         col_subject=None, min_prop=0, min_count=0, 
                         min_total_count=0, col_gene_symbols=None):
    """Calculate DEA based on Liana tutorial usage of DESeq2."""
    dea_results, quiet = {}, True
    if col_gene_symbols is None:  # if gene name column unspecified...
        col_gene_symbols = pdata.var.index.names[0]  # ...index=gene names
    facs = col_condition if not col_subject else [col_condition, col_subject]
    
    # Run DEA for Each Cell Type
    for cell_group in pdata.obs[col_cell_type].unique():
        psub = pdata[pdata.obs[col_cell_type] == cell_group].copy()  # subset
        if psub.obs.shape[0] < 4 or any((x not in psub.obs[
            col_condition].dropna() for x in [key_control, key_treatment])):
            dea_results[cell_group] = None  # store results as None
            warnings.warn("Skipping DEA calculations for {cell_group}: "
                          "doesn't have all levels of {col_condition}.")
            continue  # skip if doesn't contain both levels of contrast

        # Filter Genes by edgeR-like Thresholds
        genes = dc.filter_by_expr(
            psub, group=col_condition, min_count=min_count, min_prop=min_prop,
            min_total_count=min_total_count)  # filter ~ counts, reads
        psub = psub[:, genes].copy()  # subset by filtered genes
        if psub.obs.shape[0] < 3 or any((x not in list(psub.obs[
            col_condition].dropna()) for x in [key_control, key_treatment])):
            dea_results[cell_group] = None
            warnings.warn("Skipping DEA calculations for {cell_group}: "
                          "doesn't have all levels of {col_condition}.")
            continue  # skip if doesn't contain both levels of contrast

        # Build DESeq2 object
        psub.X = psub.layers[layer].copy()
        dds = DeseqDataSet(
            adata=psub, design_factors=facs, quiet=quiet,
            ref_level=[col_condition, key_control], refit_cooks=True)

        # Compute
        dds.deseq2()
        stat_res = DeseqStats(dds, contrast=[
            col_condition, key_treatment, key_control], quiet=quiet)  # tx v c
        stat_res.quiet = quiet
        stat_res.summary()  # Wald test
        stat_res.lfc_shrink(
            coeff=f"{col_condition}_{key_treatment}_vs_{key_control}")
        dea_results[cell_group] = stat_res.results_df
    dea_results = pd.concat(dea_results)
    dea_results = dea_results.reset_index().rename(columns={
        "level_0": col_cell_type}).set_index(col_gene_symbols)
    return dea_results


def analyze_causal_network(adata, col_condition, key_control, key_treatment,
                           col_cell_type, key_source, key_target, 
                           dea_results=None, col_sample_id=None,
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
    if dea_results is None:
        pdata = dc.get_pseudobulk(
            adata, sample_col=col_sample_id, groups_col=col_cell_type,
            mode="sum", layer=layer_counts)  # pseudobulk
        dea_results = calculate_dea_deseq2(
            pdata, col_cell_type, col_condition, key_control, key_treatment,
            min_count=5, min_total_count=10, quiet=True)  # DEA results
        dea_results = pd.concat(dea_results).reset_index().rename(
            columns={"level_0": col_cell_type}).set_index(
                col_gene_symbols)  # format results into dataframe
    dea_results = dea_results.rename_axis(col_gene_symbols).copy()
    lr_res = liana.mu.df_to_lr(
        adata, dea_df=dea_results.copy(), resource_name="consensus",
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
        item[1]), reverse=True))
    scores_in = {k: v for i, (k, v) in enumerate(lr_n.items()) if i < top_n}
    
    # Top Transcription Factor Selection
    dea_wide = dea_results[[col_cell_type, "stat"]].reset_index(0).pivot(
        index=col_cell_type, columns=col_gene_symbols, 
        values="stat").fillna(0)  # long to wide data format
    net = dc.get_collectri()  # get TF regulons
    ests, pvals = dc.run_ulm(mat=dea_wide, net=net,
                             min_n=min_n_ulm)  # enrichment analysis
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
import squidpy as sq
import liana
from liana.method import cellphonedb
import decoupler
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import traceback
import crispr as cr
import pandas as pd
import numpy as np

def analyze_receptor_ligand(
    adata, method="liana", col_condition=None, layer="log1p",
    key_sources=None, key_targets=None, col_cell_type=None, n_perms=10, 
    p_threshold=0.05, figsize=None, remove_ns=True, top_n=20, 
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
        try:
            fig = cr.pl.plot_receptor_ligand(adata=adata, **kws)  # plot
        except Exception as err:
            print(traceback.format_exc())
            fig = err
        print(fig)
        res = adata.uns[kwargs["key_added"]]
        if col_condition is not None:
            fig, res = {"Overall": fig}, {"Overall": res}
            for c in adata.obs[col_condition].unique():  # iterate conditions
                anc = adata[adata.obs[col_condition] == c]  # data
                cellphonedb(anc, groupby=col_cell_type, **kwargs)  # run cpdb
                try:
                    fig[c] = cr.pl.plot_receptor_ligand(
                        adata=anc, title=f"{col_condition} = {c}", **kws)
                except Exception as err: 
                    fig[c] = err
                print(fig[c])
                res[c] = anc.uns[kwargs["key_added"]]
            res["all"] = pd.concat([res[c] for c in res], keys=res.keys(), 
                                    names=[col_condition, ""])  # all results
    return res, fig


def analyze_causal_network(adata, col_condition, key_control, key_treatment,
                           col_cell_type, key_source, key_target, 
                           pdata=None, col_sample_id=None,
                           layer="log1p", expr_prop=0.1,
                           node_cutoff=0.1, max_penalty=1, min_penalty=0.01,
                           edge_penalty=0.01, max_seconds=60*3, 
                           solver="scipy", top_n=10, verbose=False):
    """
    Analyze causal network (adapted from Liana tutorial).
    """
    figs = {}
    if layer is not None:
        adata.X = adata.layers[layer].copy()
    
    # Pseudo-Bulk -> DEA
    if pdata is None:
        pdata = decoupler.get_pseudobulk(
            adata, sample_col=col_sample_id, groups_col=col_cell_type,
            mode="sum", layer=cr.pp.get_layer_dict()["counts"])  #  pseudobulk
    dea_results = calculate_dea_deseq2(
        pdata, col_cell_type, col_condition, key_control, key_treatment,
        min_count=5, min_total_count=10, quiet=True)  # DEA results
    dea_df = pd.concat(dea_results).reset_index().rename(
        columns={"level_0": col_cell_type}).set_index(
            adata.var.index.names[0])  # format results into dataframe
    
    # Subset to Treatment Condition
    adata = adata[adata.obs[col_condition] == key_treatment].copy()
    
    # Ligand_Receptor Interactions
    lr_res = liana.multi.df_to_lr(
        adata, dea_df=dea_df, resource_name="consensus",
        expr_prop=expr_prop, # filter interactions
        groupby=col_cell_type, stat_keys=["stat", "pvalue", "padj"], 
        use_raw=False, complex_col="stat", verbose=True, return_all_lrs=False)
    lr_res = lr_res.sort_values("interaction_stat", ascending=False)
    try:  # try plotting
        figs["ligrec"] = cr.pl.plot_receptor_ligand(
            adata=adata, key_sources=[key_source], key_targets=[key_target])
    except Exception as err:
        figs["ligrec"] = err

    # Select Cell Types of Interest
    lr_stats = lr_res[lr_res["source"].isin([key_source]) & lr_res[
        "target"].isin([key_target])].copy()
    lr_stats = lr_stats.sort_values("interaction_stat", 
                                    ascending=False, key=abs)
    
    # Select Starting Nodes (Receptors) for the Network ~ Interaction Effects
    lr_dict = lr_stats.set_index('receptor')["interaction_stat"].to_dict()
    lr_n = dict(sorted({**lr_dict}.items(), key=lambda item: abs(
        item[1]), reverse=True))
    scores_in = {k: v for i, (k, v) in enumerate(lr_n.items()) if i < top_n}

    # PPI Network
    ppis = op.interactions.OmniPath().get(genesymbols = True)
    ppis["mor"] = ppis["is_stimulation"].astype(int) - ppis[
        "is_inhibition"].astype(int)  # dummy-code directionality (I think?)
    ppis = ppis[(ppis["mor"] != 0) & (ppis["curation_effort"] >= 3) & ppis[
        "consensus_direction"]]  # keep only high quality

    input_pkn = ppis[["source_genesymbol", "mor", "target_genesymbol"]]
    input_pkn.columns = ["source", "mor", "target"]
    
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
    return df_res, problem
    

def calculate_dea_deseq2(pdata, col_cell_type, col_condition, 
                         key_control, key_treatment, col_subject=None,
                         min_prop=0.3, min_count=3, min_total_count=10, 
                         quiet=True):
    """Calculate DEA based on Liana tutorial usage of DESeq2."""
    dea_results = {}
    facs = col_condition if not col_subject else [col_condition, col_subject]
    for cell_group in pdata.obs[col_cell_type].unique():
        psub = pdata[pdata.obs[col_cell_type] == cell_group].copy()  # subset

        # Filter Genes by edgeR-like Thresholds
        genes = decoupler.filter_by_expr(
            psub, group=col_condition, min_count=min_count, min_prop=min_prop,
            min_total_count=min_total_count)  # filter ~ minimum counts, reads
        psub = psub[:, genes].copy()  # subset by filtered genes
        if any((x not in psub.obs[col_condition] for x in [
            key_control, key_treatment])):
            next  # skip if doesn't contain both levels of contrast

        # Build DESeq2 object
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
    return dea_results

import squidpy as sq
import liana
from liana.method import cellphonedb
import decoupler as dc
import omnipath
# import corneto
import traceback
from corescpy.visualization import plot_receptor_ligand
from corescpy.utils import merge
from corescpy.analysis import calculate_dea_deseq2
import pandas as pd


def analyze_receptor_ligand(adata, method="liana", n_jobs=4, seed=1618,
                            layer="log1p", layer_counts="counts", dea_df=None,
                            col_condition=None, col_cell_type=None,
                            col_sample_id=None, col_subject=None,
                            key_control=None, key_treatment=None,
                            key_sources=None, key_targets=None,
                            min_prop=0, min_count=0, min_total_count=0,
                            kws_deseq2=None, n_perms=10, p_threshold=0.05,
                            figsize=None, remove_ns=True, top_n=20,
                            cmap="magma", kws_plot=None, copy=True, plot=True,
                            resource="CellPhoneDB", **kwargs):
    """Perform receptor-ligand analysis."""
    if copy is True:
        adata = adata.copy()
    res_keys = ["squidpy", "liana_res", "lr_dea_res", "dea_results", "dea_df"]
    figs, res = {}, dict(zip(res_keys, [None] * len(res_keys)))  # for output
    kws_plot = {} if kws_plot is None else {**kws_plot}
    kws_deseq2 = merge({"n_jobs": n_jobs, "refit_cooks": True,
                        "p_threshold": p_threshold}, kws_deseq2)  # deseq2 kws
    if layer is not None:
        adata.X = adata.layers[layer].copy()

    # Squidpy Method
    if method == "squidpy":
        res["squidpy"] = sq.gr.ligrec(
            adata, n_perms=n_perms, cluster_key=col_cell_type, copy=True,
            transmitter_params={"categories": "ligand"},
            receiver_params={"categories": "receptor"}, kws_plot=None,
            interactions_params={"resources": resource}, **kwargs)
        figs["squidpy"] = sq.pl.ligrec(
            res, alpha=p_threshold, source_groups=key_sources, **kws_plot,
            target_groups=key_targets,  # pvalue_threshold=p_threshold
            remove_nonsig_interactions=remove_ns, **kws_plot)  # plot

    # Liana Method
    else:
        resource = resource.lower()  # all Liana resources are lowercase
        kwargs = {**dict(use_raw=False, return_all_lrs=True,
                         verbose=True, key_added="liana_res"), **kwargs}
        cellphonedb(adata, groupby=col_cell_type, n_jobs=n_jobs,
                    resource_name=resource, seed=seed, **kwargs)  # run l-r
        res["liana_res"] = adata.uns[kwargs["key_added"]].copy()  # L-R result
        kws = {**dict(
            cmap=cmap, p_threshold=p_threshold, top_n=top_n, figsize=figsize,
            key_sources=key_sources, key_targets=key_targets), **kws_plot}

    # DEA + Liana
    if dea_df is not None and method.lower() == "liana":
        try:  # merge DEA results & scRNA treatment group data
            res["lr_dea_res"] = liana.mu.df_to_lr(
                adata[adata.obs[col_condition] == key_treatment].copy(),  # tx
                dea_df, col_cell_type, stat_keys=[
                    "stat", "pvalue", "padj"], use_raw=False, layer=layer,
                verbose=True, complex_col="stat", expr_prop=min_prop,
                return_all_lrs=True, resource_name="consensus").sort_values(
                    "interaction_stat", ascending=False)
        except Exception:
            print(traceback.format_exc(), "Liana + DEA failed!\n\n",)

    # Plotting
    if plot is True:
        try:
            figs["lr"] = plot_receptor_ligand(
                adata=adata, lr_dea_res=res["lr_dea_res"], **kws)  # plots
        except Exception as err:
            print(traceback.format_exc(),
                  "\n\nLigand-receptor plotting failed!")
            figs["lr"] = err
    return res, adata, figs


# def analyze_cci_spatial(adata, col_cell_type=None, n_spots=125,
#                         organism="human", resource="connectomeDB2020_lit",
#                         distance=None, min_spots=20,
#                         n_pairs=10000, n_top=50,
#                         n_jobs=8, stats="all", layer="counts",
#                         plot_lr=None,
#                         pval_adj_cutoff=None, adj_method=None, **kwargs):
#     """
#     Perform spatial cell-cell communication & ligand-receptor analysis.
#     """
#     # Process Arguments
#     if isinstance(stats, str) and stats.lower().strip() == "all":
#         stats = ["lr_scores", "p_vals", "p_adjs", "-log10(p_adjs)"]
#     library_id = kwargs.pop("library_id", list(
#         adata.uns["spatial"].keys())[0])
#     scale = kwargs.pop

#     # Make Compatible with Hard-Coded Column in stlearn Code
#     # max_coor = np.max(adata.obsm["spatial"])
#     # scale = 2000 / max_coor
#     scale = kwargs.pop("scale")
#     quality = kwargs.pop("key_image", "hires")
#     spot_diameter_fullres = kwargs.pop("spot_diameter_fullres", 15)
#     if "spatial" in adata.obsm:
#         adata.obs.loc[:, "imagerow"] = adata.obsm["spatial"][:, 0] * scale
#         adata.obs.loc[:, "imagecol"] = adata.obsm["spatial"][:, 1] * scale
#     if "scalefactors" not in adata.uns["spatial"]:
#         adata.uns["spatial"][library_id]["scalefactors"] = {}
#         adata.uns["spatial"][library_id]["scalefactors"][
#             "tissue_" + quality + "_scalef"] = scale
#         adata.uns["spatial"][library_id]["scalefactors"][
#             "spot_diameter_fullres"] = spot_diameter_fullres
#         adata.uns["spatial"][library_id]["use_quality"] = quality

#     # Process Data
#     adata.X = adata.layers[layer].copy()
#     st.pp.normalize_total(adata)

#     # Create Spot Grid
#     grid = st.tl.cci.grid(adata, n_row=n_spots, n_col=n_spots,
#                           use_label=col_cell_type)

#     # Plot: Compare Clusters to Created Spots
#     fig, axes = plt.subplots(ncols=2, figsize=(20, 8))
#     st.pl.cluster_plot(grid, use_label=col_cell_type, size=10, ax=axes[0],
#                        show_plot=False)
#     st.pl.cluster_plot(adata, use_label=col_cell_type,
#                        ax=axes[1], show_plot=False)
#     axes[0].set_title(f"Grid: Dominant Spots")
#     axes[1].set_title(f"Cell {col_cell_type} Labels")
#     plt.show()

#     # Plot Cell Type Locations & Spot Maxima/Proportions by Cell Type
#     for g in list(grid.obs[col_cell_type].cat.categories):
#         fig, axes = plt.subplots(ncols=3, figsize=(20,8))
#         group_props = grid.uns[col_cell_type][g].values
#         grid.obs["Group"] = group_props
#         st.pl.feat_plot(grid, feature="Group", ax=axes[0], show_plot=False,
#                         vmax=1, show_color_bar=False)
#         st.pl.cluster_plot(grid, use_label=col_cell_type,
#                            list_clusters=[g], ax=axes[1], show_plot=False)
#         st.pl.cluster_plot(adata, use_label=col_cell_type,
#                            list_clusters=[g], ax=axes[2], show_plot=False)
#         axes[0].set_title(f"Grid {g} Proportions (Maximum = 1)")
#         axes[1].set_title(f"Grid {g} Maximum Spots")
#         axes[2].set_title(f"Individual Cell {g}")
#         plt.show()

#     # Run Analysis
#     lrs = st.tl.cci.load_lrs([resource], species=organism)
#     st.tl.cci.run(
#         grid, lrs, min_spots=min_spots, distance=distance,
#         n_pairs=n_pairs, n_cpus=n_jobs)
#     if pval_adj_cutoff is not None or adj_method is not None:  # adjust p?
#         st.tl.cci.adj_pvals(
#             grid, correct_axis="spot", pval_adj_cutoff=pval_adj_cutoff,
#             adj_method=adj_method)  # optionally, adjust p-values
#     print(grid.uns["lr_summary"])

#     # QC Plots
#     fig, axes = st.pl.cci_check(grid, col_cell_type, figsize=(16, 5))
#     fig.suptitle("CCI Check: Interactions Shouldn't Correlate Much "
#                  "with Cell Type Frequency if Well-Controlled for")
#     st.pl.lr_diagnostics(grid, figsize=(10, 2.5))

#     # Results Plots
#     st.pl.lr_summary(grid, n_top=n_top, figsize=(10, 3))  # summary plot
#     if plot_lr is True or isinstance(
#             plot_lr, (int, float)):  # if pairs unspecified, or want top N
#         plot_lr = 3 if plot_lr is None else int(plot_lr)  # top 3 = default
#         plot_lr = grid.uns["lr_summary"].index.values[:plot_lr]  # best L-Rs
#     if plot_lr not in [None, False]:  # if wanted these plots...
#         fig, axes = plt.subplots(ncols=len(stats), nrows=len(plot_lr),
#                                  figsize=(12, 6))
#         for r, x in enumerate(plot_lr):  # iterate ligand-receptors
#             for c, stat in enumerate(stats):  # iterate statistics
#                 st.pl.lr_result_plot(grid, use_result=stat, use_lr=x,
#                                      show_color_bar=False, ax=axes[r, c])
#                 axes[r, c].set_title(f"{x} {stat}")

#     # Gene Expression Plots
#     if plot_lr is not None:
#         genes = functools.reduce(lambda i, j: list(i) + list(j),
#                                 [i.split("_") for i in plot_lr])
#         for g in genes:
#             fig, axes = plt.subplots(ncols=2, figsize=(20, 5))
#             st.pl.gene_plot(grid, gene_symbols=g, ax=axes[0],
#                             show_color_bar=False, show_plot=False)
#             st.pl.gene_plot(adata, gene_symbols=g, ax=axes[1],
#                             show_color_bar=False, show_plot=False, vmax=80)
#             axes[0].set_title(f"Grid {g} Expression")
#             axes[1].set_title(f"Cell {g} Expression")
#             plt.show()
#     return grid, grid.uns["lr_summary"]


def analyze_causal_network(adata, col_condition, key_control, key_treatment,
                           col_cell_type, key_source, key_target, dea_df=None,
                           col_gene_symbols=None, col_sample_id=None,
                           resource_name="cellphonedb", top_n=10,
                           layer="log1p", layer_counts="counts",
                           expr_prop=0.1, min_n_ulm=5, node_cutoff=0.1,
                           max_penalty=1, min_penalty=0.01, edge_penalty=0.01,
                           max_seconds=60*3, solver="scipy", verbose=False):
    """Analyze causal network (adapted from Liana tutorial)."""
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
    lr_dea_res = liana.mu.df_to_lr(
        adata, dea_df=dea_df, groupby=col_cell_type, return_all_lrs=False,
        use_raw=False, stat_keys=["stat", "pvalue", "padj"], verbose=True,
        resource_name="consensus", expr_prop=expr_prop, complex_col="stat",
        ).sort_values("interaction_stat", ascending=False)  # merge DEA, scRNA

    # Subset to Treatment Condition
    adata = adata[adata.obs[col_condition] == key_treatment].copy()

    # Select Cell Types of Interest
    lr_stats = lr_dea_res[lr_dea_res["source"].isin([key_source]) & (
        lr_dea_res["target"].isin([key_target]))]  # subset source-target
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
    # fig = corneto.methods.carnival.visualize_network(df_res)
    # fig.view()
    fig = None
    return df_res, problem, fig

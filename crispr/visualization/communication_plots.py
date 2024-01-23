import liana
import numpy as np

def plot_receptor_ligand(adata=None, liana_res=None, title=None, top_n=20,
                         key_added="liana_res", p_threshold=0.01, 
                         key_sources=None, key_targets=None, figsize=None,
                         cmap="magma", **kwargs):
    """Plot Liana receptor-ligand analyses."""
    if figsize is None:  # auto-calculate figure size if not provided
        # figsize = (len(key_sources) * len(key_targets) / 4, top_n / 4)
        figsize = (30, 30)
    size_range = kwargs.pop("size_range", (1, 6))  # size range for dots
    if liana_res is None and adata is not None:
        liana_res = adata.uns[key_added].copy()
    l_r = [list(liana_res[x].unique()) for x in ["source", "target"]]
    kss, ktt = [list(set(x if x else l_r[i]).intersection(set(l_r[i]))) 
                for i, x in enumerate([key_sources, key_targets])]
    kws = dict(return_fig=True, cmap=cmap, source_labels=kss, 
               target_labels=ktt, top_n=top_n, figure_size=figsize)  # kws
    kws.update(kwargs)  # update with any non-overlapping user kws
    # lr_res = liana.multi.df_to_lr(
    #     adata, dea_df=dea_df,
    #                        resource_name="consensus",
    #                        expr_prop=0.1, # calculated for adata as passed - used to filter interactions
    #                        groupby=groupby,
    #                        stat_keys=["stat", "pvalue", "padj"],
    #                        use_raw=False,
    #                        complex_col="stat", # NOTE: we use the Wald Stat to deal with complexes
    #                        verbose=True,
    #                        return_all_lrs=False,
    #                        )
    fig = {}
    fig["dot"] = liana.pl.dotplot(
        liana_res=liana_res, colour="interaction_stat", 
        orderby_ascending=False, size_range=size_range, 
        order_by_absolute=True, inverse_size=True, orderby="lr_means", 
        size="cellphone_pvals", **kws)  # dot plot
    fig["tile_means"] = liana.pl.tileplot(
        liana_res=liana_res, fill="lr_means", orderby="cellphone_pvals",
        label="props", label_fun=lambda x: f"{x:.2f}", orderby_ascending=True,
        **kws)  # tile plot
    fig["tile_expr"] = liana.pl.tileplot(
        liana_res=liana_res, fill="expr", top_n=top_n, label="props",
        # label="cell_type", 
        label_fun=lambda x: "*" if x < p_threshold else np.nan,
        filter_fun=lambda x: x["cellphone_pvals"] <= p_threshold, 
        orderby="lr_means", orderby_ascending=False, 
        orderby_absolute=False, 
        # source_title="Ligand", 
        target_title="Receptor")  # tile plot for expression
    if title:
        for q in fig:
            fig[q].labels.title = title
    for c in fig:
        print(fig[c])
    return fig
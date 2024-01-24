import liana
import numpy as np

def plot_receptor_ligand(adata=None, liana_res=None, title=None, top_n=20,
                         key_added="liana_res", p_threshold=0.01, 
                         key_sources=None, key_targets=None, figsize=None,
                         lr_dea_res=None,  # Liana LR-DEA results dataframe
                         dot_size="cellphone_pvals",
                         cmap="magma", **kwargs):
    """Plot Liana receptor-ligand analyses (and, optionally, DEA)."""
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
    fig = {}
    if adata is not None:
        fig["dea_dot"] = liana.pl.dotplot(
            adata=adata, colour="lr_means",  # color by interaction strength
            orderby_ascending=False, size_range=size_range, inverse_size=True, 
            orderby="lr_means", size=dot_size, **kws)  # dot plot
    if lr_dea_res is not None:
        # fig["dea_dot"] = liana.pl.dotplot(
        #     liana_res=lr_dea_res, colour="interaction_stat", top_n=top_n,
        #     size="interaction_props", inverse_size=True, 
        #     orderby_absolute=True, orderby="interaction_stat", 
        #     orderby_ascending=False, size_range=size_range)  # dot
        fig["dea_tile"] = liana.pl.tileplot(
            liana_res=lr_dea_res, fill="expr", label="padj", 
            label_fun = lambda x: "*" if x < 0.05 else np.nan, 
            orderby="interaction_stat", orderby_ascending=False,
            orderby_absolute=False, **kws,
            source_title="Ligand", target_title="Receptor")  # tile plot
    if title:
        for q in fig:
            fig[q].labels.title = title
    for c in fig:
        print(fig[c])
    return fig
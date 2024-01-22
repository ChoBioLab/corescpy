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
    size_range = kwargs.pop("size_range", (1, 6))
    if liana_res is None and adata is not None:
        liana_res = adata.uns[key_added].copy()
    l_r = [list(liana_res[x].unique()) for x in ["source", "target"]]
    kss, ktt = [list(set(x if x else l_r[i]).intersection(set(l_r[i]))) 
                for i, x in enumerate([key_sources, key_targets])]
    kws = dict(
        filterby="cellphone_pvals", return_fig=True, cmap=cmap,
        source_labels=kss, target_labels=ktt,
        filter_lambda=lambda x: x <= p_threshold, 
        top_n=top_n, figure_size=figsize)  # plot kws
    kws.update(kwargs)  # update with any non-overlapping user kws
    fig = {}
    fig["dot"] = liana.pl.dotplot(
        liana_res=liana_res, colour="lr_means", 
        orderby_ascending=False, size_range=size_range, 
        inverse_size=True, orderby="lr_means", size="cellphone_pvals", **kws)
    fig["tile_means"] = liana.pl.tileplot(
        liana_res=liana_res, fill="means", orderby="cellphone_pvals", **kws,
        label="props", label_fun=lambda x: f"{x:.2f}", orderby_ascending=True)
    fig["tile_expr"] = liana.pl.tileplot(
        liana_res=liana_res, fill="expr", label="padj", top_n=top_n,
        label_fun=lambda x: "*" if x < p_threshold else np.nan,
        orderby="interaction_stat", orderby_ascending=False,
        orderby_absolute=False, source_title="Ligand", 
        target_title="Receptor")
    if title:
        for q in fig:
            fig[q].labels.title = title
    for c in fig:
        print(fig[c])
    return fig
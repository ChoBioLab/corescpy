import squidpy as sq
import liana
from liana.method import cellphonedb
import pandas as pd

def analyze_receptor_ligand(
    adata, method="liana", col_condition=None, layer="log1p",
    key_sources=None, key_targets=None, col_cell_type=None, n_perms=10, 
    p_threshold=0.05, figsize=None, remove_ns=True, top_n=20, 
    cmap="magma", kws_plot=None, resource="CellPhoneDB", copy=True):
    """Perform receptor-ligand analysis."""
    if copy is True:
        adata = adata.copy()
    if kws_plot is None:
        kws_plot = {}
    if layer is not None:
        adata.X = adata.layers[layer].copy()
        
    # Source/Target Cell Types to Plot
    key_sources, key_targets = [list(x) if x else adata.obs[
        col_cell_type].unique() for x in [key_sources, key_targets]]
    ktt, kss = [list(set(x).intersection(set(list(adata.obs[ 
        col_cell_type].unique())))) for x in [key_targets, key_sources]]
        
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
        kws = dict(
            filterby="cellphone_pvals", return_fig=True,
            filter_fun=lambda x: x["cellphone_pvals"] <= p_threshold, 
            top_n=top_n, size_range=(1, 6), top_n=top_n)  # plot kws default
        kws.update(kws_plot)  # update with any non-overlapping user kws
        if figsize is None:  # auto-calculate figure size if not provided
            figsize = (len(kss) * len(ktt) / 4, kws["top_n"] / 4)
        fig = liana.pl.dotplot(
            adata=adata, colour="lr_means", orderby="lr_means", cmap=cmap,
            source_labels=kss, target_labels=ktt, size="cellphone_pvals",  
            # orderby_ascending=False, 
            inverse_size=True, **kws)  # dot plot; top means
        res = adata.uns[kwargs["key_added"]]
        if col_condition is not None:
            fig.labels.title = "Overall"
            print(fig)
            fig, res = {"Overall": fig}, {"overall": res}
            for c in adata.obs[col_condition].unique():  # iterate conditions
                fig[c], anc = {}, adata[adata.obs[col_condition] == c]  # data
                k_t, k_s = [list(set(x).intersection(set(list(anc.obs[
                    col_cell_type].unique())))) for x in [ktt, kss]]  # s/t
                cellphonedb(anc, groupby=col_cell_type, **kwargs)  # run cpdb
                fig[c]["dot"] = liana.pl.dotplot(
                    adata=anc, colour="lr_means", size="cellphone_pvals",
                    inverse_size=True, source_labels=k_s, target_labels=k_t,
                    # orderby_ascending=False, 
                    orderby="lr_means", **kws)  # dotplot; top means
                fig[c]["tile"] = liana.pl.tileplot(
                    adata=anc, fill="means", orderby="cellphone_pvals",
                    label_fun=lambda x: f"{x:.2f}", label="props", cmap=cmap, 
                    # orderby_ascending=True, 
                    source_labels=k_s, target_labels=k_t, **kws)  # tile (~p)
                for q in fig[c]:
                    fig[c][q].labels.title = f"{col_condition} = {c}"
                    print(fig[c][q])
                res[c] = anc.uns[kwargs["key_added"]]
            res["combined"] = pd.concat(
                [res[c] for c in res], keys=res.keys(), 
                names=[col_condition, ""])  # results for all conditions
    return res, fig
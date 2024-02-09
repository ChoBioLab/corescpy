import liana
from squidpy.pl._color_utils import _get_palette
from squidpy.pl._graph import _get_data
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_non_empty_sequence,
    _get_valid_values,
)
import matplotlib.pyplot as plt
import seaborn as sns
import corescpy as cr
import pandas as pd
import numpy as np


def plot_receptor_ligand(adata=None, liana_res=None, title=None, top_n=20,
                         key_added="liana_res", p_threshold=0.01,
                         key_sources=None, key_targets=None, figsize=None,
                         lr_dea_res=None,  # Liana LR-DEA results dataframe
                         dot_size="cellphone_pvals",
                         group="both", cmap="magma", **kwargs):
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
    kws = dict(return_fig=True, source_labels=kss, figure_size=figsize,
               cmap=cmap, target_labels=ktt, top_n=top_n)  # arguments
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
            label_fun=lambda x: "*" if x < 0.05 else np.nan,
            orderby="interaction_stat", orderby_ascending=False,
            orderby_absolute=False, **kws, dendrogram=group,
            source_title="Ligand", target_title="Receptor")  # tile plot
    if title:
        for q in fig:
            fig[q].labels.title = title
    for c in fig:
        print(fig[c])
    return fig


def plot_cooccurrence(adata, col_cell_type, cluster_key=None, palette=None,
                      figsize=None, fontdict=None, legend_kwargs=None,
                      key_cell_type=None, dpi=100, wspace=2,
                      right_margin=0.15, **kwargs):
    """
    Plot co-occurrence of cell types in spatial data.

    Adapted from source code of Squidpy functions.
    """
    # Arguments
    fontdict = cr.tl.merge({"fontsize": 14}, fontdict)
    legend_kwargs = cr.tl.merge({"loc": "center left",
                                 "bbox_to_anchor": (1, 0.5)}, legend_kwargs)
    if "loc" not in legend_kwargs:
        legend_kwargs["loc"] = "center left"
        legend_kwargs.setdefault("bbox_to_anchor", (1, 0.5))

    # Data    out = occurrence_data["occ"]
    occurrence_data = _get_data(adata, cluster_key=col_cell_type,
                                func_name="co_occurrence")
    out = occurrence_data["occ"]
    categories = adata.obs[col_cell_type].cat.categories
    clusters = categories if key_cell_type is None else key_cell_type
    clusters = _assert_non_empty_sequence(clusters, name="clusters")
    clusters = sorted(_get_valid_values(clusters, categories))
    _assert_categorical_obs(adata, key=col_cell_type)
    interval = occurrence_data["interval"][1:]
    categories = adata.obs[col_cell_type].cat.categories
    palette = _get_palette(adata, cluster_key=col_cell_type,
                           categories=categories, palette=palette)

    # Plot
    nrow, ncol = cr.pl.square_grid(len(clusters))
    fig, axs = plt.subplots(
        nrow, ncol, dpi=dpi, constrained_layout=True,
        figsize=(5 * len(clusters), 5) if figsize is None else figsize)
    axs = np.ravel(axs)  # make into iterable
    for g, ax in zip(clusters, axs):
        idx = np.where(categories == g)[0][0]
        dff = pd.DataFrame(out[idx, :, :].T, columns=categories).melt(
            var_name=col_cell_type, value_name="probability")
        dff["distance"] = np.tile(interval, len(categories))
        sns.lineplot(x="distance", y="probability", data=dff,
                     dashes=False, hue=col_cell_type,
                     hue_order=categories, palette=palette, ax=ax, **kwargs)
        ax.legend().set_visible(False)
        ax.set_title(rf"$\frac{{p(exp|{g})}}{{p(exp)}}$", fontdict=fontdict)
        ax.set_ylabel("value")
    fig.legend(**legend_kwargs)
    plt.subplots_adjust([11, 12], right=1 - right_margin)
    plt.show()

    return fig, axs

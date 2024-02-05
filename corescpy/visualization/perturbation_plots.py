# from scanpy.plotting import _utils
import seaborn as sns
import corescpy as cr
import scanpy as sc
import pertpy as pt
import decoupler
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings
import functools
import pandas as pd
import numpy as np


def plot_targeting_efficiency(adata, key_control="NT", key_nonperturbed="NP",
                              key_treatment="KO", col_guide_rna="guide_ids",
                              guide_split="-", feature_split=None,
                              mixscape_class_global="mixscape_class_global",
                              **kwargs):
    """
    Plot targeting efficiency (Mixscape scores) of guide RNAs.

    Adapted from `pertpy.pl.ms.barplot()`.

    """
    if "palette" not in kwargs:
        kwargs.update({"palette": {
            key_treatment: "blue", key_nonperturbed: "r",
            key_control: "grey"}})
    if mixscape_class_global not in adata.obs:
        raise ValueError("Please run `pt.tl.mixscape` first.")
    guides = adata.obs[[mixscape_class_global, col_guide_rna]]
    if feature_split is not None:
        guides = guides[[mixscape_class_global]].join(
            guides[col_guide_rna].str.split(feature_split).explode(
                ).to_frame(col_guide_rna))  # multi-guide cells -> multi-rows
    count = pd.crosstab(index=guides[mixscape_class_global],
                        columns=guides[col_guide_rna])
    all_cells_pct = pd.melt(count / count.sum(),
                            ignore_index=False).reset_index()
    ko_cells_percentage = all_cells_pct[all_cells_pct[
        mixscape_class_global] == key_treatment].sort_values(
            "value", ascending=False)  # % KO/KD/Tx
    new_levels = ko_cells_percentage[col_guide_rna]
    all_cells_pct[col_guide_rna] = pd.Categorical(
        all_cells_pct[col_guide_rna], categories=new_levels, ordered=False)
    all_cells_pct[mixscape_class_global] = pd.Categorical(
        all_cells_pct[mixscape_class_global], categories=[
            key_control, key_nonperturbed, key_treatment], ordered=False)
    all_cells_pct["Gene"] = all_cells_pct[col_guide_rna].str.rsplit(
        guide_split, expand=True)[0]
    all_cells_pct["guide_number"] = all_cells_pct[
        col_guide_rna].str.rsplit(guide_split, expand=True)[1]
    all_cells_pct["guide_number"] = guide_split + all_cells_pct[
        "guide_number"]
    np_ko_cells = all_cells_pct[all_cells_pct["Gene"] != key_control]
    _, cols = cr.pl.square_grid(len(np_ko_cells["Gene"].unique()))
    p_1 = sns.catplot(data=np_ko_cells, x="mixscape_class_global", y="value",
                      col="Gene", col_wrap=cols, kind="bar",
                      hue="mixscape_class_global", **kwargs)
    return p_1


def plot_mixscape(adata, col_target_genes, key_treatment, key_control="NT",
                  assay=None, adata_protein=None, protein_of_interest=None,
                  key_nonperturbed="NP", color="red",
                  layer="X_pert", target_gene_idents=True,
                  subsample_number=5, figsize=None, ncol=3,
                  col_guide_rna=None, guide_split="-", feature_split="|"):
    """Make Mixscape perturbation plots."""
    figs = {}
    target_gene_idents = [target_gene_idents] if isinstance(
        target_gene_idents, str) else list(
            adata.uns["mixscape"].keys()) if (
                target_gene_idents is True) else list(target_gene_idents)
    figs["DEX_ordered_by_ppp_heat"] = {}
    figs["ppp_violin"] = {}
    figs["perturbation_score"] = {}
    figs["perturbation_clusters"] = {}
    if adata_protein:
        if "mixscape_class_global" not in adata_protein:
            adata_protein.loc[:, "mixscape_class_global"] = adata.loc[
                adata_protein.index]["mixscape_class_global"]
    adata = adata[assay].copy() if assay else adata.copy()
    for g in target_gene_idents:  # iterate target genes of interest
        try:
            figs["DEX_ordered_by_ppp_heat"][
                g] = pt.pl.ms.heatmap(
                    adata=adata, subsample_number=subsample_number,
                    labels=col_target_genes, target_gene=g,
                    layer=layer, control=key_control, show=False
                    )  # differential expression heatmap, sort by PPs
            plt.show()
        except Exception as err:
            figs["DEX_ordered_by_ppp_heat"][g] = err
            warnings.warn(f"{err}\n\nCould not plot DEX heatmap!")
        tg_conds = [
            key_control, f"{g} {key_nonperturbed}",
            f"{g} {key_treatment}"]  # conditions: gene g
        figs["ppp_violin"][g] = pt.pl.ms.violin(
            adata=adata, target_gene_idents=tg_conds,
            rotation=45, groupby="mixscape_class", show=False,
            keys=f"mixscape_class_p_{key_treatment}".lower()
            )  # gene: perturbed, NP, control
    figs["ppp_violin"]["global"] = pt.pl.ms.violin(
        adata=adata, target_gene_idents=[
            key_control, key_nonperturbed, key_treatment],
        rotation=45, keys=f"mixscape_class_p_{key_treatment}".lower(),
        groupby="mixscape_class_global")  # same, but global
    try:
        tg_conds = [key_control] + functools.reduce(
            lambda i, j: i + j, [[
                f"{g} {key_nonperturbed}", f"{g} {key_treatment}"
                ] for g in target_gene_idents])  # conditions: all genes
        figs["ppp_violin"]["all"] = pt.pl.ms.violin(
            adata=adata,
            # keys=f"mixscape_class_p_{key_treatment}".lower(),
            target_gene_idents=tg_conds, rotation=45,
            groupby="mixscape_class")  # gene: perturbed, NP, control
    except Exception as err:
        figs["ppp_violin"]["all"] = err
        print(err)
    for g in target_gene_idents:  # iterate target genes of interest
        try:
            figs["perturbation_score"][g] = pt.pl.ms.perturbscore(
                adata=adata, labels=col_target_genes,
                target_gene=g, color=color, perturbation_type=key_treatment)
            print(figs["perturbation_score"][g])
        except Exception as err:
            figs["perturbation_score"][g] = err
            warnings.warn(f"{err}\n\nCould not plot scores ({g})!")
    if adata_protein is not None:
        try:
            figs["protein_expression"] = pt.pl.ms.violin(
                adata=adata_protein, target_gene_idents=target_gene_idents,
                keys=protein_of_interest, hue="mixscape_class_global",
                groupby=col_target_genes)
        except Exception as err:
            figs["perturbation_score"][g] = err
            warnings.warn("{err}\n\nCould not plot protein expression for "
                          f"{protein_of_interest}!")
    if col_guide_rna is not None:
        try:
            figs["targeting_efficiency"] = cr.pl.plot_targeting_efficiency(
                adata, col_guide_rna=col_guide_rna, key_control=key_control,
                key_treatment=key_treatment, key_nonperturbed=key_nonperturbed,
                guide_split=guide_split, feature_split=feature_split,
                mixscape_class_global="mixscape_class_global")
        except Exception as err:
            figs["targeting_efficiency"] = err
            warnings.warn(f"{err}\n\nCould not plot targeting efficiency!")
    try:  # LDA clusters
        figs["perturbation_clusters"] = plt.figure(figsize=(15, 15))
        axis = figs["perturbation_clusters"].add_subplot(111)
        pt.pl.ms.lda(adata=adata, perturbation_type=key_treatment, ax=axis,
                     control=key_control)  # UMAP
        figs["perturbation_clusters"].tight_layout()
        figs["perturbation_clusters"].suptitle(
            "Perturbation-Specific Clusters")
        print(figs["perturbation_clusters"])
    except Exception as err:
        figs["perturbation_clusters"] = err
        warnings.warn(f"{err}\n\nPerturbation cluster plot failed!")
    if ncol is None:
        nrow, ncol = cr.pl.square_grid(len(target_gene_idents))
    else:
        nrow = int(np.ceil(len(target_gene_idents) / ncol))
    if figsize is None:
        figsize = (3 * ncol, 3 * nrow)
    figs["gex_violin"], axs = plt.subplots(nrow, ncol, figsize=figsize)
    if "flatten" in dir(axs):  # if enough target genes for multi-row plot
        axs = axs.flatten()  # flatten axis so can access by flat index
    # for i, x in enumerate(target_gene_idents):  # iterate target genes
    #     try:
    #         sc.pl.violin(
    #             adata[adata.obs[col_target_genes].isin([x, key_control], x,
    #             groupby="mixscape_class_global", ax=axs[i])
    #     except Exception as err:
    #         print(f"{err}\n\nGene expression violin plot failed for {x}!")
    #         figs[f"gex_violin_{x}"] = err
    return figs


def plot_gsea_results(adata, gsea_results, p_threshold=0.0001, layer=None,
                      ifn_pathways=True, figsize=(60, 20), **kwargs):
    """Plot results from cr.ax.perform_gsea()."""
    figs = {}
    col_condition = gsea_results["col"].iloc[0]
    score_sort = gsea_results[gsea_results.pval < p_threshold]  # filter by p
    score_sort = score_sort.loc[score_sort.score.abs().sort_values(
        ascending=False).index]  # sort by absolute score
    figs["bar_score"] = sns.catplot(data=score_sort.head(20),
                                    x="score", y="source", kind="bar")
    figs["bar_p"] = sns.catplot(data=gsea_results.head(20),
                                x="-log10(pval)", y="source", kind="bar")
    # fig, axes = plt.subplots(1, 3, figsize=figsize,
    #                          tight_layout=True, sharey=True)
    # for i, q in enumerate(gsea_results.iloc[:, :3].columns):
    #     axes[i].set_title(q)
    #     sns.heatmap(gsea_results, x=q, y="source",
    #                 vmin=-1, vmax=1, ax=axes[i],
    #                 cmap=["coolwarm", "coolwarm", "viridis_r"][i])  # heatmaps

    # Cell-Level
    if ifn_pathways not in [None, False]:
        if ifn_pathways is True:  # choose (~ p & score) pathways to plot
            ifn_pathways = list(score_sort.reset_index().head(7).source)
        adata.obs[ifn_pathways] = adata.obsm["aucell_estimate"][ifn_pathways]
        if "ncols" in kwargs:
            ccc = kwargs["n_cols"]
        else:
            ccc = cr.pl.square_grid(int(1 if isinstance(
                col_condition, str) else 2) + len(ifn_pathways))[1]
        cond = list([col_condition] if isinstance(
            col_condition, str) else col_condition) + list(ifn_pathways)
        figs["umap"] = sc.pl.umap(
            adata, color=cond, ncols=ccc, layer=layer, **kwargs)  # score UMAP
    return figs


def plot_pathway_interference_results(adata, pathway, col_cell_type=None,
                                      obsm_key="mlm_estimate",
                                      standard_scale="var",
                                      cmap="coolwarm", vcenter=0):
    """Plot results from cr.ax.perform_pathway_interference()."""
    figs = {}
    if col_cell_type not in adata.obs:
        warnings.warn(f"{col_cell_type} not in `.obs`. Skipping violin plot.")
    acts = decoupler.get_acts(adata, obsm_key=obsm_key)
    if col_cell_type is not None and col_cell_type in acts.obs:
        figs["umap"] = sc.pl.umap(acts, color=[pathway, col_cell_type],
                                  cmap=cmap, vcenter=vcenter)
        figs["violin"] = sc.pl.violin(acts, keys=[pathway],
                                      groupby=col_cell_type, rotation=90)
        figs["matrix"] = sc.pl.matrixplot(
            acts, var_names=acts.var_names, groupby=col_cell_type,
            dendrogram=True, standard_scale=standard_scale,
            colorbar_title="X-Scaled Scores", cmap=cmap)
    else:
        figs["umap"] = sc.pl.umap(acts, color=[pathway], cmap=cmap, vcenter=0)
    return figs


def plot_distance(res_pairwise_genes=None, res_pairwise_clusters=None,
                  res_linkage=None, col_cell_type=None, res_contrasts=None,
                  distance_type="edistance", p_adjust=True, **kwargs):
    """Plot distance metrics (partially adapted from Pertpy tutorial)."""
    figs = {}
    palette = kwargs.pop("palette", {True: "green", False: "red"})
    kwargs = {"cmap": "Reds_r", "figsize": (20, 20), **kwargs}

    highlight_real_range = kwargs.pop("highlight_real_range", True)

    # Heatmap of Distances
    if highlight_real_range is True:
        vmin = np.min(np.ravel(res_pairwise_genes.values)[np.ravel(
            res_pairwise_genes.values) != 0])
        if "vmin" in kwargs:
            warnings.warn("vmin already set in kwargs plot. Setting to "
                          f" {vmin} as highlight_real_range is True.")
        kwargs.update(dict(vmin=vmin))
    figs[f"distance_heat_{distance_type}_conditions"] = sns.clustermap(
        res_pairwise_genes, **kwargs)  # cluster heatmap
    plt.show()

    # Dendrogram of Linkages/Hierarchies
    if res_pairwise_clusters is not None and res_linkage is not None:
        plt.figure(figsize=kwargs["figsize"])
        _ = dendrogram(res_linkage, labels=res_pairwise_clusters.index,
                       orientation="left", color_threshold=0)  # dendrogram
        plt.xlabel(f"Distance ({distance_type})")
        plt.ylabel(col_cell_type)
        plt.gca().yaxis.set_label_position("right")
        figs[f"distance_cluster_hierarchies_{distance_type}"] = plt.gcf()
        figs[f"distance_cluster_hierarchies_{distance_type}"].tight_layout()
        plt.show()

    # Contrast Distance & Signifance
    suff = "_adj" if p_adjust is True else ""
    if res_contrasts is not None:
        for x in res_contrasts:
            tab, ref = res_contrasts[x], x.split(" = ")[1]
            with sns.axes_style("darkgrid"):
                sns.scatterplot(
                    data=tab[tab.index != ref], x=f"pvalue{suff}" + str(),
                    y="distance", hue=f"significant{suff}", palette=palette)
            plt.title(f"{distance_type} Distance: Contrast Results ({x})")
            plt.xlabel(f"P" + str(" Adjusted" if p_adjust is True else ""))
            plt.ylabel(f"Distance to Contrast Group")
            figs[f"contrast_{distance_type}_{x}"] = plt.gcf()
            plt.show()
    return figs
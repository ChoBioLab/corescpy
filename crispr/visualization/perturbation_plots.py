# from scanpy.plotting import _utils
import seaborn as sns 
import crispr as cr
import scanpy as sc
import pertpy as pt
import matplotlib.pyplot as plt
import warnings
import functools
import pandas as pd 
import numpy as np


def plot_targeting_efficiency(
    adata, key_control="NT", key_nonperturbed="NP", key_treatment="KO",
    col_guide_rna="guide_ids", guide_split="-", feature_split=None, 
    mixscape_class_global="mixscape_class_global", **kwargs):
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
    figs["ppp_violin"][f"global"] = pt.pl.ms.violin(
        adata=adata, target_gene_idents=[
            key_control, key_nonperturbed, key_treatment], 
        rotation=45, keys=f"mixscape_class_p_{key_treatment}".lower(),
        groupby="mixscape_class_global")  # same, but global
    try:
        tg_conds = [key_control] + functools.reduce(
            lambda i, j: i + j, [[f"{g} {key_nonperturbed}", 
                    f"{g} {key_treatment}"] 
            for g in target_gene_idents])  # conditions: all genes
        figs["ppp_violin"]["all"] = pt.pl.ms.violin(
            adata=adata, keys=f"mixscape_class_p_{key_treatment}".lower(),
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
            figs["targeting_efficiency"][g] = err
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
        figsize = (5 * ncol, 5 * nrow)
    figs[f"gex_violin"], axs = plt.subplots(nrow, ncol, figsize=figsize)
    axs = axs.flatten()
    # for i, x in enumerate(target_gene_idents):  # iterate target genes
    #     try:
    #         sc.pl.violin(
    #             adata[adata.obs[col_target_genes] == x], keys=x,
    #             groupby="mixscape_class_global", ax=axs[i])
    #     except Exception as err:
    #         print(f"{err}\n\nGene expression violin plot failed for {x}!")
    #         figs[f"gex_violin_{x}"] = err
    return figs
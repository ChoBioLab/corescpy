from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_classic,
    xlab,
    ylab,
)
from scanpy.plotting import _utils
import pandas as pd 


def plot_perturbation_scores_by_guide(
    adata, mixscape_class_global="mixscape_class_global", 
    col_guide_rna="guide_ids", guide_split="-",
    key_control="NT", key_treatment="Perturbed",
    panel_spacing=None, kws_text_size=None):
    """
    Plot perturbation scores by guide RNA in facetted barplots.
    Modified from pertpy Mixscape barplot method.
    """
    if mixscape_class_global not in adata.obs:
        raise ValueError("Please run `pt.tl.mixscape` first.")
    if panel_spacing is None:
        panel_spacing = [0.1, 0.1]
    if isinstance(panel_spacing, (int, float)):
        panel_spacing = [panel_spacing, panel_spacing]
    if kws_text_size is None:
        kws_text_size = {}
    for i in ["axis_text_x_size", "axis_text_y_size", 
              "axis_text_title_size", "axis_text_x_size", 
              "legend_title_size", "legend_text_size"]:
        if i not in kws_text_size:
            kws_text_size[i] = 8 if "title" in i or "legend" in i else 0.3
    
    count = pd.crosstab(index=adata.obs[mixscape_class_global], 
                        columns=adata.obs[col_guide_rna])
    all_cells_percentage = pd.melt(count / count.sum(), 
                                   ignore_index=False).reset_index()
    KO_cells_percentage = all_cells_percentage[all_cells_percentage[
        mixscape_class_global] == key_treatment]
    KO_cells_percentage = KO_cells_percentage.sort_values(
        "value", ascending=False)

    new_levels = KO_cells_percentage[col_guide_rna]
    all_cells_percentage[col_guide_rna] = pd.Categorical(
        all_cells_percentage[col_guide_rna], 
        categories=new_levels, ordered=False)
    all_cells_percentage[mixscape_class_global] = pd.Categorical(
        all_cells_percentage[mixscape_class_global], 
        categories=[key_control, key_treatment, "NP"], ordered=False)
    all_cells_percentage["gene"] = all_cells_percentage[
        col_guide_rna].str.rsplit(guide_split, expand=True)[0]
    all_cells_percentage["guide_number"] = all_cells_percentage[
        col_guide_rna].str.rsplit(guide_split, expand=True)[1]
    all_cells_percentage["guide_number"] = "g" + all_cells_percentage[
        "guide_number"]
    NP_KO_cells = all_cells_percentage[
        all_cells_percentage["gene"] != key_control]

    p_1 = (
        ggplot(NP_KO_cells, aes(x="guide_number", y="value",
                                fill=mixscape_class_global))
        + scale_fill_manual(values=["#7d7d7d", "#c9c9c9", "#ff7256"])
        + geom_bar(stat="identity")
        + theme_classic()
        + xlab("sgRNA")
        + ylab("% of cells")
    )
    p_1 = (
        p_1
        + theme(
            axis_text_x=element_text(
                size=kws_text_size["axis_text_x_size"], hjust=2),
            axis_text_y=element_text(
                size=kws_text_size["axis_text_y_size"]),
            axis_title=element_text(
                size=kws_text_size["axis_text_title_size"]),
            strip_text=element_text(
                size=kws_text_size["axis_text_x_size"], face="bold"),
            panel_spacing_x=panel_spacing[0],
            panel_spacing_y=panel_spacing[1],
        )
        + facet_wrap("gene", ncol=5, scales="free")
        + labs(fill="mixscape class")
        + theme(legend_title=element_text(
            size=kws_text_size["legend_title_size"]), 
                legend_text=element_text(
                    size=kws_text_size["legend_text_size"]))
    )
    _utils.savefig_or_show("mixscape_barplot", show=True, save=False)
    return p_1
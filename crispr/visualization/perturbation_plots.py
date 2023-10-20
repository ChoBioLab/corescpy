# from scanpy.plotting import _utils
import seaborn as sns 
import crispr as cr
import pandas as pd 


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
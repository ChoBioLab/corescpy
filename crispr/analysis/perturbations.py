import pertpy as pt

def calculate_targeting_efficiency(adata, assay=None, guide_rna_column="NT"):
    """_summary_

    Args:
        adata (_type_): _description_
        assay (_type_, optional): _description_. Defaults to None.
        guide_rna_column (str, optional): _description_. Defaults to "NT".

    Returns:
        _type_: _description_
    """
    figs = {}  # for figures
    figs.update({"barplot": pt.pl.ms.barplot(adata[assay] if assay else adata, 
                            guide_rna_column=guide_rna_column)})
    return figs


def calculate_perturbations(adata, target_gene, target_gene_idents, 
                            assay=None, control="NT", 
                            color="green"):
    """Calculate perturbation scores (from Pertpy Mixscape tutorial)."""
    
    # Identify Cells without Detectible Pertubations
    mix = pt.tl.Mixscape()  # mixscape object
    mix.perturbation_signature(adata[assay] if assay else adata, 
                               "perturbation", "NT", "replicate")  # signatures
    mix.mixscape(adata=adata[assay] if assay else adata, control=control, 
                 labels="gene_target", layer="X_pert")  # mixscape routine
    mix.lda(adata=adata[assay] if assay else adata, labels="gene_target", 
            layer="X_pert")  # linear discriminant analysis (LDA)
    if plot is True: 
    
    
    # Cell Perturbation Scores
    fig_ps = pt.pl.ms.perturbscore(adata=adata[assay] if assay else adata, 
                                   labels='gene_target', 
                                   target_gene=target_gene,
                                   color=color)  # plot perturbation scores
    if plot is True: 
        figs.update({"perturbation_scores": fig_ps})
    if plot is True:
        fig_ppp = pt.pl.ms.violin(adata=adata[assay] if assay else adata, 
                                  target_gene_idents=target_gene_idents,
                                  groupby="mixscape_class")  # plot PPPs
        figs.update({"PPP": fig_ppp})
        fig_dehm = pt.pl.ms.heatmap(adata=adata[assay] if assay else adata, 
                                    labels="gene_target", 
                                    target_gene=target_gene
                                    layer="X_pert", control=control)  # plot DE
        figs.update({"DE_heatmap": fig_dehm})
        fig_lda = pt.pl.ms.lda(
            adata=adata[assay] if assay else adata)  # plot LDA
        figs.update({"lda": fig_lda})
    if plot is True:
        return figs
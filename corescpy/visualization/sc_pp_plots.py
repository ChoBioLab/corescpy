from typing import Optional, Union
import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import rcParams
from anndata import AnnData
from adjustText import adjust_text
from scanpy.pl import _utils

# ----------------------------------------------------------------------------
# Plot result of preprocessing functions
# Adapted from Scanpy
# ----------------------------------------------------------------------------

# Package used for adding well aligned labels on the plot
# def plot_hvgs(adata, genes, n_top=10, title=""):
#     adata = adata.copy()
#     with pl.rc_context({"figure.figsize":(5, 5)}):
#         lab_x, lab_y = "means", "dispersions"
#         adata.var["is_highly_variable"] = adata.var[
#             "highly_variable"].astype(bool).astype(str)
#         axs = sc.pl.scatter(adata, x="means", y="dispersions",
#                             color="is_highly_variable", show=False)

#         # Move plot title from Axes to Legend
#         axs.set_title(title)
#         axs.get_legend().set_title("Highly Variable")

#         # Select genes to be labeled
#         texts = []
#         genes = n_top
#         for gene in genes:
#             # Position of object to be marked
#             x_loc = adata.var.at[gene, lab_x]
#             y_loc = adata.var.at[gene, lab_y]
#             # Text color
#             color_point = "k"
#             texts.append(axs.text(x_loc, y_loc, gene, color=color_point,
#                                   fontsize=10))

#         # Label selected genes on the plot
#         _= adjust_text(texts, expand_points=(2, 2), arrowprops=dict(
#             arrowstyle="->",  color="gray",  lw=1), ax=axs)


def plot_hvgs(
    adata_or_result: Union[AnnData, pd.DataFrame, np.recarray],
    log: bool = False, figsize=None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    highly_variable_genes: bool = True,
    palette: Optional[Union[list, None]] = None
):
    """
    Plot dispersions or normalized variance versus means for genes.

    Produces Supp. Fig. 5c of Zheng et al. (2017) and MeanVarPlot() and
    VariableFeaturePlot() of Seurat.

    Parameters
    ----------
    adata
        Result of :func:`~scanpy.pp.highly_variable_genes`.
    log
        Plot on logarithmic axes.
    show
         Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ends in {{`'.pdf'`, `'.png'`, `'.svg'`}}.
    """
    if palette is None:
        palette = ["red", "grey"]
    if isinstance(adata_or_result, AnnData):
        result = adata_or_result.var
        seurat_v3_flavor = adata_or_result.uns["hvg"]["flavor"] == "seurat_v3"
    else:
        result = adata_or_result
        if isinstance(result, pd.DataFrame):
            seurat_v3_flavor = "variances_norm" in result.columns
        else:
            seurat_v3_flavor = False
    if highly_variable_genes:
        gene_subset = result.highly_variable
    else:
        gene_subset = result.gene_subset
    means = result.means

    if seurat_v3_flavor:
        var_or_disp = result.variances
        var_or_disp_norm = result.variances_norm
    else:
        var_or_disp = result.dispersions
        var_or_disp_norm = result.dispersions_norm
    if not figsize:
        figsize = rcParams["figure.figsize"]
    pl.figure(figsize=(2 * figsize[0], figsize[1]))
    pl.subplots_adjust(wspace=0.3)
    for idx, d in enumerate([var_or_disp_norm, var_or_disp]):
        pl.subplot(1, 2, idx + 1)
        for lab, color, mask in zip(["highly variable genes", "other genes"
                                     ], palette, [gene_subset, ~gene_subset]):
            means_, var_or_disps_ = means[mask], d[mask]
            pl.scatter(means_, var_or_disps_, label=lab, c=color,
                       s=figsize[1] / 5)
        if log:  # there's a bug in autoscale
            pl.xscale("log")
            pl.yscale("log")
            y_min = np.min(var_or_disp)
            y_min = 0.95 * y_min if y_min > 0 else 1e-1
            pl.xlim(0.95 * np.min(means), 1.05 * np.max(means))
            pl.ylim(y_min, 1.05 * np.max(var_or_disp))
        if idx == 0:
            pl.legend()
        pl.xlabel(
            ("$log_{10}$ " if False else "") + "mean expressions of genes")
        data_type = "dispersions" if not seurat_v3_flavor else "variances"
        pl.ylabel(
            ("$log_{10}$ " if False else "")
            + "{} of genes".format(data_type)
            + (" (normalized)" if idx == 0 else " (not normalized)")
        )
    _utils.savefig_or_show("filter_genes_dispersion", show=show, save=save)
    if show is False:
        return pl.gca()


# backwards compat
def filter_genes_dispersion(
    result: np.recarray,
    log: bool = False,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
    palette: Optional[Union[list, None]] = None, **kwargs
):
    """
    Plot dispersions versus means for genes.

    Produces Supp. Fig. 5c of Zheng et al. (2017) and MeanVarPlot()
    of Seurat.

    Parameters
    ----------
    result
        Result of :func:`~scanpy.pp.filter_genes_dispersion`.
    log
        Plot on logarithmic axes.
    show
         Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure.
        A string is appended to the default filename.
        Infer the filetype if ends on {{`'.pdf'`, `'.png'`, `'.svg'`}}.
    """
    plot_hvgs(
        result, log=log, show=show, save=save, highly_variable_genes=False,
        palette=palette, **kwargs
    )

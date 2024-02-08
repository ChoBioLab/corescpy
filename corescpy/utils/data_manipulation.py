import decoupler as dc
from scanpy._settings import settings
import scanpy as sc
import pandas as pd
import numpy as np
from corescpy.processing import get_layer_dict

layers = get_layer_dict()


def create_pseudobulk(adata, col_cell_type, col_sample_id=None,
                      layer=layers["counts"], mode="sum",
                      kws_process=True, **kwargs):
    """Get pseudo-bulk of scRNA-seq data."""
    if kws_process is True:
        kws_process = dict(target_sum=1e6, max_value=10, n_comps=10)
    if layer and layer not in adata.layers:
        raise ValueError(f"{layer} not in adata.layers. Set layer argument to"
                         " None or the name of the layer with count data.")
    pdata = dc.get_pseudobulk(
        adata, sample_col=col_sample_id, groups_col=col_cell_type,
        layer=layer, mode=mode, **kwargs)
    pdata.layers[layers["counts"]] = pdata.X.copy()
    if kws_process is True or isinstance(kws_process, dict):
        sc.pp.normalize_total(pdata, target_sum=kws_process["target_sum"])
        sc.pp.log1p(pdata)
        pdata.layers[layers["log1p"]] = pdata.X.copy()
        sc.pp.scale(pdata, max_value=kws_process["max_value"])
        pdata.layers[layers["scaled"]] = pdata.X.copy()
        sc.tl.pca(pdata, n_comps=kws_process["n_comps"])
    return pdata


def create_condition_combo(adata, col_condition, col_label_new=None, sep="_"):
    """Create a column representing combination of multiple columns."""
    if col_label_new is None:
        col_label_new = "_".join(col_condition)
    if isinstance(adata, pd.DataFrame):  # if adata is dataframe
        adata[col_label_new] = adata[col_condition[0]]  # start w/ 1st
        for x in col_condition[1:]:  # loop to add "_{condition value}"...
            adata[col_label_new] = adata[col_label_new].astype(
                "string") + sep + adata[x]  # ...to make full combination
    else:  # if adata is an AnnData object
        adata.obs = create_condition_combo(adata.obs, col_condition,
                                           col_label_new=col_label_new)
    return adata


def _merge_pca_subset(adata, adata_subset, n_comps=None,
                      key_added="X_pca", retain_cols=True):
    ann = adata.copy()
    ixs = np.array(pd.Series(ann.var.index.values).isin(list(
        adata_subset.var.index.values)))
    n_comps = adata_subset.varm["PCs"].shape[1]
    ann.uns["pca"] = adata_subset.uns["pca"]
    ann.obsm[key_added] = adata_subset.obsm[key_added]
    for i in adata_subset.varm:
        ann.varm["PCs"] = np.zeros(shape=(ann.n_vars, n_comps))
        ann.varm["PCs"][ixs] = adata_subset.varm["PCs"]
    if retain_cols is True:
        ann.obs = ann.obs.join(adata_subset.obs, lsuffix="_pre_pca_subset")
    else:
        ann.obs = ann.obs.drop(list(ann.obs.columns.intersection(
            adata_subset.obs.columns)), axis=1).join(adata_subset.obs)
    return ann


rfx_convert = r"""
require(Seurat)
require(zellkonverter)

extract_layers <- function(seu, assay = NULL, layers = NULL) {
    layers_x <- list()
    for (x in assay) {
        lays <- ifelse(is.null(layers), slotNames(seu@assays[[x]]), intersect(
            layers, slotNames(seu@assays[[x]])))  # all possible layers
        lays <- sapply(lays, function(
            u) ifelse(dim(seu@assays[[x]][u])[1] > 0, u, NA))  # non-empty
        lays <- lays[!is.na(lays)]  # names of *available* layers
        if (length(lays) == 0) next  # skip if no available layers
        layers_x[[x]] <- lapply(lays, function(i) seu@assays[[x]][i])  # .Xs
    }
    return(layers_x)
}

convert <- function(file, file_new = NULL, assay = NULL, overwrite = FALSE,
                    write_metadata = FALSE, layers = NULL) {
    if (is.null(file_new)) {
        file_new <- gsub(".RDS", "_converted.h5ad", file)
        if (file.exists(file_new) && !overwrite) {
            stop(paste0("Out file ", file_new, " exists; overwrite=F."))
        } else {
            cat(paste0("Output path not provided; setting as ", file_new))
        }
    }
    cat(paste0("\n\n<<< READING DATA >>>\n", file))
    seu <- readRDS(file)  # read RDS into Seurat object
    print(str(seu, 3))  # print object structure (3 levels deep)
    if (is.null(assay)) {
        assay <- names(seu@assays)  # all assays if not set
    }
    cat("\n\n<<< CONVERTING DATA TO SINGLE CELL OBJECT >>>")
    sce <- Seurat::as.SingleCellExperiment(seu, assay = assay)  # convert
    print(sce)
    if (!identical(file_new, FALSE)) {  # if "file_new" is False; don't write
        cat(paste0("\n\n<<< READING DATA >>>\n", file_new))
        zellkonverter::writeH5AD(sce, file_new)  # write .h5ad
    }
    layers_x <- extract_layers(seu, assays, layers)
    if (write_metadata) {  # write metadata about object for later checks
        print("WRITING METADATA NOT YET SUPPORTED")
        file_meta <- gsub(".RDS", "_converted_metadata.csv", file)
        clusters <- ifelse("active.ident" %in% slotNames(seu),
                           as.data.frame(seu@active.ident), NULL)
        meta <- list(assays=names(seu@assays), uns=names(seu@reductions),
                     var_names=rownames(seu@assays[[names(seu@assays)[1]]]),
                     X=layers_x, clusters=clusters, obs=seu@meta.data)
        if (file.exists(file_meta) && !overwrite) {
            stop(paste0("Metadata file ", file_meta,
                        "exists and overwrite is not allowed."))
        }
        # write.csv(meta, file = file_meta, row.names = FALSE)
    }
    return(list(seu, sce, meta))
}
"""

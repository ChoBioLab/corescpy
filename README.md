# CRISPR Pipeline

Developer: Elizabeth Aslinger (easlinger)

Correspondence: elizabeth.aslinger@aya.yale.edu

[Jira Epic](https://mssm-ipm.atlassian.net/browse/CHOLAB-676)

---

## Installation

1. Open a Unix terminal (often Ctrl + Alt  + T on Linux).

2. Install conda environment from .yml file (replace "env-crispr"
with desired environment name):
`conda create -n crispr python=3.10.4  # create python environment`

3. Activate the conda environment with `conda activate crispr`.

4. Clone the repository to your local computer:
`git clone git@github.com:ChoBioLab/crispr.git`,
`git clone https://github.com/ChoBioLab/crispr.git`, or
look above for the green "Code" button and press it for instructions.

5. Naviate to the repository directory (replace
<DIRECTORY> with your path):
`cd <DIRECTORY>`

6. Install the package with pip. (Ensure you have pip installed.)
`pip install .`

## Usage

1. You can now load `crispr` like any other distributed Python package.
Open a Python terminal and type:
`import crispr as cr`

2. You can now call functions from the analysis module using
`cr.ax.<FUNCTION>()`, from the preprocessing using `cr.ax.pp...`, etc.
in Python; however, you are most likely to interact with the `Crispr`
class object. Here is example code you might run
(replacing <...> with your argument specifications) to load and
preprocess your data.
```
from crispr.crispr_class import Crispr
self = Crispr(adata, <...>)
self.preprocess(<...>)
```
etc.

Here are the methods (applicable to scRNA-seq generally, not just perturbations) **in order** of a typical workflow _(replace ... with argument specifications)_:

* `self.preprocess(...)`: Perform filtering, normalization, scaling, quality control analysis, selection for highly variable genes, regressing out confounds, and intial exploration of the data (e.g., cell counts, mitochondrial counts, etc.).
* `self.cluster(...)`: Perform dimensionality reduction (e.g., PCA) and clustering (Louvain or Leiden), as well as related visualization.
* `self.plot(...)`: Create additional basic plots (e.g., dot, matrix, and violin gene expression plots).

The following perturbation-specific methods can be executed optionally and in any order:

* `self.run_augur(...)`: Score and plot how strongly different cell types responded to perturbation(s). This score is operationalized as the accuracy with which a machine learning model can use gene expression data to predict the perturbation condition to which cells of a given type belong. Augur provides scores aggregated across cells of a given type rather than for individual cells.
* `self.run_mixscape(...)`: Quantify and plot the extent to which individual cells responded to CRISPR perturbation(s), and identify which perturbation condition cells were not detectibly perturbed in terms of their gene expression.
* `self.compute_distance(...)`: Calculate and visualize various distance metrics that quantify the similarity in gene expression profiles across perturbation conditions.
* `self.run_composition_analysis(...)`: Analyze and visualize shifts in the cell type composition across perturbation conditions.
* `self.run_dialogue(...)`: Create plots showing multi-cellular programs.

## Package Overview

<u> Argument Conventions: </u>

Certain arguments used throughout the `crispr` package (including outside the `crispr.crispr_class.Crispr()` class), hold to conventions intended to foster readability, maintain consistency, and promote clarity, both for end-users and future developers.

* Arguments starting in `col_` and `key_`
    - The "col_" prefix indicates that an argument refers to a column name (often in `.adata.obs` or `.adata.var`), while the "key_" prefix means you're meant to specify a type of entry in a column. For instance, assume the column "condition" contains the names of different experimental conditions (drug A, drug B, drug C). In a function where you want to compare, for instance, drug A vs. control, you would specify `key_treatment="drug A"` and k
    - These names may
        * already exist (or will exist in the `.adata` attribute immediately upon creating the AnnData object from the data file) or
        * may yet to be created, namely, after object initialization by running the object's methods. Thus, you may specify what you want certain columns to be named (e.g., the binary perturbed/non-perturbed column) or what entries within a column will be called (e.g., "Control" for rows within the `col_control` corresponding to cells that have control guide RNA IDs in `col_guide_rna`), for aesthetics, customizability to your design/interpretability, and/or to avoid duplicating pre-existing names.
    - These arguments will be entered as items (with the argument names as keys) in dictionaries stored in the object attributes `._columns` and `._keys`, respectively.
    - These arguments will often be passed by default (or will force them as specifications) to various object methods.
    - In certain methods, you can specify a new column to use just for that function. For instance, if you have a column containing CellTypist annotations and want to use those clusters instead of the "leiden" ones for the `run_dialogue()` method, you can specify in that method (`run_dialogue(col_cell_type="majority_voting")`) without changing the attribute (`self._columns`) that contains your original specification here.

<!-- break -->

* `col_perturbed` (binary) vs. `col_condition` (can have >= 3 categories)
    - In the `Crispr` class object, `col_perturbed` is meant to be a binary column that has `key_control` as the entry for control rows and `key_treatment` for all other experimental conditions.
        * For instance, for a CRISPR design targeting more than one gene, `col_perturbed` would contain only `key_treatment` (i.e., all perturbed cells, regardless of targeted gene) while `col_condition` would contain entries specifying the particular gene(s) targeted (or `key_control`).
        * A drug design targeting more than one gene, `col_perturbed` would contain only `key_treatment` (i.e., all perturbed cells, regardless of targeted gene) while `col_condition` would contain entries specifying the particular gene(s) targeted (or `key_control`).
        * If the design only targets one gene/has one treatment conditions/etc., these columns would simply be equivalent.
    - In the `Crispr` class object, it is created during object initialization as a column (named after your specification of `col_perturbed`) in `.obs`. All rows in `.obs[col_condition]` that do not = `key_control` will be set as `key_treatment`.
    - In the `crispr` package more broadly, if a function calls for a `col_perturbed` argument, that indicates that it works with binary categories only. If it is fed a column with three or more categories, it will either subset the data to include only rows where that column = `key_treatment` or `key_control` (desirable behavior if you want to compare only a subset of the existing conditions, but undesirable if you want to look at, say, any drug vs. control, where the desired "drug" category consists of rows where the column = "drug A" and "drug B"), or it will throw an error.

<!-- break -->

* `col_guide_rna`

### Initialization Method Arguments

* `file_path` **(str, AnnData, or dictionary)**: Path or object containing data. Used in initialization to create the initial `self.adata` attribute (an AnnData or MuData object). Either
    - a path to a 10x directory (with matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz),
    - a path to an .h5ad or .mu file (Scanpy/AnnData/Muon-compatible),
    - an AnnData or MuData object (e.g., already loaded with Scanpy or Muon, or by using `crispr.pp.create_object(file_path)`), or
    - a dictionary containing keyword arguments to pass to  `crispr.pp.combine_matrix_protospacer()` (in order to load information about perturbations from other file(s); press the arrow to expand details here),

<details><summary>Click to expand details</summary>

```
crd = "<YOUR DIRECTORY HERE>"
# e.g., "/home/asline01/projects/crispr/examples/data/crispr-screening/HH03"

subd = "<YOUR SUB-DIRECTORY WITH THE .mtx, barcodes, features files HERE>"
# e.g., "filtered_feature_bc_matrix"

proto = "<YOUR PROTOSPACER .csv HERE; file should be under `crdir` directory>"
# e.g., "crispr_analysis/protospacer_calls_per_cell.csv"

file_path = dict(directory=crd, subdirectory_mtx=subd, file_protospacer=proto)
```
If you have the typical/default file tree/naming (e.g., "filtered_feature_bc_matrix" and "crispr_analysis/protospacer_calls_per_cell.csv" are contained in the directory defined in `file_path["directory"]`), you should be able to specify just `file_path=dict(directory=<YOUR DIRECTORY HERE>)` (e.g., `file_path=dict(directory="/home/projects/crispr-screening/crispr-screening/analysis/cellranger/cr_count_2023-05-15_1837/HH02/outs")`).

</details>

   or
    - to concatenate multiple datasets, a dictionary (keyed by your desired subject/sample names to be used in `col_sample_id`) consisting of whatever objects you would pass to `create_object()`'s `file` argument for the individual objects. You must also specify `col_sample` (a tuple as described in the documentation below). The other arguments passed to the `crispr.pp.create_object()` function (e.g., `col_gene_symbols`) can be specified as normal if they are common across samples; otherwise, specify them as lists in the same order as the `file` dictionary.

<!-- break -->

* `assay` **(str, optional)**: Name of the gene expression assay if loading a multi-modal data object (e.g., "rna"). Defaults to None (i.e., `self.adata` is single-modal).

* `assay_protein` **(str, optional)**: Name of the assay containing the protein expression modality, if available. For instance, if "adt", `self.adata["adt"]` would be expected to contain the AnnData object for the protein modality. ONLY FOR MULTI-MODAL DATA for certain bonus visualization methods. Defaults to None.

* `col_gene_symbols` **(str, optional)**: Column name in `.var` for gene symbols. Defaults to "gene_symbols".

* `col_cell_type` **(str, optional)**: Column name in `.obs` for cell type. Defaults to "leiden" (anticipating that you will run `self.cluster(...)` with `method_cluster="leiden"`). This column may be
    - pre-existing in data (e.g., pre-run clustering column or manual annotations), or
    - expected to be created via `Crispr.cluster()`.

<!-- break -->

* `col_sample_id` **(str or tuple, optional)**: Column in `.obs` with sample IDs. Defaults to "standard_sample_id". If this column does not yet exist in your data and needs to be created by concatenating datasets, you must provide `file_path` as a dictionary keyed by desired `col_sample_id` values as well as signal that this needs to happen by specifying col_sample_id as a tuple, with the second element containing a dictionary of keyword arguments to pass to `AnnData.concatenate()` or None (to use defaults).

* `col_batch` **(str, optional)**: Column in `.obs` with batch IDs. Defaults to None.

* `col_condition` **(str, optional)**: Either the name of an existing column in `.obs` indicating the experimental condition to which each cell belongs or (for CRISPR designs) the **desired** name of the column that will be created from `col_guide_rna` to indicate the gene(s) targeted in each cell.
    - If there are multiple conditions besides control (e.g., multiple types of drugs and/or exposure times, multiple target genes in CRISPR), this column distinguishes the different conditions, in contrast to `col_perturbed` (a binary treated/perturbed vs. control indicator).
    - In CRISPR designs (i.e., when `col_guide_rna` is specified), this column will be where each guide RNA's target gene will be stored, whether pre-existing (copied directly from `col_guide_rna` if `kws_process_guide_rna` is None) or created during the Crispr object initialization by passing `col_guide_rna` and `kws_process_guide_rna` to `crispr.pp.filter_by_guide_counts()` in order to convert particular guide RNA IDs to their target(s) (e.g., STAT1-1|CCL8-2-1|NegCtrl32a => STAT1|CCL8|Control). Defaults to None.
    - For non-CRISPR designs (e.g., pharmacological treatment):
        - This column should exist in the AnnData or MuData object (either already available upon simply reading the specified file with no other alterations, or as originally passed to the initialization method if given instead of a file path).
        - It should contain a single `key_control`, but it can have multiple categories of other entries that all translate to `key_treatment` in `col_perturbed`.
        - If you have multiple control conditions, you should pass an already-created AnnData object to the `file_path` argument of the `Crispr` class initialization method after adding a separate column with a name different from those specified in any of the other column arguments. You can then pass that column name manually to certain functions' `col_control` arguments and specify the particular control condition in `key_control`.
    - In most methods, `key_control` and `key_treatment`, as well as `col_perturbed` or `col_condition` (for methods that don't require binary labeling), can be newly-specified so that you can compare different conditions within this column. If the argument is named `col_perturbed`, passing a column with more than two categories usually results in subsetting the data to compare only the two conditions specified in `key_treatment` and `key_control`. The exception is where there is a `key_treatment_list` or similarly-named argument.

<!-- break -->

* `col_perturbed` **(str, optional)**: Column in `.obs` where class methods will be able to find the binary experimental condition variable. It will be created during `Crispr` object initialization as a binary version of `col_condition`. Defaults to "perturbation". For CRISPR designs, all entries containing the patterns specified in `kws_process_guide_rna["key_control_patterns"]` will be changed to `key_control`, and all cells with targeting guides will be changed to `key_treatment`.

<!-- break -->

* `col_guide_rna` **(str, optional)**: Column in `.obs` with guide RNA IDs. Defaults to "guide_ids". This column should always be specified for CRISPR designs and should NOT be specified for other designs.
    - If only one kind of guide RNA is used, then this should be a column containing the name of the gene targeted (for perturbed cells) and the names of any controls, and `key_treatment` should be the name of the gene targeted. Then, `col_condition` will be a direct copy of this column.
    - Entries in this column should be either gene names in `self.adata.var_names` (or `key_control` or one of the patterns in `kws_process_guide_rna["key_control_patterns"]`), plus, optionally, suffixes separating guide #s (e.g., STAT1-1-2, CTRL-1) and/or with a character that splits separate guide RNAs within that cell (if multiply-transfected cells are present). These characters should be specified in `kws_process_guide_rna["guide_split"]` and `kws_process_guide_rna["feature_split"]`, respectively. For instance, they would be "-" and "|", if `col_guide_rna` entries for a cell multiply transfected by two sgRNAs targeting STAT1, two control guide RNAs, and a guide targeting CCL5  would look like "STAT1-1-1|STAT1-1-2|CNTRL-1-1|CCL5-1".
    - Currently, everything after the first dash (or whatever split character is specified) is discarded when creating `col_target_genes`, so keep that in mind.
    - This column will be stored as `<col_guide_rna>_original` if `kws_process_guide_rna` is not None, as that will result in a processed version of this column being stored under `self.adata.obs[<col_guide_rna>]`.

<!-- break -->

* `col_num_umis` **(str, optional)**: Name of column in `.obs` with the UMI counts. This should be specified if `kws_process_guide_rna` is not None. For designs with multiply-transfected cells, it should follow the same convention established in `kws_process_guide_rna["feature_split"]`. Defaults to "num_umis".

* `key_control` **(str, optional)**: The label that is or will be in `col_condition`, `col_guide_rna`, and `col_perturbed` indicating control rows. Defaults to "NT". Either
    - exists as entries in pre-existing column(s), or
    - is the name you want the control entries (detected using `.obs[<col_guide_rna>]` and `kws_process_guide_rna["key_control_patterns"]`) to be categorized as control rows under the new version(s) of `.obs[<col_guide_rna>]`, `.obs[<col_target_genes>]`, and/or `.obs[<col_perturbed>]`. For instance, entries like "CNTRL-1", "NEGCNTRL", "Control", etc. in `col_guide_rna` would all be keyed as "Control" in (the new versions of) `col_target_genes`, `col_guide_rna`, and `col_perturbed` if you specify `key_control="Control` and `kws_process_guide_rna=dict(key_control_patterns=["CTRL", "Control"])`.

<!-- break -->

* `key_treatment` **(str, optional)**: What entries in `col_perturbed` indicate a treatment condition (e.g., drug administration, CRISPR knock-out/down) as opposed to a control condition? This name will also be used for Mixscape classification labeling. Defaults to "KO".

<!-- break -->

* `key_nonperturbed` **(str, optional)**: What will be stored in the `mixscape_class_global` and related columns/labels after running Mixscape methods. Indicates cells without a detectible perturbation. Defaults to "NP".

* `kws_process_guide_rna` (dict, optional): Dictionary of keyword arguments to pass to `crispr.pp.filter_by_guide_counts()`. (See below and crispr.processing.preprocessing documentation). Defaults to None (no processing will take place, in which case BE SURE THAT `col_target_genes` already exists in the data once loaded and contains the already-filtered, summed up, generic gene-named, etc. versions of the guide RNA column). Keys of this dictionary should be:
    - key_control_patterns (list, optional): List (or single string) of patterns in guide RNA column entries that correspond to a control. For instance, if control entries in the original `col_guide_rna` column include `NEGCNTRL` and `Control.D`, you should specify ['Control', 'CNTRL'] (assuming no non-control sgRNA names contain those patterns). If blank entries should be interpreted as control guides, then include np.nan/numpy.nan in this list. Defaults to None, which turns to [np.nan].
    - `max_percent_umis_control_drop` (int, optional): If control UMI counts are $<=$ this percentage of the total counts for that cell, and if a non-control sgRNA is also present and meets other filtering criteria, then consider that cell pseudo-single-transfected (non-control gene). Defaults to 75.
    - `min_percent_umis` (int, optional): sgRNAs with counts below this percentage will be considered noise for that guide. Defaults to 40.
    - `feature_split` (str, optional): For designs with multiple guides, the character that splits guide names in `col_guide_rna`. For instance, "|" for `STAT1-1|CNTRL-1|CDKN1A`. Defaults to "|". If only single guides, you should set to None.
    - `guide_split` (str, optional): The character that separates guide (rather than gene target)-specific IDs within gene. For instance, guides targeting STAT1 may include STAT1-1, STAT1-2-1, etc.; the argument would be "-" so the function can identify all of those as targeting STAT1. Defaults to "-".

<!-- break -->

* `remove_multi_transfected` (bool, optional): In designs with multiple guides per cell, remove multiply-transfected cells (i.e., cells where more than one target guide survived
    application of any filtering criteria set in `kws_process_guide_rna`). If `kws_process_guide_rna["max_percent_umis_control_drop"]` is greater than 0, then cells with one target guide and control guides which together make up less than `max_percent_umis_control_drop`% of total UMI counts will be considered pseudo-single-transfected for the target guide. Defaults to True. Some functionality may be limited and/or problems occur if set to False and if multiply-transfected cells remain in data.

<!-- break -->

### Crispr Object Properties

The `crispr.crispr_object.Crispr()` class object is an end user's main way of interacting with the package as a whole. (See above for an overview of the workflow.) See the notebooks in [/examples](https://github.com/ChoBioLab/crispr/tree/a8564fb02d7ef2983432c6ca6035d2b77458dbbe/examples) for additional help.

#### Major Attributes Descriptions

* `.adata`: AnnData object. Columns or other objects created in the course of running certain methods may also be stored in its various attributes. Below are listed some of the major attributes of `.adata`. Note that for multi-modal data (self._assay is not None), some of these attributes may need to be accessed by `.adata[self._assay].<attribute>`, but for brevity, we'll refer to `.adata` here. Not all will/have to be present, except `.X`, `.obs`, and `.var`.
    -  `.X`: Sparse matrix of data originally passed to the function to create an AnnData object (e.g., from CellRanger output).
    -  `.layers`: Contains different versions of `adata.X`, such as scaled data (`adata.layers["scaled"]`) or that created by calculating a
                  Mixscape perturbation signature (`adata.layers["X_pert"]`, by default).
    -  `.obs`: pandas.DataFrame of observations (rows=cells). You can store additional data about individual cells/samples/etc. here by assigning a new column.
    -  `.obsm`: xxxxxxxxxxxxxxxxxxxxxxxxxx
    -  `.obsm`: xxxxxxxxxxxxxxxxxxxxxxxxxx
    -  `.var`: pandas.DataFrame of observations (rows=cells). You can store additional data about individual cells/samples/etc. here by assigning a new column.
            Often contains the gene symbols and EnsemblIDs (either of which is often the index/`.var_names`),
            "feature_types" (e.g., "Gene Expression"), and, after preprocessing, may contain columns such as the number of cells expressing that feature ("n_cells"),
            whether that feature is a mitochonrial ("mt") and/or highly variable ("highly_variable") gene, mean and total counts, percent dropout, means, dispersions,
            and normalized versions of these metrics.
    -  `.obs_names`: Row indices of `.obs` (e.g., cell barcodes). Changing this attribute changes this index, and has other potential benefits/consequences.
    -  `.var_names`: Row indices of `.var` (i.e., gene names). Changing this attribute changes this index, and has other potential benefits/consequences.
    -  `.n_obs`: Number of observations (i.e., cells).
    -  `.n_vars`: Number of features (i.e., genes/proteins/etc.).
* `.results`: A dictionary with many of the dataframes, arrays, and other non-plot-based output of analyses.
* `.figures`: A dictionary with figures from output of your analyses.
* `.info`: Contains miscellaneous information about your data, the Crispr object (e.g., its analysis history), argument specifications, etc.

#### Accessing AnnData and Attributes Directly and Using Aliases

The AnnData object is stored in the attribute `adata`, so if your object is called `self`, you can access it using `self.adata`. (For examples going forward, we will assume the object is called `self`, but you can substitute any name you want by assigning the Crispr object to some other name instead.)

If you have multiple modalities, you can access the gene expression modality using either `self.adata[self._assay]` (having specified assay=the name of the RNA modality in your AnnData, which is usually "rna," in the `Cripsr()` initialization method call when you first create your object) or using the alias `self.rna`.

Thus, if you have multi-modal data in `self.adata`, it's convenient to access the AnnData attributes specifically of your AnnData's gene expression modality using, for instance, the alias `self.rna.obs` instead of the long-form `self.adata[self._assay].obs`.

These aliases are not only convenient for their brevity, but also allow for a more generalizable way to call specific objects. For instance, if you wanted to write a script that frequently calls the `.obs` attribute of the RNA data, and you want it to work for both uni- and multi-modal data, instead of repeatedly writing, for example:

```
if self._assay is None:
    custom_function(self.adata[self._assay].obs)
else:
    custom_function(self.adata[self._assay].obs)
```

you may simply say `self.rna.obs`, knowing it will work whether or not multiple assays exist in the object's AnnData attribute.

Finally, this approach saves memory: All these versions of the attribute are stored in a single place in memory so you can call the attributes in various ways without duplicating them and taking up more space.

---

## Resources for Background Knowledge


[Pertpy Tutorials](https://pertpy.readthedocs.io/en/latest/tutorials/index.html)
[Single Cell Best Practices](https://www.sc-best-practices.org/conditions/perturbation_modeling.html)
[Augur](https://github.com/neurorestore/Augur)
[Mixscape (Seurat)](https://satijalab.org/seurat/articles/mixscape_vignette.html)

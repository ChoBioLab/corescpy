import pertpy
import os
import pathlib
import crispr as cr
import pandas as pd
import numpy as np


# Detect current directory of file
DIR = pathlib.Path(__file__).parent.resolve()
DIR = os.path.join(DIR, "data")

files_data = {
    "CRISPRi_scr": f"{DIR}/crispr-screening/filtered_feature_bc_matrix_HH06.h5",
    "CRISPRi_wgs": f"{DIR}/replogle_2022_k562_gwps.h5ad",  # perturb-seq (WGS) 
    "CRISPRi_ess": f"{DIR}/replogle_2022_k562_esss.h5ad",  # perturb-seq
    "pool": f"{DIR}/norman_2019_raw.h5ad",
    "bulk": f"{DIR}/burczynski_crohn.h5ad",
    "screen": f"{DIR}/dixit_2016_raw.h5ad",
    "perturb-seq": f"{DIR}/adamson_2016_upr_perturb_seq.h5ad",
    "ECCITE": f"{DIR}/papalexi_2021.h5ad",
    "coda": f"{DIR}/haber_2017_regions.h5ad",
    "CRISPRa": f"{DIR}/tian_2021_crispra.h5ad",  # CROP-seq CRISPRa
    "augur_ex": f"{DIR}/bhattacherjee.h5ad"
}

col_cell_type_data = {
    "CRISPRi_scr": "leiden",  # because unannotated
    "CRISPRi_wgs": "leiden",
    "CRISPRi_ess": "leiden",
    "pool": "",
    "bulk": None,
    "screen": None,
    "perturb-seq": "cell_label",
    "ECCITE": "leiden",
    "coda": "cell_label",
    "augur_ex": "cell_type"  # "subtype" also
}

col_gene_symbols_data = {
    "CRISPRi_scr": "gene_symbols",
    "CRISPRi_wgs": "gene",  # ?
    "CRISPRi_ess": "gene_symbols",
    "pool": "gene_symbols",
    "bulk": None,
    "screen": None,
    "ECCITE": None,
    "coda": "gene",
    "augur_ex": "name"
}

assays_data = {
    "CRISPRi_scr": None,
    "CRISPRi_wgs": None,
    "CRISPRi_ess": None,
    "pool": None,
    "bulk": None,
    "screen": None,
    "ECCITE": ["rna", "adt"],  # RNA, protein
    "coda": None,
    "augur_ex": None
}

col_split_by_data = {
    "CRISPRi_scr": None,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": np.nan,
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "replicate",
    "coda": np.nan,
    "augur_ex": np.nan
    
}

col_perturbation_data = {
    "CRISPRi_scr": "name",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "Condition",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "perturbation",
    "coda": "condition",
    "augur_ex": "label"
}

key_control_data = {
    "CRISPRi_scr": "Non-Targeting",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "Control",  # must modify NaNs in guide_ids column
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "coda": "Control",
    "ECCITE": "NT",
    "augur_ex": "Maintenance_Cocaine"
}

key_treatment_data = {
    "CRISPRi_scr": np.nan,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": np.nan,
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "Perturbed",
    "coda": "Salmonella",
    "augur_ex": "Withdraw_48h_Cocaine"
}

label_perturbation_type_data = {
    "CRISPRi_scr": "KD",
    "CRISPRi_wgs": "KD",
    "CRISPRi_ess": "KD",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "KO",
    "coda": "Salmonella",
    "augur_ex": "Cocaine_Withdrawal"
}

col_target_genes_data = {
    "CRISPRi_scr": "target_gene_name",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "guide_ids",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "gene_target",
    "coda": np.nan,
    "augur_ex": np.nan
}

col_guide_rna_data = {
    "CRISPRi_scr": "name",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "guide_ids",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "guide_ID",
    "coda": np.nan,
    "augur_ex": np.nan
}

layer_perturbation_data = {
    "CRISPRi_scr": np.nan,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": None,
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "X_pert",
    "coda": None, 
    "augur_ex": None
}

col_sample_id_data = {
    "CRISPRi_scr": np.nan,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "gemgroup",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "orig.ident",
    "coda": np.nan, 
    "augur_ex": "orig.idents"
}


def load_example_data(file, col_gene_symbols, write_public=False):
    """(Down)load data for examples/vignettes.
    Args:
        file (str): Name of data (see keys of dictionaries in config).
        col_gene_symbols (str): Name of column in `AnnData.obs` 
            that has gene symbols.
        write_public (bool, optional): If you have to download from pertpy.data
            (i.e., hasn't already been saved in examples/data)
            write it to examples/data once downloaded? Defaults to False.
    """
    adata, err = None, None
    if file in files_data:  # if file previously downloaded then stored
        file_path = files_data[file]
    else:
        file_path = file
    print(f"File Path: {file_path}")
    if os.path.exists(file_path):
        print(f"\n\n{file_path} exists.")
        # try:  # try to create scanpy object from file
        if os.path.splitext(file_path)[1] == ".h5":
            kwargs = dict(genome=None, gex_only=False, backup_url=None)
        else:
            kwargs = {}
        adata = cr.pp.create_object(file_path, assay=None, 
                                    col_gene_symbols=col_gene_symbols,
                                    **kwargs)
        # except Exception as e:
        #     err = f"\n\n{'=' * 80}\n\n{file_path} failed to load:\n\n{e}"
        #     print(err)
    if adata is None:  # if file doesn't exist or failed to load
        if file in files_data:
            print(f"\n\nLooking for downloadable files for: {file}.")
            if file == "CRISPRi_wgs":  # CRISPRi Perturb-seq Pertpy data
                adata = pertpy.data.replogle_2022_k562_gwps()  # HJ design
                # adata = pertpy.data.replogle_2022_k562_essential()  # ~1 hr. 
                # adata = pertpy.data.replogle_2022_rpe1()
                # adata = pertpy.data.adamson_2016_upr_perturb_seq()  # ~8 min.
            elif file == "CRISPRi_ess":
                adata = pertpy.data.replogle_2022_k562_essential()  # HJ design
                col_target_genes = col_target_genes_data[file]
                adata.obs[col_perturbation_data[file]] = adata.obs.apply(
                            lambda x: key_control_data[file] if str(
                                x["guide_ids"]) == key_control_data[
                                    file] or pd.isnull(
                                        x["guide_ids"]) else x[
                                            "guide_identities"], axis=1)
                adata.obs[col_target_genes] = adata.obs[
                    col_target_genes].str.strip(" ").replace(
                        "", np.nan).apply(
                            lambda x: key_control_data[file] if pd.isnull(
                                x) else x)
            elif file == "screen":  # Perturb-seq CRISPR screen Pertpy data
                adata = pertpy.data.dixit_2016_raw()
            elif file == "bulk":  # bulk RNA-seq data
                adata = pertpy.data.burczynski_crohn()
            elif file == "pool":
                adata = pertpy.data.norman_2019_raw()  # download ~ 10 minutes
            elif file == "ECCITE":
                adata = pertpy.data.papalexi_2021()  # sc CRISPR screen+protein
            elif file == "augur_ex":  # Pertpy's AUGUR example dataset
                adata = pertpy.data.bhattacherjee()
            elif file == "coda":
                adata = pertpy.data.haber_2017_regions()
            elif file == "CRISPRa":
                adata = pertpy.data.tian_2021_crispra()
            elif file == "perturb-seq":
                adata = pertpy.data.adamson_2016_upr_perturb_seq()
            else:
                if err:
                    raise ValueError(f"{file} error:\n\n{err}.")
                else:
                    raise ValueError(
                        f"{file} not a valid public data download option.")
            if write_public is True:
                adata.write(file_path)
        else:
            raise ValueError(f"{file_path} does not exist.")
    return adata
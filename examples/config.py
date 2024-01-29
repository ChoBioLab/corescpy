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
    "CRISPRi_scr": dict(directory=f"{DIR}/crispr-screening/HH06"),
    "CRISPRi_scr_multi": {
        # "HH03": dict(directory=f"{DIR}/crispr-screening/HH03"),
        "HH06": dict(directory=f"{DIR}/crispr-screening/HH06"),
        "CR4": dict(directory=f"{DIR}/crispr-screening/HH-Hu-CR4"),
        "CR5": dict(directory=f"{DIR}/crispr-screening/HH-Hu-CR5")
    },
    "CRISPRi_wgs": f"{DIR}/replogle_2022_k562_gwps.h5ad",  # perturb-seq (WGS) 
    "CRISPRi_ess": f"{DIR}/replogle_2022_k562_esss.h5ad",  # perturb-seq
    "pool": f"{DIR}/norman_2019_raw.h5ad",
    "bulk": f"{DIR}/burczynski_crohn.h5ad",
    "screen": f"{DIR}/dixit_2016_raw.h5ad",
    "perturb-seq": f"{DIR}/adamson_2016_upr_perturb_seq.h5ad",
    "ECCITE": f"{DIR}/papalexi_2021.h5ad",
    "coda": f"{DIR}/haber_2017_regions.h5ad",
    # "CRISPRa": f"{DIR}/tian_2021_crispra.h5ad",  # CROP-seq CRISPRa
    "CRISPRa": f"{DIR}/tian_2021_crispra.h5ad",  # Perturb-seq CRISPRa
    "augur_ex": f"{DIR}/bhattacherjee.h5ad",
    "adamson": f"{DIR}/adamson_2016_upr_perturb_seq.h5ad"
}

col_cell_type_data = {
    "CRISPRi_scr": "predicted_labels",
    "CRISPRi_scr_multi": "predicted_labels",
    "CRISPRi_wgs": "leiden",
    "CRISPRi_ess": "leiden",
    "pool": "",
    "bulk": None,
    "screen": None,
    "perturb-seq": "celltype",
    "ECCITE": "leiden",
    "coda": "cell_label",
    "augur_ex": "cell_type",  # "subtype" also,
    "adamson": "leiden"
}

col_gene_symbols_data = {
    "CRISPRi_scr": "gene_symbols",
    "CRISPRi_scr_multi": "gene_symbols",
    "CRISPRi_wgs": "gene",  # ?
    "CRISPRi_ess": "gene_symbols",
    "pool": "gene_symbols",
    "bulk": None,
    "screen": None,
    "perturb-seq": "gene_symbol",
    "ECCITE": "name",
    "coda": "index",
    "augur_ex": "name",
    "adamson": "leiden"
}

assays_data = {
    "CRISPRi_scr": None,
    "CRISPRi_scr_multi": None,
    "CRISPRi_wgs": None,
    "CRISPRi_ess": None,
    "pool": None,
    "bulk": None,
    "screen": None,
    "perturb-seq": None,
    "ECCITE": ["rna", "adt"],  # RNA, protein
    "coda": None,
    "augur_ex": None,
    "adamson": None
}

col_split_by_data = {
    "CRISPRi_scr": None,
    "CRISPRi_scr_multi": "orig.ident",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": None,
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "perturb-seq": None,
    "ECCITE": "replicate",
    "coda": np.nan,
    "augur_ex": "nan"
}

col_perturbed_data = {
    "CRISPRi_scr": "perturbation",
    "CRISPRi_scr_multi": "perturbation",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "perturbation",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "perturb-seq": "perturbation",
    "ECCITE": "perturbation",
    "coda": "condition",
    "augur_ex": "label"
}

key_control_data = {
    "CRISPRi_scr": "NT",
    "CRISPRi_scr_multi": "NT",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "NT",  # must modify NaNs in guide_ids column
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "coda": "Control",
    "perturb-seq": "Control",  # must modify NaNs in perturbation column
    "ECCITE": "NT",
    "augur_ex": "Maintenance_Cocaine"
}

key_treatment_data = {
    "CRISPRi_scr": "KD",
    "CRISPRi_scr_multi": "KD",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": None,
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "KO",
    "perturb-seq": "KO",
    "coda": "Salmonella",
    "augur_ex": "Withdraw_48h_Cocaine"
}

# key_nonperturbed_data = {
#     "CRISPRi_scr": "NP",
#     "CRISPRi_wgs": np.nan,
#     "CRISPRi_ess": np.nan,
#     "pool": np.nan,
#     "bulk": np.nan,
#     "screen": np.nan,
#     "ECCITE": "KO",
#     "perturb-seq": np.nan,
#     "coda": np.nan,
#     "augur_ex": np.nan
# }

col_condition_data = {
    "CRISPRi_scr": "target_gene_name",
    "CRISPRi_scr_multi": "target_gene_name",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "target_gene",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "perturb-seq": "target_gene_name",
    "ECCITE": "gene_target",
    "coda": "condition",
    "augur_ex": "label"
}

col_guide_rna_data = {
    "CRISPRi_scr": "feature_call",
    "CRISPRi_scr_multi": "feature_call",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "guide_ids",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "perturb-seq": "perturbation",
    "ECCITE": "guide_ID",
    "coda": np.nan,
    "augur_ex": None
}

col_num_umis_data = {
    "CRISPRi_scr": "num_umis",
    "CRISPRi_scr_multi": "num_umis",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "UMI count",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "perturb-seq": None,
    "ECCITE": "guide_ID",
    "coda": np.nan,
    "augur_ex": None
}

# layer_perturbation_data = {
#     "CRISPRi_scr": np.nan,
#     "CRISPRi_wgs": np.nan,
#     "CRISPRi_ess": "X_pert",
#     "pool": np.nan,
#     "bulk": np.nan,
#     "screen": np.nan,
#     "perturb-seq": None,
#     "ECCITE": "X_pert",
#     "coda": None, 
#     "augur_ex": None
# }

col_sample_id_data = {
    "CRISPRi_scr": None,
    "CRISPRi_scr_multi": ("orig.ident", None),
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "gemgroup",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "perturb-seq": None,
    "ECCITE": "orig.ident",
    "coda": "batch", 
    "augur_ex": "orig.ident"
}

col_batch_data = {
    "CRISPRi_scr": None,
    "CRISPRi_scr_multi": "orig.ident",
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "gemgroup",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "perturb-seq": None,
    "ECCITE": "MULTI_ID",
    "coda": "batch", 
    "augur_ex": None
}

kws_process_guide_rna_data = {
    "CRISPRi_scr": dict(feature_split="|", guide_split="-", 
                        key_control_patterns=["CTRL"], 
                        remove_multi_transfected=True,
                        min_n_target_control_drop=None, 
                        max_pct_control_drop=75, 
                        min_pct_dominant=80, min_pct_avg_n=40),
    "CRISPRi_scr_multi": dict(feature_split="|", guide_split="-", 
                              key_control_patterns=["CTRL"], 
                              remove_multi_transfected=True,
                              min_n_target_control_drop=None, 
                              max_pct_control_drop=75, 
                              min_pct_dominant=80, min_pct_avg_n=40),
    "CRISPRi_wgs": None,
    "CRISPRi_ess": dict(feature_split=",", guide_split="-", 
                        key_control_patterns=["CTRL"]),
    "pool": None,
    "bulk": None,
    "screen": None,
    "perturb-seq": dict(feature_split=None, guide_split="_",
                        key_control_patterns=[np.nan]),
    "ECCITE": None,
    "coda": None, 
    "augur_ex": None
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
    if adata is None:  # if file doesn't exist or failed to load
        if file in files_data:
            print(f"\n\nLooking for downloadable files for: {file}.")
            if file == "CRISPRi_wgs":  # CRISPRi Perturb-seq Pertpy data
                adata = pertpy.data.replogle_2022_k562_gwps()  # HJ design
                # adata = pertpy.data.replogle_2022_k562_essential()  # ~1 hr. 
                # adata = pertpy.data.replogle_2022_rpe1()
                # adata = pertpy.data.adamson_2016_upr_perturb_seq()  # ~8 min.
            elif file == "CRISPRi_ess":
                print(f"Setting {col_perturbed_data}")
                adata = pertpy.data.replogle_2022_k562_essential()  # HJ design
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
    if file == "CRISPRi_ess":
        adata = adata[adata.obs["guide_ids"].isin(
            ["NT", "CDKN1A", "CDKN1A,CDKN1B", "CEBPA", "CEBPB", 
            "CEBPA,CEBPB", "CEBPB,OSR2", "S1PR2,SGK1",
            "DUSP9,KLF1", "SAMD1,UBASH3B", "TGFBR2", 
            "FEV,ISL2", "PRTG,TGFBR2",
            "JUN", "CLDN6,KLF1", "CBFA2T3,POU3F2", "FOXA1,HOXB9",
            "DLX2,ZBTB10", "SAMD1,TGFBR2", "ZBTB10", "CEBPE,SPI1", "PTPN13",
            "CEBPE,PTPN12", "CDKN1B,CDKN1C", "FOXF1,FOXL2", "AHR,FEV",
            "KLF1,TGFBR2", "CDKN1A,CDKN1B"])]  # subset for speed
    if file == "coda": 
        adata.var.loc[:, col_gene_symbols_data[file]] = adata.var.index.values
    return adata
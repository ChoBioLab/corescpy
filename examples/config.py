import pertpy
import os
import pathlib
import crispr as cr
import numpy as np


# Detect current directory of file
DIR = pathlib.Path(__file__).parent.resolve()
DIR = os.path.join(DIR, "data")

files_data = {
    "CRISPRi_scr": f"{DIR}/filtered_feature_bc_matrix_HH03",
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
label_cell_type_data = {
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
gene_symbols_data = {
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
label_perturbation_data = {
    "CRISPRi_scr": np.nan,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "guide_identity",
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "perturbation",
    "coda": "condition",
    "augur_ex": "label"
    }
key_control_data = {
    "CRISPRi_scr": np.nan,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": "NegCtrl0_NegCtrl0__NegCtrl0_NegCtrl0", 
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "coda": "Control",
    "ECCITE": "NT",
    "augur_ex": "Maintenance_Cocaine"
    }
perturbation_type_data = {
    "CRISPRi_scr": np.nan,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": np.nan,
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "KO",
    "coda": "Salmonella",
    "augur_ex": "Withdraw_48h_Cocaine"
    }
target_genes_data = {
    "CRISPRi_scr": np.nan,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": np.nan,
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "gene_target",
    "coda": np.nan,
    "augur_ex": np.nan
    }
layer_perturbation_data = {
    "CRISPRi_scr": np.nan,
    "CRISPRi_wgs": np.nan,
    "CRISPRi_ess": np.nan,
    "pool": np.nan,
    "bulk": np.nan,
    "screen": np.nan,
    "ECCITE": "gene_target",
    "coda": None, 
    "augur_ex": None,
    }


def load_example_data(file, col_gene_symbol, write_public=False):
    """(Down)load data for examples/vignettes.
    Args:
        file (str): Name of data (see keys of dictionaries in config).
        col_gene_symbol (str): Name of column in `AnnData.obs` 
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
    print(file_path)
    if os.path.exists(file_path):
        print(f"\n\n{file_path} exists.")
        try:  # try to create scanpy object from file
            adata = cr.pp.create_object(file_path, assay=None, 
                                        col_gene_symbol=col_gene_symbol)
        except Exception as e:
            err = f"\n\n{'=' * 80}\n\n{file_path} failed to load:\n\n{e}"
            print(err)
    if adata is None:  # if file doesn't exist or failed to load
        if file in files_data:
            print(f"\n\nLooking for downloadable files for: {file}.")
            if file == "CRISPRi_wgs":  # CRISPRi Perturb-seq Pertpy data
                adata = pertpy.data.replogle_2022_k562_gwps()  # HJ design
                # adata = pertpy.data.replogle_2022_k562_essential()  # ~1 hr.
                # adata = pertpy.data.replogle_2022_rpe1()
                # adata = pertpy.data.adamson_2016_upr_perturb_seq()  # ~8 min.
            elif file == "CRISPRi_ess":
                pertpy.data.replogle_2022_k562_essential()  # HJ design
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
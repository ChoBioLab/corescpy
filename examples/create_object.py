import crispr
import crispr.processing
import crispr.processing.preprocessing as pp
import os

# File Paths
DIR_PRJ = "/home/asline01/projects/crispr-screening"
DIR_CRC = "/analysis/cellranger/cr_count_2023-05-09_0828"
SUBJ = "HH03"
file = os.path.join(DIR_PRJ, DIR_CRC, f"/{SUBJ}/outs/filtered_feature_bc_matrix")

# Create Object
adata = pp.create_object_scanpy(file)  # create scanpy object
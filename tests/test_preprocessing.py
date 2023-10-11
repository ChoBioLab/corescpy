import crispr as cr
import numpy as np


class GuideRNATests:
    file = "adamson_2016_upr_perturb_seq.h5ad"
    kwargs_init = {
        "assay": None, "assay_protein": None, "col_cell_type": "celltype",
        "col_gene_symbols": "gene_symbol", "col_sample_id": None, 
        "col_batch": None, "col_perturbation": "perturbation", 
        "col_guide_rna": None, "col_num_umis": None, 
        "kws_process_guide_rna": None, 
        "col_target_genes": "perturbation", "key_control": "Control", 
        "key_treatment": "63(mod)_pBA580", "remove_multi_transfected": True}
    cro = cr.pp.create_object(file, **kwargs_init)
    out = cr.pp.filter_by_guide_counts(
        cro.adata, "feature_call", "num_umis", 
        max_percent_umis_control_drop=75,
        min_percent_umis=40, feature_split="|", guide_split="-",
        key_control_patterns=["CTRL"], key_control="Control")

    def test_output_processing_guides(self):
        """Check that output format conforms to Crispr method expectations.""" 
        col_guide_rna = self.kwargs_init["col_guide_rna"]
        assert(isinstance(self.out, (tuple, set, list, np.ndarray)))
        assert(len(self.out) == 2)
        tg_info, feats_n = self.out
        assert(f"{col_guide_rna}_list_filtered" in tg_info.columns)
        assert(tg_info[f"{col_guide_rna}_list_filtered"].apply(
            lambda x: isinstance(x, list)).all())  # all are lists
        assert(self.cro._columns_created["guide_percent"] in feats_n.columns)
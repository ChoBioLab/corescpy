import crispr as cr
import numpy as np


class GuideRNATests:
    file = "data/adamson_2016_pilot.h5ad"
    kwargs_init = {
        "assay": None, "assay_protein": None, "col_cell_type": "celltype",
        "col_gene_symbols": "gene_symbol", "col_sample_id": None, 
        "col_batch": None, "col_perturbation": "perturbation", 
        "col_guide_rna": "perturbation", "col_num_umis": None, 
        "kws_process_guide_rna": dict(
            feature_split="", guide_split="_", key_control_patterns=[np.nan]), 
        "col_target_genes": "perturbation", "key_control": "Control", 
        "key_treatment": "63(mod)_pBA580", "remove_multi_transfected": True}
    
    for x in kwargs_init["kws_process_guide_rna"]:
        print(f"{x}='{[x]}'" if isinstance(kws[x], str) else f"{x}={kws[x]}")


    adata = cr.pp.create_object(file, **kwargs_init)
    
    out = cr.pp.filter_by_guide_counts(
        adata, kwargs_init["col_guide_rna"], kwargs_init["col_num_umis"],
        max_percent_umis_control_drop=75,
        min_percent_umis=40, **kwargs_init["kws_process_guide_rna"], 
        key_control="Control")

    def test_output_processing_guides(self):
        """Check that output format conforms to Crispr method expectations.""" 
        col_guide_rna = self.kwargs_init["col_guide_rna"]
        col_gp = cr.Crispr._columns_created["guide_percent"]
        assert(isinstance(self.out, (tuple, set, list, np.ndarray)))
        assert(len(self.out) == 2)
        tg_info, feats_n = self.out
        assert(f"{col_guide_rna}_list_filtered" in tg_info.columns)
        assert(tg_info[f"{col_guide_rna}_list_filtered"].apply(
            lambda x: isinstance(x, list)).all())  # all are lists
        assert(col_gp in feats_n.columns), f"""
        {col_gp} not in `cr.pp.filter_by_guide_counts` 
        as expected by Crispr class methods.
        Columns in 2nd element of output:\n{feats_n.columns}"""
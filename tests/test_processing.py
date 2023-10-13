import crispr as cr
import pertpy as pt
import numpy as np    

class GuideRNATests:
    kwargs = {
        "assay": None, "assay_protein": None, "col_cell_type": "celltype",
        "col_gene_symbols": "gene_symbol", "col_sample_id": None, 
        "col_batch": None, "col_perturbation": "perturbation", 
        "col_guide_rna": "perturbation", "col_num_umis": "UMI count", 
        "kws_process_guide_rna": dict(
            feature_split=None, guide_split="_", key_control_patterns=[np.nan]), 
        "col_target_genes": "perturbation", "key_control": "Control", 
        "key_treatment": "KO", "remove_multi_transfected": True}
    # adata = cr.pp.create_object("data/adamson_2016_pilot.h5ad", **kwargs)
    adata = pt.dt.adamson_2016_upr_perturb_seq()
    out = cr.pp.filter_by_guide_counts(
        adata, kwargs["col_guide_rna"], kwargs["col_num_umis"],
        max_percent_umis_control_drop=75,
        min_percent_umis=40, **kwargs["kws_process_guide_rna"], 
        key_control="Control")

    def test_output_processing_guides(self):
        """Check that output format conforms to Crispr method expectations.""" 
        col_guide_rna = GuideRNATests.kwargs["col_guide_rna"]
        col_gp = cr.Crispr._columns_created["guide_percent"]
        assert(isinstance(GuideRNATests.out, (tuple, set, list, np.ndarray)))
        assert(len(GuideRNATests.out) == 2)
        tg_info, feats_n = GuideRNATests.out
        assert(f"{col_guide_rna}_list_filtered" in tg_info.columns)
        assert(tg_info[f"{col_guide_rna}_list_filtered"].apply(
            lambda x: isinstance(x, list)).all())  # all are lists
        assert(col_gp in feats_n.columns), f"""
        {col_gp} not in `cr.pp.filter_by_guide_counts` 
        as expected by Crispr class methods.
        Columns in 2nd element of output:\n{feats_n.columns}"""
        
        
# class Preprocessing:
    
#     def test_filtering(adata):
#         self.preprocess(adata)
#         if assay is None:
#             adata = adata[adata.obs.n_genes_by_counts < max_genes_by_counts, 
#                           :]
#             adata = adata[adata.obs.pct_counts_mt < max_pct_mt, :]
#         else:
#             adata[assay] = adata[assay][
#                 adata[assay].obs.n_genes_by_counts < max_genes_by_counts, :]
#             adata[assay] = adata[assay][
#                 adata[assay].obs.pct_counts_mt < max_pct_mt, :]  # MT counts
#             adata[assay].raw = adata[assay
#                                      ]  # freeze normalized, filtered data
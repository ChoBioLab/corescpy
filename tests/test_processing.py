import crispr as cr
import pertpy as pt
import numpy as np    

                
class CrisprObject:
    """Object creation tests."""
    
    kwargs = {
        "assay": None, "assay_protein": None, "col_cell_type": "celltype",
        "col_gene_symbols": "gene_symbol", "col_sample_id": None, 
        "col_perturbation": "perturbation", 
        "col_guide_rna": "perturbation", "col_num_umis": "UMI count", 
        "kws_process_guide_rna": dict(
            feature_split=None, guide_split="_", key_control_patterns=[np.nan], 
            remove_multi_transfected=True), 
        "col_target_genes": "perturbation", "key_control": "Control", 
        "key_treatment": "KO"}
    adata = pt.dt.adamson_2016_upr_perturb_seq()
    self = cr.Crispr(adata, **kwargs)
    
    def test_attribute_presence(self):
        """Ensure expected attributes are present."""
        assert "adata" in dir(CrisprObject.self)
        assert "rna" in dir(CrisprObject.self)
        assert "uns" in dir(CrisprObject.self)
        assert "obs" in dir(CrisprObject.self)
        assert "_columns" in dir(CrisprObject.self)
        assert "_layers" in dir(CrisprObject.self)
        assert "_keys" in dir(CrisprObject.self)
    
    
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
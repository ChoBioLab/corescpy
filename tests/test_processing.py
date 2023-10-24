import crispr as cr
import pertpy as pt
import numpy as np    

class GuideRNATests:
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
        
    # def test_name_processing(self):
        # cgrna = pd.read_excel("data/process_guide_rna/multi.xlsx", 
        #                       index_col=0)
        # feature_split, guide_split = [str(cgrna.iloc[0][x]) for x in [
        #     "feature_split", "guide_split"]]
        # key_control_patterns = cgrna["key_control_patterns"].iloc[0]
        # print(key_control_patterns)
        # col_guide_rna = str(cgrna.iloc[0]["col_guide_rna"])
        # col_guide_rna_series = cgrna["col_guide_rna"]
        # targets, grnas = cr.pp.detect_guide_targets(col_guide_rna_series, 
        #                         feature_split=feature_split,
        #                         guide_split=guide_split,
        #                         key_control_patterns=key_control_patterns,
        #                         key_control="Control")
                
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
    # adata = cr.pp.create_object("data/adamson_2016_pilot.h5ad", **kwargs)
    adata = pt.dt.adamson_2016_upr_perturb_seq()
    # out = cr.pp.filter_by_guide_counts(
    #     adata, kwargs["col_guide_rna"], kwargs["col_num_umis"],
    #     max_percent_umis_control_drop=75,
    #     min_percent_umis=40, **kwargs["kws_process_guide_rna"], 
    #     key_control="Control")
    self = cr.Crispr(adata, **kwargs)
    
	def test_attribute_presence:
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
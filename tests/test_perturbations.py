from crispr.class_sc import Omics
import pertpy as pt

                
class TestDistance:
    """Object creation tests."""

    kwargs_init = dict(col_gene_symbols="gene_symbol",  
                       col_cell_type="leiden", 
                       col_perturbed="perturbed", 
                       col_guide_rna="guide_id", 
                       col_condition="target", 
                       key_control="control", 
                       key_treatment="KO")
    try:
        adata = pt.dt.distance_example()
    except Exception:
        adata = pt.dt.distance_example_data()
    self = 
    
    def test_distance_metrics(self):
        """Ensure expected attributes are present."""
        oca = TestDistance.self.run_composition_analysis(
            est_fdr=0.05, generate_sample_level=True)
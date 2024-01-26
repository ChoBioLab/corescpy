import crispr as cr
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
    adata.layers["log1p"] = adata.X.copy()  # already log
    self = cr.Crispr(adata, **kwargs_init)
    self.cluster(kws_neighbors=dict(n_neighbors=30), use_rep="X_pca", 
                 n_comps=30, resolution=0.5, layer="log1p")
    
    def test_distance_metrics(self):
        """Ensure expected attributes are present."""
        outs = {}
        for x in ["mmd", "edistance"]:
            print(x)
            outs[x] = TestDistance.self.compute_distance(
                x, method="X_pca", n_perms=100, 
                alpha=0.0015, kws_plot=dict(robust=False, figsize=(10, 10)))
        
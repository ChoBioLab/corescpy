import crispr as cr
import scipy
import pertpy as pt
import numpy as np


class TestDistance:
    """Object creation tests."""

    kwargs_init = dict(col_gene_symbols="gene_symbol",
                       col_cell_type="leiden",
                       col_perturbed="perturbed",
                       col_guide_rna="grna_lenient",
                       col_condition="target",
                       key_control="control",
                       key_treatment="KO",
                       kws_process_guide_rna=False)
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
            print(outs[x][-2])

    def test_mixscape(self):
        """Ensure expected attributes are present."""
        self.run_mixscape()

    def test_augur(self):
        """Ensure expected attributes are present."""
        self.run_augur()


class TestAdamson:
    """Adamson 2016 dataset."""
    kws_pga = dict(feature_split=None, guide_split="_",
                   key_control_patterns=["CTRL"],
                   remove_multi_transfected=True)
    kwargs = dict(col_gene_symbols="gene_symbol", col_cell_type="leiden",
                  col_sample_id=None, col_batch=None, col_subject=None,
                  col_condition="target_gene", col_num_umis="UMI count",
                  col_perturbed="perturbed", col_guide_rna="perturbation",
                  key_control="Control", key_treatment="KO",
                  kws_process_guide_rna=kws_pga)
    adata = pt.dt.adamson_2016_upr_perturb_seq()
    adata.obs[adata.obs.perturbation == "*", "perturbation"] = "CTRL"
    self = cr.Crispr(adata, **kwargs)

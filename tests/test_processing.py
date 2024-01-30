import crispr as cr
import scanpy as sc
import pertpy as pt
import scipy
import numpy as np

class TestOmics:
    """Object creation tests."""
    kws_pga = dict(feature_split=None, guide_split="_",
                   key_control_patterns=[np.nan],
                   remove_multi_transfected=True)
    kwargs = dict(col_gene_symbols="gene_symbol", col_cell_type="leiden",
                  col_sample_id=None, col_batch=None, col_subject=None,
                  col_condition="target_gene", col_num_umis="UMI count",
                  col_perturbed="perturbed", col_guide_rna="perturbation",
                  key_control="Control", key_treatment="KO")
    adata = sc.datasets.pbmc68k_reduced()
    self = cr.Crispr(adata, **kwargs, kws_process_guide_rna=kws_pga)

    def test_attribute_presence(self):
        """Ensure expected attributes are present."""
        assert "adata" in dir(TestOmics.self)
        assert "rna" in dir(TestOmics.self)
        assert "uns" in dir(TestOmics.self)
        assert "obs" in dir(TestOmics.self)
        assert "_columns" in dir(TestOmics.self)
        assert "_layers" in dir(TestOmics.self)
        assert "_keys" in dir(TestOmics.self)
        assert ".layers" in dir(TestOmics.self)

    def test_var(self):
        """Ensure `.var` meets expectations."""
        var_ixn = TestOmics.self.obs.index.names[0]
        assert var_ixn == TestOmics.self._columns["col_gene_symbols"]

    def test_obs(self):
        """Ensure `.obs` meets expectations."""
        pass



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

import corescpy as cr
import scipy
import pertpy as pt
import numpy as np


class TestCiteSeq:
    """Test CITE-seq with guide RNA data in separate modality."""
    col_guide_rna, col_num_umis, col_condition = "guide", "num_umis", "target"
    col_condition = "target_gene"
    feature_split, guide_split = "|", "g"
    key_control = "Control"
    kws_pg = dict(feature_split=feature_split, guide_split=guide_split,
                  key_control_patterns=["NT"], remove_multi_transfected=True,
                  max_pct_control_drop=None, min_pct_avg_n=None,
                  min_n_target_control_drop=None, min_pct_dominant=51)
    kwargs = dict(assay="rna", assay_gdo="gdo", assay_protein="adt",
                  col_batch="orig.ident", col_subject_id="replicate",
                  col_sample_id="MULTI_ID", col_condition=col_condition,
                  col_num_umis=col_num_umis, col_perturbed="perturbed",
                  col_cell_type="leiden", col_guide_rna=col_guide_rna,
                  key_control=key_control, key_treatment="KO")
    adata = pt.data.papalexi_2021()
    adata.mod["gdo"].X = scipy.sparse.csr_matrix(adata.mod["gdo"].X.A - 1)
    self = cr.Crispr(adata, **kwargs, kws_process_guide_rna=kws_pg)
    _ = self.preprocess()
    _ = self.cluster()

    def test_celltypist():
        _ = TestCiteSeq.self.annotate_clusters("Immune_All_Low.pkl")

    def test_guide_assign(tol=2):
        """See if guide assignment roughly matches author's."""
        guides = TestCiteSeq.self.rna.obs[[TestCiteSeq.self._columns[
            "col_condition"], "gene_target"]].copy()
        guides.columns = ["us", "them"]
        print(guides[guides.us != guides.them])
        assert np.mean(guides.us != guides.them) * 100 < tol  # < tol %

    def test_distance_metrics(self, method="edistance"):
        """Ensure expected attributes are present."""
        out = TestCiteSeq.self.compute_distance(
            "edistance", n_jobs=4, alpha=0.00015,
            kws_plot=dict(robust=False, figsize=(10, 10)))
        print(out)

    def test_mixscape(self):
        """Ensure expected attributes are present."""
        out_mix = TestCiteSeq.self.run_mixscape()

    def test_augur(self):
        """Ensure expected attributes are present."""
        out_aug = TestCiteSeq.self.run_augur()


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

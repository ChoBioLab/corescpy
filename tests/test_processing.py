import squidpy as sq
import scanpy as sc
import numpy as np
import corescpy as cr


class TestOmics:
    """Object creation & basic processing tests."""
    adata = sc.datasets.pbmc68k_reduced()
    self = cr.Omics(adata, col_cell_type="bulk_labels")

    def test_attribute_presence(self):
        """Ensure expected attributes are present."""
        assert "adata" in dir(TestOmics.self)
        assert "rna" in dir(TestOmics.self)
        assert "uns" in dir(TestOmics.self)
        assert "obs" in dir(TestOmics.self)
        assert "_columns" in dir(TestOmics.self)
        assert "_layers" in dir(TestOmics.self)
        assert "_keys" in dir(TestOmics.self)
        assert ".layers" in dir(TestOmics.self.adata)

    def test_var(self):
        """Ensure `.var` meets expectations."""
        var_ixn = TestOmics.self.rna.obs.index.names[0]
        assert var_ixn == TestOmics.self._columns["col_gene_symbols"]
        assert self.rna.var.sort_index().index.values[0] == "ABHD17A"

    def test_obs(self):
        """Ensure `.obs` meets expectations."""
        pass

    def test_processing(self):
        """Test processing."""
        _ = TestOmics.self.preprocess()

    def test_clustering(self):
        """Test clustering."""
        _ = TestOmics.self.cluster()

    def test_plotting(self):
        """Test plotting."""
        _ = TestOmics.self.plot()

    def test_rna(self):
        """Test that `.rna reflects adata changes, & vice-versa."""
        TestOmics.self.adata.obs.loc[:, "adata_change"] = np.random.randn(
            len(TestOmics.self.adata.obs))
        np.testing.assert_array_equal(
            np.array(TestOmics.self.adata.obs["adata_change"]),
            np.array(TestOmics.self.rna.obs["adata_change"]))
        TestOmics.self.rna.obs.loc[:, "rna_change"] = np.random.randn(
            len(TestOmics.self.rna.obs))
        np.testing.assert_array_equal(
            np.array(TestOmics.self.adata.obs["rna_change"]),
            np.array(TestOmics.self.rna.obs["rna_change"]))


class TestSpatialVisium:
    """Object creation & basic processing tests."""
    library_id = "V1_Human_Brain_Section_2"
    adata = sq.datasets.visium(library_id, include_hires_tiff=False)
    kwargs = dict(col_gene_symbols="gene_symbol", col_cell_type="leiden",
                  col_sample_id="Sample", col_batch=None, col_subject=None,
                  visium=True, library_id=library_id)
    self = cr.Spatial(adata, **kwargs)

    def test_attribute_presence(self):
        """Ensure expected attributes are present."""
        assert "adata" in dir(TestSpatialVisium.self)
        assert "rna" in dir(TestSpatialVisium.self)
        assert "uns" in dir(TestSpatialVisium.self)
        assert "obs" in dir(TestSpatialVisium.self)
        assert "_columns" in dir(TestSpatialVisium.self)
        assert "_layers" in dir(TestSpatialVisium.self)
        assert "_keys" in dir(TestSpatialVisium.self)
        assert ".layers" in dir(TestSpatialVisium.self.adata)

    def test_var(self):
        """Ensure `.var` meets expectations."""
        var_ixn = TestSpatialVisium.self.rna.var.index.names[0]
        assert var_ixn == TestSpatialVisium.self._columns["col_gene_symbols"]
        assert TestSpatialVisium.self.rna.var.sort_index(
            ).index.values[0] == "ABHD17A"

    def test_obs(self):
        """Ensure `.obs` meets expectations."""
        pass

    def test_processing(self):
        """Test processing."""
        _ = TestSpatialVisium.self.preprocess()

    def test_clustering(self):
        """Test clustering."""
        _ = TestSpatialVisium.self.cluster()

    def test_plotting(self):
        """Test plotting."""
        _ = TestSpatialVisium.self.plot()

    def test_plotting_spatial(self):
        """Test plotting."""
        fig_cluster = TestSpatialVisium.self.plot_spatial()
        fig_gex = TestSpatialVisium.self.plot_spatial(color=list(
            TestSpatialVisium.self.rna.var.index.names[:6]))
        assert not isinstance(fig_cluster, (Exception, str))
        assert not isinstance(fig_gex, (Exception, str))

    def test_rna(self):
        """Test that `.rna reflects adata changes, & vice-versa."""
        TestSpatialVisium.self.adata.obs.loc[:, "adata_change"] = np.arange(
            1, TestSpatialVisium.self.adata.obs.shape[0])
        np.testing.assert_array_equal(*[np.array(x.obs[
            "adata_change"]) for x in [TestSpatialVisium.self.adata,
                                       TestSpatialVisium.self.rna]])
        TestSpatialVisium.self.rna.obs.loc[:, "rna_change"] = np.arange(
            1, TestSpatialVisium.self.rna.obs.shape[0])
        np.testing.assert_array_equal(*[np.array(x.obs[
            "rna_change"]) for x in [TestSpatialVisium.self.adata,
                                     TestSpatialVisium.self.rna]])

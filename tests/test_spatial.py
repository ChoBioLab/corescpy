import os
import squidpy as sq
import corescpy as cr

DIR_TEST_FILE = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(DIR_TEST_FILE, "data")


# class TestXenium:
#     """Visium data tests."""
#     url = "https://s3.embl.de/spatialdata/spatialdata-sandbox/"
#     file = "xenium_rep1_io"
#     if os.path.exists(os.path.join(DIR_DATA, file)) is False:
#         os.system(f"wget -P {DIR_DATA} {url}{file}.zip")
#         os.system(f"unzip {file}.zip -d {os.path.join(DIR_DATA, file)}")
#     self = cr.Spatial(os.path.join(DIR_DATA, file))

#     def test_xenium_attributes(self):
#         """Ensure proper attributes."""
#         assert "adata" in dir(TestXenium.self)
#         assert "rna" in dir(TestXenium.self)
#         assert "original_ix" in dir(TestXenium.self.rna.uns)

#     def test_xenium_ax(self):
#         """Test Xenium analysis."""
#         genes = list(TestXenium.self.rna.var_names[:3])
#         _ = TestXenium.self.preprocess()
#         _ = TestXenium.self.cluster()
#         out_ce = TestXenium.self.calculate_centrality()
#         out_co = TestXenium.self.find_cooccurrence()
#         out_sv = TestXenium.self.find_svgs(genes=genes, n_perms=5)
#         out_rl = TestXenium.self.calculate_receptor_ligand(
#             col_condition=False, p_threshold=0.001, remove_ns=True)
#         for x in [out_ce, out_co, out_sv, out_rl]:
#             if isinstance(x[-1], dict):  # if figure output is a dictionary
#                 for i in x[-1]:
#                     assert not isinstance(x[-1][i], str)  # ensure not error
#             else:
#                 assert not isinstance(x[-1], str)  # ensure not error


class TestVisium:
    """Visium data tests."""
    library_id = "V1_Human_Brain_Section_2"
    adata = sq.datasets.visium(library_id, include_hires_tiff=False)
    kwargs = dict(col_gene_symbols="gene_symbol", col_cell_type="leiden",
                  col_sample_id="Sample", col_batch=None, col_subject=None,
                  visium=True, library_id=library_id)
    self = cr.Spatial(adata, **kwargs)

    def test_visium_attributes(self):
        """Ensure proper attributes."""
        assert "adata" in dir(TestVisium.self)
        assert "rna" in dir(TestVisium.self)

    def test_visium_ax(self):
        """Test Visium analysis."""
        genes = list(TestVisium.self.rna.var_names[:3])
        _ = TestVisium.self.preprocess()
        _ = TestVisium.self.cluster()
        out_ce = TestVisium.self.calculate_centrality()
        out_co = TestVisium.self.find_cooccurrence()
        out_sv = TestVisium.self.find_svgs(genes=genes, method="moran",
                                           n_perms=10)
        out_rl = TestVisium.self.calculate_receptor_ligand(
            col_condition=False, p_threshold=0.001, remove_ns=True)
        for x in [out_ce, out_co, out_sv, out_rl]:
            if isinstance(x[-1], dict):  # if figure output is a dictionary
                for i in x[-1]:
                    assert not isinstance(x[-1][i], str)  # ensure not error
            else:
                assert not isinstance(x[-1], str)  # ensure not error

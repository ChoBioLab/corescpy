import crispr as cr
import scanpy as sc
import squidpy as sq
import scipy
import numpy as np


class TestVisium:
    """Visium data tests."""
    kws_pga = dict(feature_split=None, guide_split="_",
                   key_control_patterns=[np.nan],
                   remove_multi_transfected=True)
    library_id = "V1_Human_Brain_Section_2"
    adata = sq.datasets.visium(library_id, include_hires_tiff=False)
    kwargs = dict(col_gene_symbols="gene_symbol", col_cell_type="leiden",
                  col_sample_id=None, col_batch=None,
                  col_subject="", visium=True, library_id=library_id)
    self = cr.Spatial(adata, **kwargs, kws_process_guide_rna=kws_pga)

    def test_visium_plots(self):
        """Basic plots."""
        assert "adata" in dir(TestVisium.self)
        assert "rna" in dir(TestVisium.self)
        sq.spatial_segment(TestVisium.self.rna)
        sq.spatial_scatter(TestVisium.self.rna)

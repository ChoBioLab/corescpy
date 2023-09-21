#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""


import pandas as pd
import numpy as np
import scanpy as sc
import crispr as cr

class Crispr(object):
    """An object class for CRISPR analysis and visualization"""

    def __init__(self, file_path, 
                 assay=None, assay_protein=None,
                 layer_perturbation=None, label_perturbation_type="Perturbed",
                 col_gene_symbols="gene_symbols", col_cell_type="leiden",
                 col_sample_id="standard_sample_id", 
                 col_perturbation="perturbation",
                 col_target_genes="guide_ids",
                 col_guide_rna="guide_ids",
                 key_control="control", key_treatment="NT", **kwargs):
        """CRISPR class initialization."""
        self._assay = assay
        self._assay_protein = assay_protein
        self._layer_perturbation = layer_perturbation
        self._label_perturbation_type = label_perturbation_type
        self._columns = dict(col_gene_symbols=col_gene_symbols,
                             col_cell_type=col_cell_type, 
                             col_sample_id=col_sample_id, 
                             col_perturbation=col_perturbation,
                             col_guide_rna=col_guide_rna,
                             col_target_genes=col_target_genes)
        self._keys = dict(key_control=key_control, key_treatment=key_treatment)
        self._file_path = file_path
        self.adata = file_path
        self.figures = {}

    @property
    def adata(self, value):
        """Specify file path to data."""
        return self._adata

    @adata.setter
    def adata(self, value):
        """Set file path and load object."""
        self._file_path = value
        if not isinstance(value, sc.AnnData):
            self._adata = cr.pp.create_object(
                value, assay=None, col_gene_symbols=self._columns[
                    "col_gene_symbols"])
    
    def preprocess(self, assay=None, assay_protein=None, 
                   clustering=False, run_label="main", **kwargs):
        """Preprocess data."""
        if assay is None:
            assay = self._assay
        if assay_protein is None:
            assay_protein = self._assay_protein
        cr.pp.process_data(self.adata, assay=assay, 
                           assay_protein=assay_protein, **kwargs)  # preprocess
        if clustering not in [None, False]:
            if clustering is True:
                clustering = {}
            if isinstance(clustering, dict):
                figs_cl = cr.ax.clustering.cluster(
                    self.adata, assay=assay, **clustering)  # clustering
                self.figures[run_label].update({"clustering": figs_cl})
            else:
                raise TypeError(
                    "'clustering' must be dict (keyword arguments) or bool.")
            
                    
    def cluster(self, assay=None, method_cluster="leiden", 
                plot=True, colors=None, paga=False, 
                kws_pca=None, kws_neighbors=None, 
                kws_umap=None, kws_cluster=None, 
                run_label="main", test=False):
        """Perform dimensionality reduction and create UMAP."""
        if assay is None:
            assay = self._assay
        figs_cl = cr.ax.clustering(
            self.adata.copy() if copy is True else self.adata, 
            assay=None, method_cluster=method_cluster,
            plot=plot, colors=colors, paga=paga, 
            kws_pca=kws_pca, kws_neighbors=kws_neighbors,
            kws_umap=kws_umap, kws_cluster=kws_umap)
        if test is False:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"clustering": figs_cl})
        return figs_cl
        
        
    def run_mixscape(self, target_gene_idents=True, 
                     min_de_genes=5, col_split_by=None,
                     label_perturbation_type=None, run_label="main", 
                     test=False, plot=True):
        """Run Mixscape.""" 
        if assay is None:
            assay = self._assay
        if label_perturbation_type is None:
            label_perturbation_type=self._label_perturbation_type
        figs_mix = cr.ax.perform_mixscape(
            self.adata.copy() if test is True else self.adata, 
            self._columns["col_perturbation"], assay=assay, 
            key_control=self._keys["key_control"],
            label_perturbation_type=self._label_perturbation_type, 
            layer_perturbation=self._layer_perturbation, 
            target_gene_idents=target_gene_idents,
            col_target_genes=self._columns["col_target_genes"], 
            min_de_genes=min_de_genes, col_split_by=col_split_by, plot=plot)
        if test is False:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"mixscape": figs_mix})
        return figs_mix
    
    def run_augur(self, assay=None, 
                  augur_mode="default", classifier="random_forest_classifier", 
                  subsample_size=20, n_threads=None, 
                  select_variance_features=False, 
                  seed=1618, plot=True, run_label="main", test=False):
        """Run Augur."""
        if test is True:
            annd = self.adata.copy()
        data, results, figs_aug = cr.ax.perform_augur(
            self.adata.copy() if test is True else self.adata, 
            assay=assay, classifier=classifier, 
            augur_mode=augur_mode, subsample_size=subsample_size,
            select_variance_features=select_variance_features, 
            col_cell_type=self._columns["col_cell_type"], 
            key_control=self._keys["key_control"],
            key_treatment=self._keys["key_treatment"],
            col_perturbation=self._columns["col_perturbation"], 
            seed=seed, n_threads=n_threads, plot=plot)
        if test is False:
            if run_label not in self.results:
                self.results[run_label] = {}
            self.results[run_label].update(
                {"Augur": {"results": results, "data": data}})
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"Augur": figs_aug})
        return data, results, figs_aug
    
    def save_output(self, directory_path, run_keys="all", overwrite=False):
        """Save figures, results, adata object."""
        # TODO: FINISH
        raise NotImplementedError("Saving output isn't yet fully implemented.")
        if isinstance(run_keys, (str, float)):
            run_keys = [run_keys]
        elif run_keys == "all":
            run_keys = list(self.figures.keys())
            run_keys += list(self.results.keys())
            run_keys = list(pd.unique(self.results))
        for r in run_keys:
            os.makedirs(os.path.join(directory_path, r), exist_ok=True)
            if r in self.figures:
                for f in self.figures[r]:
                    dirp = os.path.join(directory_path, r, f)
                    os.makedirs(dirp, exist_ok=overwrite)
                    # self.figures[r][f].savefig(
                    #     os.path.join(dirp, f"{f}.pdf"))
            if r in self.results:
                for f in self.results[r]:
                    dirp = os.path.join(directory_path, r)
                    if not os.path.exists(dirp)
                    os.makedirs(dirp, exist_ok=overwrite)
            
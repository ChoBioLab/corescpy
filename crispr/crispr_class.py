#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=line-too-long, invalid-name
"""
@author: E. N. Aslinger
"""

import scanpy as sc
# import subprocess
import os
import warnings
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import crispr as cr


class Crispr(object):
    """An object class for CRISPR analysis and visualization"""

    def __init__(self, file_path, 
                 assay=None, assay_protein=None,
                 layer_perturbation=None, label_perturbation_type="Perturbed",
                 col_gene_symbols="gene_symbols", col_cell_type="leiden",
                 col_sample_id="standard_sample_id", 
                 col_batch=None,
                 col_perturbation="perturbation",
                 col_target_genes="guide_ids",
                 col_guide_rna="guide_ids",
                 key_control="NT", key_treatment="perturbed", **kwargs):
        """CRISPR class initialization."""
        self._assay = assay
        self._assay_protein = assay_protein
        self._layer_perturbation = layer_perturbation
        self._label_perturbation_type = label_perturbation_type
        self._columns = dict(col_gene_symbols=col_gene_symbols,
                             col_cell_type=col_cell_type, 
                             col_sample_id=col_sample_id, 
                             col_batch=col_batch,
                             col_perturbation=col_perturbation,
                             col_guide_rna=col_guide_rna,
                             col_target_genes=col_target_genes)
        self._keys = dict(key_control=key_control, key_treatment=key_treatment)
        self._file_path = file_path
        self.adata = file_path
        self.figures = {"main": {}}
        self.results = {"main": {}}

    @property
    def adata(self):
        """Specify file path to data."""
        return self._adata

    @adata.setter
    def adata(self, value):
        """Set file path and load object."""
        self._file_path = value
        if not isinstance(value, sc.AnnData):
            print("<<<CREATING OBJECT>>>")
            self._adata = cr.pp.create_object(
                value, assay=None, col_gene_symbols=self._columns[
                    "col_gene_symbols"])
        else:
            self._adata = value
            
    def plot(self, genes=None, assay="default", 
             title=None,  # NOTE: will apply to multiple plots
             marker_genes_dict=None,
             kws_gex=None, kws_clustering=None, 
             kws_gex_violin=None, kws_gex_matrix=None, 
             cmap_continuous="coolwarm", cmap_categorical="viridis",
             run_label="main"):
        
        # Setup 
        figs = {}
        if not isinstance(genes, (list, np.ndarray)) and (
            genes is None or genes == "all"):
            names = self.adata.var.reset_index()[
                self._columns["col_gene_symbols"]].copy()  # gene names
            if genes == "all": 
                genes = names.copy()
        else:
            names = genes.copy()
        genes, names = list(pd.unique(genes)), list(pd.unique(names))
        if assay == "default":
            assay = self._assay
        if assay:  # store available `.obs` columns
            cols_obs = self.adata[assay].obs.columns
        else:
            cols_obs = self.adata.obs.columns
        if "scaled" not in self.adata.layers:
            self.adata.layers['scaled'] = sc.pp.scale(
                self.adata, copy=True).X  # scaling (Z-scores)
        if layers == "all":  # to include all layers
            layers = list(self.adata.layers.copy())
        if None in layers:
            layers.remove(None)
            
        # Pre-Processing
        print("\n<<< PLOTTING PRE-PROCESSING >>>")
        if "preprocessing" in self.figures[run_label]:
            figs["preprocessing"] = self.figures[run_label]["preprocessing"]
            
        # Gene Expression Heatmap(s)
        if kws_gex is None:
            kws_gex = {"dendrogram": True, "show_gene_labels": True}
        if "cmap" not in kws_gex:
            kws_gex.update({"cmap": cmap_continuous})
        print("\n<<< PLOTTING GEX (Heatmap) >>>")
        rows, cols = cr.pl.square_grid(len([None] + list(layers)))
        figs["gene_expression"], axes_gex = plt.subplots(
            rows, cols, figsize=(20 * rows, int(4 * cols / 3)), 
            gridspec_kw={'wspace': 0.9})
        for j, i in enumerate([None] + list(layers)):
            lab = f"gene_expression_{i}" if i else "gene_expression"
            if i is None:
                hm_title = "Gene Expression"
            elif i == self._layer_perturbation:
                hm_title = f"Gene Expression (Perturbation Layer)"
            else:
                hm_title = f"Gene Expression ({i})"
            try:
                sc.pl.heatmap(
                    self.adata[assay] if assay else self.adata, names,
                    self._columns["col_cell_type"], layer=i,
                    ax=axes_gex[j],
                    gene_symbols=self._columns["col_gene_symbols"], 
                    show=False)  # GEX heatmap
                axes_gex[j].set_title("Raw" if i is None else i.capitalize())
                # figs[lab] = plt.gcf(), figs[lab]
                # figs[lab][0].suptitle(title if title else hm_title)
                # figs[lab][0].supxlabel("Gene")
                # figs[lab][0].show()
            except Exception as err:
                warnings.warn(
                    f"{err}\n\nCould not plot GEX heatmap ('{hm_title}').")
                figs[lab] = err
        
        # Gene Expression Violin Plots
        if kws_gex_violin is None:
            kws_gex_violin = {}
        if "cmap" not in kws_gex_violin:
            kws_gex_violin.update({"cmap": cmap_continuous})
        if "groupby" in kws_gex_violin or "col_cell_type" in kws_gex_violin:
            lab_cluster = kws_gex_violin.pop(
                "groupby" if "groupby" in kws_gex_violin else "col_cell_type")
        else:
            lab_cluster = self._columns["col_cell_type"]
        if lab_cluster not in cols_obs:
            lab_cluster = None   # None if cluster label N/A in `.obs`
        for i in zip(["dendrogram", "swap_axes", "cmap"], 
                     [True, False, cmap_continuous]):
            if i[0] not in kws_gex_violin:  # add default arguments
                kws_gex_violin.update({i[0]: i[1]})
        print("\n<<< PLOTTING GEX (Violin) >>>")
        for i in [None] + list(layers):
            try:
                lab = f"gene_expression_violin"
                title_gexv = title if title else "Gene Expression"
                if i:
                    lab += "_" + str(i)
                    if not title:
                        title_gexv = f"{title_gexv} ({i})"
                figs[lab] = sc.pl.stacked_violin(
                    self.adata[assay] if assay else self.adata, 
                    marker_genes_dict if marker_genes_dict else genes,
                    layer=i, groupby=lab_cluster, return_fig=True, 
                    gene_symbols=self._columns["col_gene_symbols"], 
                    title=title_gexv, show=False,
                    **kws_gex_violin)  # violin plot of GEX
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster)
                figs[lab].show()
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot GEX violins.")
                figs["gene_expression_violin"] = err
        
        # Gene Expression Matrix Plots
        if kws_gex_matrix is None:
            kws_gex_matrix = {}
        if "cmap" not in kws_gex_matrix:
            kws_gex_matrix.update({"cmap": cmap_continuous})
        if "groupby" in kws_gex_matrix or "col_cell_type" in kws_gex_matrix:
            lab_cluster = kws_gex_matrix.pop(
                "groupby" if "groupby" in kws_gex_matrix else "col_cell_type")
        else:
            lab_cluster = self._columns["col_cell_type"]
        if lab_cluster not in cols_obs:
            lab_cluster = None   # None if cluster label N/A in `.obs`
        for i in zip(["dendrogram", "swap_axes", "cmap"], 
                     [True, False, cmap_continuous]):
            if i[0] not in kws_gex_matrix:  # add default arguments
                kws_gex_matrix.update({i[0]: i[1]})
        print("\n<<< PLOTTING GEX (Matrix) >>>")
        print(kws_gex_matrix)
        try:
            for i in [None] + list(self.adata.layers):
                lab = f"gene_expression_matrix"
                title_gexm = title if title else "Gene Expression"
                if i:
                    lab += "_" + str(i)
                    if not title:
                        title_gexm = f"{title_gexm} ({i})"
                if i == "scaled":
                    bar_title = "Expression (Mean Z-Score)"
                else: 
                    bar_title = "Expression"
                figs[lab] = sc.pl.matrixplot(
                    self.adata[assay] if assay else self.adata, genes,
                    layer=i, return_fig=True, groupby=lab_cluster,
                    title=title_gexm,
                    gene_symbols=self._columns["col_gene_symbols"],
                    **{"colorbar_title": bar_title, **kws_gex_matrix
                       },  # colorbar title overriden if already in kws_gex
                    show=False)  # violin plot of GEX
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster)
                figs[lab].show()
        except Exception as err:
            warnings.warn(f"{err}\n\nCould not plot GEX matrix plot.")
            figs["gene_expression_matrix"] = err
        
        # UMAP
        if kws_clustering is None:
            kws_clustering = {"frameon": False, "legend_loc": "on_data"}
        title_umap = str(title if title else kws_clustering[
            "title"] if "title" in kws_clustering else "UMAP")
        color = kws_clustering.pop(
            "color") if "color" in kws_clustering else None  # color-coding
        if "cmap" in kws_clustering:  # in case use wrong form of argument
            kws_clustering["color_map"] = kws_clustering.pop("cmap")
        if "palette" not in kws_clustering:
            kws_clustering.update({"palette": cmap_categorical})
        if "X_umap" in self.adata.obsm:
            print("\n<<< PLOTTING UMAP >>>")
            if "col_cell_type" in kws_clustering:  # if provide cell column
                lab_cluster = kws_clustering.pop("col_cell_type")
            else: 
                lab_cluster = self._columns["col_cell_type"] 
            try:
                figs["clustering"] = sc.pl.umap(
                    self.adata[assay] if assay else self.adata, 
                    color=lab_cluster, return_fig=True, 
                    title=title_umap,  **kws_clustering)  # UMAP ~ cell type
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot UMAP clusters.")
                figs["clustering"] = err
            if genes is not None:
                print("\n<<< PLOTTING GEX ON UMAP >>>")
                try:
                    figs["clustering_gene_expression"] = sc.pl.umap(
                        self.adata[assay] if assay else self.adata, 
                        title=names,
                        return_fig=True, 
                        gene_symbols=self._columns["col_gene_symbols"],
                        color=names)  # UMAP ~ GEX
                except Exception as err:
                    warnings.warn(f"{err}\n\nCould not plot GEX UMAP.")
                    figs["clustering_gene_expression"] = err
            if color is not None:
                print(f"\n<<< PLOTTING {color} on UMAP >>>")
                try:
                    figs[f"clustering_{color}"] = sc.pl.umap(
                        self.adata[assay] if assay else self.adata, 
                        title=title if title else None, return_fig=True, 
                        color=color)  # UMAP ~ GEX
                except Exception as err:
                    warnings.warn(f"{err}\n\nCould not plot UMAP ~ {color}.")
                    figs[f"clustering_{color}"] = err
        else:
            print("\n<<< UMAP NOT AVAILABLE TO PLOT. RUN `.cluster()`. >>>")
        return figs
            
    
    def preprocess(self, assay=None, assay_protein=None, 
                   kws_process_guide_rna=None,
                   clustering=False, run_label="main", **kwargs):
        """Preprocess data."""
        if assay_protein is None:
            assay_protein = self._assay_protein
        if assay is None:
            assay = self._assay
        if kws_process_guide_rna is not None:  # process perturbation columns
            assay_gr = kws_process_guide_rna.pop(
                "assay") if "assay" in kws_process_guide_rna else assay
            self.process_guide_rna(assay=assay_gr, **kws_process_guide_rna)
        _, figs = cr.pp.process_data(
            self.adata, assay=assay, **self._columns,
            assay_protein=assay_protein, **kwargs)  # preprocess
        if run_label not in self.figures:
            self.figures[run_label] = {}
        self.figures[run_label]["preprocessing"] = figs
        if clustering not in [None, False]:
            if clustering is True:
                clustering = {}
            if isinstance(clustering, dict):
                figs_cl = self.cluster(**clustering)  # clustering
                figs = figs + [figs_cl]
            else:
                raise TypeError(
                    "'clustering' must be dict (keyword arguments) or bool.")
        return figs
    
    
    def process_guide_rna(self, assay=None, feature_split="|", guide_split="-",
                          key_control_patterns="CTRL"):
        """
        Convert specific guide RNA entries to general gene targets,
        including dummy-coded columns for each gene target (if no counts) 
        or count columns for each target, plus a column of gene target 
        categories stripped of specific guide ID suffixes (e.g., -1, -2-1).
        """
        if isinstance(key_control_patterns, str):
            key_control_patterns = [key_control_patterns]
        targets = self.adata.obs[self._columns["col_guide_rna"]].str.strip(
            " ").replace("", np.nan)
        if np.nan in key_control_patterns:
            key_control_patterns = list(pd.Series(key_control_patterns).dropna())
            targets = targets.replace(
                np.nan, key_control_patterns[0])  # NaNs replaced w/ control key
        targets = targets.apply(
            lambda x: [re.sub(f"{guide_split}.*", "", i) for i in x.split(
                feature_split)])  # each entry -> list of target genes
        targets = targets.apply(lambda x: [
            self._keys["key_control"] if any(
                (k in i for k in key_control_patterns)) else i 
                for i in x])  # find control keys among targets
        binary = targets.apply(
            lambda x: any(np.array(x) != self._keys["key_control"])).to_frame(
                self._columns["col_perturbation"])  # binary perturbed/not
        if assay: 
            self.adata[assay].obs = self.adata[assay].obs.join(
                targets.to_frame(self._columns["col_target_genes"]), 
                lsuffix="_original")
            self.adata[assay].obs = self.adata[assay].obs.join(
                binary, lsuffix="_original")
        else:
            self.adata.obs = self.adata.obs.join(
                targets.to_frame(self._columns["col_target_genes"]), 
                lsuffix="_original")
            self.adata.obs = self.adata.obs.join(binary, lsuffix="_original")
        for t in targets.explode().unique():
            if assay:
                self.adata[assay].obs[
                    f"{self._keys['key_treatment']}_{t}"] = targets.apply(
                        lambda x: self._keys['key_treatment'] if t in x else 
                        self._keys['key_control']
                        )  # treatment or control key for each target
            else:
                self.adata.obs[
                    f"{self._keys['key_treatment']}_{t}"] = targets.apply(
                        lambda x: self._keys['key_treatment'] if t in x else 
                        self._keys['key_control']
                        )  # treatment or control key for each target
                
                    
    def cluster(self, assay=None, method_cluster="leiden", 
                plot=True, colors=None, paga=False, 
                kws_pca=None, kws_neighbors=None, 
                kws_umap=None, kws_cluster=None, 
                run_label="main", test=False, **kwargs):
        """Perform dimensionality reduction and create UMAP."""
        if assay is None:
            assay = self._assay
        figs_cl = cr.ax.cluster(
            self.adata.copy() if test is True else self.adata, 
            assay=assay, method_cluster=method_cluster,
            **self._columns, **self._keys,
            plot=plot, colors=colors, paga=paga, 
            kws_pca=kws_pca, kws_neighbors=kws_neighbors,
            kws_umap=kws_umap, kws_cluster=kws_cluster, **kwargs)
        if test is False:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"clustering": figs_cl})
        return figs_cl
        
        
    def run_mixscape(self, 
                     assay=None, target_gene_idents=True, 
                     col_split_by=None, min_de_genes=5,
                     label_perturbation_type=None, run_label="main", 
                     test=False, plot=True):
        """Run Mixscape.""" 
        if assay is None:
            assay = self._assay
        if label_perturbation_type is None:
            label_perturbation_type=self._label_perturbation_type
        figs_mix = cr.ax.perform_mixscape(
            self.adata.copy() if test is True else self.adata, assay=assay,
            **self._columns,
            key_control=self._keys["key_control"],
            label_perturbation_type=self._label_perturbation_type, 
            layer_perturbation=self._layer_perturbation, 
            target_gene_idents=target_gene_idents,
            min_de_genes=min_de_genes, col_split_by=col_split_by, plot=plot)
        if test is False:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label].update({"mixscape": figs_mix})
        return figs_mix
    
    def run_augur(self, assay=None, 
                  col_perturbation=None, key_treatment=None,
                  augur_mode="default", classifier="random_forest_classifier", 
                  kws_augur_predict=None, n_folds=3,
                  subsample_size=20, n_threads=True, 
                  select_variance_features=False, 
                  seed=1618, plot=True, run_label="main", test=False):
        """Run Augur."""
        if test is True:
            annd = self.adata.copy()
        if key_treatment is None:
            key_treatment = self._keys["key_treatment"]
        if col_perturbation is None:
            col_perturbation = self._columns["col_perturbation"]
        if n_threads is True:
            n_threads = os.cpu_count() - 1 # use available CPUs - 1
        if assay is None:
            assay = self._assay
        if kws_augur_predict is None:
            kws_augur_predict = {}
        # if run_label != "main":
        #     kws_augur_predict.update(
        #         {"key_added": f"augurpy_results_{run_label}"}
        #         )  # run label incorporated in key in adata
        data, results, figs_aug = cr.ax.perform_augur(
            self.adata.copy() if test is True else self.adata, 
            assay=assay, classifier=classifier, 
            augur_mode=augur_mode, subsample_size=subsample_size,
            select_variance_features=select_variance_features, 
            n_folds=n_folds,
            **{**self._columns, "col_perturbation": col_perturbation},  
            kws_augur_predict=kws_augur_predict,
            key_control=self._keys["key_control"], key_treatment=key_treatment,
            layer=self._layer_perturbation,
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
    
    def compute_distance(self, distance_type="edistance", method="X_pca", 
                         kws_plot=None, highlight_real_range=False,
                         run_label="main", plot=True):
        """Compute and visualize distance metrics."""
        output = cr.ax.compute_distance(
            self.adata, **self._columns, **self._keys,
            distance_type=distance_type, method=method,
            kws_plot=kws_plot, highlight_real_range=highlight_real_range, 
            plot=plot)
        if plot is True:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label]["distances"] = {}
            self.figures[run_label]["distances"].update(
                {f"{distance_type}_{method}": output[-1]})
            if run_label not in self.results:
                self.results[run_label] = {}
            self.results[run_label]["distances"] = {}
            self.results[run_label]["distances"].update(
                {f"{distance_type}_{method}": output})
        return output
    
    def compute_distance(self, distance_type="edistance", method="X_pca", 
                         kws_plot=None, highlight_real_range=False,
                         run_label="main", plot=True):
        """Analyze cell type composition changes."""
        output = cr.ax.compute_distance(
            self.adata, **self._columns, **self._keys,
            distance_type=distance_type, method=method,
            kws_plot=kws_plot, highlight_real_range=highlight_real_range, 
            plot=plot)
        if plot is True:
            if run_label not in self.figures:
                self.figures[run_label] = {}
            self.figures[run_label]["distances"] = {}
            self.figures[run_label]["distances"].update(
                {f"{distance_type}_{method}": output[-1]})
            if run_label not in self.results:
                self.results[run_label] = {}
            self.results[run_label]["distances"] = {}
            self.results[run_label]["distances"].update(
                {f"{distance_type}_{method}": output})
        return output
          
    
    # def save_output(self, directory_path, run_keys="all", overwrite=False):
    #     """Save figures, results, adata object."""
    #     # TODO: FINISH
    #     raise NotImplementedError("Saving output isn't yet fully implemented.")
    #     if isinstance(run_keys, (str, float)):
    #         run_keys = [run_keys]
    #     elif run_keys == "all":
    #         run_keys = list(self.figures.keys())
    #         run_keys += list(self.results.keys())
    #         run_keys = list(pd.unique(self.results))
    #     for r in run_keys:
    #         os.makedirs(os.path.join(directory_path, r), exist_ok=True)
    #         if r in self.figures:
    #             for f in self.figures[r]:
    #                 dirp = os.path.join(directory_path, r, f)
    #                 os.makedirs(dirp, exist_ok=overwrite)
    #                 # self.figures[r][f].savefig(
    #                 #     os.path.join(dirp, f"{f}.pdf"))
    #         if r in self.results:
    #             for f in self.results[r]:
    #                 dirp = os.path.join(directory_path, r)
    #                 if not os.path.exists(dirp)
    #                 os.makedirs(dirp, exist_ok=overwrite)
            
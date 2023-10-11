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
import seaborn as sns
import re
import pertpy as pt
import pandas as pd
import numpy as np
import crispr as cr
from crispr.defaults import (names_layers)

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Crispr(object):
    """An object class for CRISPR analysis and visualization"""

    def __init__(self, file_path, 
                 assay=None, 
                 assay_protein=None,
                 layer_perturbation=None, 
                 col_gene_symbols="gene_symbols", 
                 col_cell_type="leiden",
                 col_sample_id="standard_sample_id", 
                 col_batch=None,
                 col_perturbation="perturbation",
                 col_target_genes=None,
                 col_guide_rna="guide_ids",
                 col_num_umis="num_umis",
                 key_control="NT", 
                 key_treatment="perturbed", 
                 key_nonperturbed="NP", 
                 kws_process_guide_rna=None,
                 remove_multi_transfected=True,
                 **kwargs):
        """CRISPR class initialization."""
        print("\n\n<<<INITIALIZING CRISPR CLASS OBJECT>>>\n")
        self._assay = assay
        self._assay_protein = assay_protein
        self._layer_perturbation = layer_perturbation
        self._file_path = file_path
        if kwargs:
            print(f"\nUnused keyword arguments: {kwargs}.\n")
        
        # Create Object & Attributes to Store Results/Figures
        self.adata = cr.pp.create_object(
            self._file_path, assay=None, col_gene_symbols=col_gene_symbols)
        self.figures = {"main": {}}
        self.results = {"main": {}}
        self._info = {"descriptives": {}, 
                      "guide_rna": {}}  # extra info to store by methods
        print(self.adata.obs, "\n\n") if assay else None
        print("\n\n", self.adata[assay].obs if assay else self.adata)
        
        # Process Guide RNAs (optional)
        self._info["guide_rna"]["keywords"] = kws_process_guide_rna
        if kws_process_guide_rna is not None:
            print("\n\n<<<PERFORMING gRNA PROCESSING AND FILTERING>>>\n")
            tg_info, feats_n = cr.pp.filter_by_guide_counts(
                self.adata[assay] if assay else self.adata, 
                col_guide_rna, col_num_umis=col_num_umis,
                key_control=key_control, **kws_process_guide_rna
                )  # process (e.g., multi-probe names) & filter by sgRNA counts
            self._info["guide_rna"]["counts_unfiltered"] = feats_n.loc[
                self.adata.obs.index]
            for q in [col_guide_rna, 
                      col_num_umis]:  # replace w/ processed entries
                tg_info.loc[:, q] = tg_info[q + "_filtered"]
            if remove_multi_transfected is True:
                self._info["guide_rna"]["counts_single_multi"] = tg_info.copy()
                tg_info = tg_info.loc[tg_info[
                    f"{col_guide_rna}_list_filtered"].apply(
                        lambda x: np.nan if len(x) > 1 else x).dropna().index]
            self._info["guide_rna"]["counts"] = tg_info.copy()
            if assay:
                self.adata[assay].obs = self.adata[assay].obs.join(
                    tg_info, lsuffix="_original")
                self.adata[assay].obs.loc[:,  col_target_genes] = self.adata[
                    assay].obs[col_guide_rna]  
                # ^ col_target_genes=processed col_guide_rna 
                if remove_multi_transfected is True:
                    self.adata[assay] = self.adata[assay][
                        ~self.adata[assay].obs[col_guide_rna].isnull()]
            else:
                self.adata.obs = self.adata.obs.join(
                    tg_info, lsuffix="_original")
                if remove_multi_transfected is True:
                    self.adata = self.adata[
                        ~self.adata.obs[col_guide_rna].isnull()]
                self.adata.obs.loc[:,  col_target_genes] = self.adata.obs[
                    col_guide_rna]  # col_target_genes=processed col_guide_rna 
                if remove_multi_transfected is True:
                    # drop cells w/ multiple guides that survived filtering
                    self.adata = self.adata[
                        ~self.adata.obs[col_guide_rna].isnull()]
                    
            # Perturbed vs. Untreated Column
            if col_perturbation not in self.adata.obs:
                print("\n\n<<<CREATING PERTURBED/CONTROL COLUMN>>>\n")
                if assay:
                    self.adata[assay].obs.loc[
                        :, col_perturbation] = self.adata[assay].obs[
                            col_target_genes].apply(
                                lambda x: key_control if pd.isnull(x) or (
                                    x == key_control) else key_treatment)
                else:
                    self.adata.obs.loc[:, col_perturbation] = self.adata.obs[
                        col_target_genes].apply(
                            lambda x: key_control if pd.isnull(x) or (
                                x == key_control) else key_treatment)
        
            # Store Columns & Keys within Columns as Dictionary Attributes
            if col_target_genes is None:  # if unspecified...
                col_target_genes = col_guide_rna  # ...= guide ID (maybe processed)
            self._columns = dict(col_gene_symbols=col_gene_symbols,
                                 col_cell_type=col_cell_type, 
                                 col_sample_id=col_sample_id, 
                                 col_batch=col_batch,
                                 col_perturbation=col_perturbation,
                                 col_guide_rna=col_guide_rna,
                                 col_num_umis=col_num_umis,
                                 col_target_genes=col_target_genes)
            self._keys = dict(key_control=key_control, 
                              key_treatment=key_treatment,
                              key_nonperturbed=key_nonperturbed)
            for q in [self._columns, self._keys]:
                print("\n".join([
                    f"{x} = " + ["", "'"][int(isinstance(q[x], str))] + str(q[
                        x]) + ["", "'"][int(isinstance(q[x], str))] 
                    for x in q]))
            print("\n\n", self.adata[assay].obs if assay else self.adata)
        
            # Correct 10x CellRanger Guide Count Incorporation
            # raise NotImplementedError(
            #     "Delete guides from var_names not yet implemented!")

    # @property
    # def adata(self):
    #     """Specify file path to data."""
    #     return self._adata

    # @adata.setter
    # def adata(self, value):
    #     """Set file path and load object."""
    #     self._file_path = value
    #     if isinstance(value, str):
    #         print("\n\n<<<CREATING OBJECT>>>")
    #         self._adata = cr.pp.create_object(
    #             value, assay=None, col_gene_symbols=self._columns[
    #                 "col_gene_symbols"])
    #     else:
    #         self._adata = value
    
    def describe(self, group_by=None, plot=False):
        desc, figs = {}, {}
        gbp = [self._columns["col_cell_type"]]
        if group_by:
            gbp += [group_by]
        if "descriptives" not in self._info:
            self._info["descriptives"] = {}
        print("\n\n\n", self.adata.obs.describe().round(2),
              "\n\n\n")
        print(f"\n\n{'=' * 80}\nDESCRIPTIVES\n{'=' * 80}\n\n")
        print(self.adata.obs.describe())
        for g in gbp:
            print(self.adata.obs.groupby(g).describe().round(2), 
                  f"\n\n{'-' * 40}\n\n") 
        try:
            if "guide_rna_counts" not in self._info["descriptives"]:
                self._info["descriptives"][
                    "guide_rna_counts"] = self.count_gRNAs()
            desc["n_grnas"] = self._info["descriptives"][
                "guide_rna_counts"].groupby("gene").describe().rename(
                    {"mean": "mean sgRNA count/cell"})
            print(desc["n_grnas"])
            if plot is True:
                n_grna = self._info["descriptives"][
                    "guide_rna_counts"].to_frame("sgRNA Count")
                if group_by is not None:
                    n_grna = n_grna.join(self.adata.obs[
                        group_by].rename_axis("bc"))  # join group_by variable
                figs["n_grna"] = sns.catplot(data=n_grna.reset_index(),
                    y="gene", x="sgRNA Count", kind="violin", hue=group_by,
                    height=len(self._info["descriptives"][
                        "guide_rna_counts"].reset_index().gene.unique())
                    )  # violin plot of gNRA counts per cell
        except Exception as err:
            warnings.warn(f"{err}\n\n\nCould not describe sgRNA count.")
        # try:
        print(f"\n\n{'=' * 80}\nCELL COUNTS\n{'=' * 80}\n")
        print(f"\t\t\tTotal Cells: {self.adata.shape[0]}")
        desc["n_cells"], figs["n_cells"] = {}, {}
        desc["n_cells"]["all"] = self.adata.shape[0]
        for g in gbp:
            if g in self.adata.obs:
                print(f"\t\t\n***** By {g} *****\n")
                desc["n_cells"][g] = self.adata.obs.groupby(g).apply(
                    lambda x: x.shape[0])
                print("\t\t\t\n", dict(desc["n_cells"][g].T))
                if plot is True:
                    figs["n_cells"][g] = sns.catplot(
                        data=desc["n_cells"][g].to_frame("N").reset_index(), 
                        y="N", kind="bar", x=g, hue=g)  # bar plot of cell #s
        # except Exception as err:
        #     warnings.warn(f"{err}\n\n\nCould not describe cell counts.")
        self._info["descriptives"].update(desc)
        return figs
    
    def get_guide_rna_counts(self, target_gene_idents=None, group_by=None,
                             col_cell_type=None, **kwargs):
        """Plot guide RNA counts by cell type & up to 1 other variable."""
        if isinstance(target_gene_idents, str):
            target_gene_idents = [target_gene_idents]
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        if group_by is None:
            group_by = [col_cell_type]
        if isinstance(group_by, str):
            group_by = [group_by]
        if group_by is True:
            group_by = [col_cell_type]
        if len(group_by) > 1:
            kwargs.update({"row": group_by[1]})
        if "share_x" not in kwargs: 
            kwargs.update({"share_x": True})
        # sufxs = ["all", "filtered"]
        # cols = [self._columns[x] for x in ["col_guide_rna", "col_num_umis"]] 
        # grna = [self._info["guide_rna"]["counts_raw"][
        #     [f"{i}_list_{x}" for i in cols]] for x in sufxs]
        # for g in grna:
        #     g.columns = cols
        # grna = pd.concat(grna, keys=sufxs, names=["version", "bc"])
        # dff = grna.loc["all"].apply(
        #     lambda x: pd.Series([np.nan] * x.shape[0], index=x.index) 
        #     if len(pd.unique(x[cols[0]])) > 1 else pd.Series(
        #         pd.unique(x[cols[1]], index=pd.unique(x[cols[0]]))), 
        #         axis=1).dropna()
        # dff = dff.apply(
        #     lambda x: pd.Series(x[cols[1]], index=x[cols[0]]), 
        #     axis=1).stack().sort_index().rename_axis(
        #         ["b", "Target"]).to_frame("N gRNAs").join(
        #             self.adata.obs[[col_cell_type]].rename_axis(
        #                 "b")).reset_index()
        dff = self._info["guide_rna"]["counts_unfiltered"].reset_index(
            "Gene").rename({"Gene": "Target"}, axis=1).join(
                self.adata.obs[group_by])  # gRNA counts + cell types
        if target_gene_idents:
            dff = dff[dff.Target.isin(target_gene_idents)]
        if group_by is not None:
            kwargs.update({"col": group_by[0]})
        if len(group_by) == 1:
            kwargs.update({"col_wrap": cr.pl.square_grid(len(
                dff[group_by[0]].unique()))[1]})
        # fig = sns.catplot(data=dff[dff.Target.isin(
        #     target_gene_idents)] if target_gene_idents else dff, 
        #             x="N gRNAs", y="Target", col_wrap=cr.pl.square_grid(
        #                 len(self.adata.obs[col_cell_type].unique()))[1],
        #             col=col_cell_type, kind="violin", 
        #             **kwargs)
        fig = sns.catplot(data=dff, x="Percent of Cell Guides", y="Target",
                          kind="violin", **kwargs)
        fig.fig.suptitle("Guide RNA Counts by Cell Type")
        fig.fig.tight_layout()
        return self._info["guide_rna"]["counts_unfiltered"], fig
    
    def preprocess(self, assay=None, assay_protein=None,
                   clustering=False, run_label="main", 
                   remove_doublets=True, **kwargs):
        """Preprocess data."""
        if assay_protein is None:
            assay_protein = self._assay_protein
        if assay is None:
            assay = self._assay
        _, figs = cr.pp.process_data(
            self.adata, assay=assay, assay_protein=assay_protein, 
            remove_doublets=remove_doublets, **self._columns,
            **kwargs)  # preprocess
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
                    "`clustering` must be dict (keyword arguments) or bool.")
        return figs
                
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
    
    def find_markers(self, assay=None, n_genes=5, layer="scaled", 
                     method="wilcoxon", key_reference="rest", 
                     plot=True, **kwargs):
        if assay is None:
            assay = self._assay
        marks, figs_m = cr.ax.find_markers(
            self.adata, assay=assay, method=method, n_genes=n_genes, 
            layer=layer, key_reference=key_reference, plot=plot,
            col_cell_type=self._columns["col_cell_type"], **kwargs)
        print(marks)
        return marks, figs_m
        
    def run_mixscape(self, assay=None, target_gene_idents=True, 
                     col_split_by=None, min_de_genes=5,
                     run_label="main",
                     test=False, plot=True, **kwargs):
        """Run Mixscape.""" 
        if assay is None:
            assay = self._assay
        figs_mix = cr.ax.perform_mixscape(
            self.adata.copy() if test is True else self.adata, assay=assay,
            **self._columns, **self._keys,
            layer_perturbation=self._layer_perturbation, 
            target_gene_idents=target_gene_idents,
            min_de_genes=min_de_genes, col_split_by=col_split_by, 
            plot=plot, **kwargs)
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
                  seed=1618, plot=True, run_label="main", test=False,
                  **kwargs):
        """Run Augur."""
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
            seed=seed, n_threads=n_threads, plot=plot, **kwargs)
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
    
    def run_gsea(self, key_condition=None, 
                 filter_by_highly_variable=False, 
                 run_label="main", **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        output = cr.ax.perform_gsea(
            self.adata, filter_by_highly_variable=filter_by_highly_variable, 
            **{**self._keys, "key_condition": self._keys[
                "key_condition"] if key_condition is None else key_condition
               }, **self._columns, **kwargs)  # GSEA
        if run_label not in self.results:
            self.results[run_label] = {}
        self.results[run_label]["gsea"] = output
        return output
          
    def run_composition_analysis(self, reference_cell_type, 
                                 assay=None, analysis_type="cell_level", 
                                 col_perturbation=None,
                                 est_fdr=0.05, generate_sample_level=False,
                                 plot=True, run_label="main", **kwargs):
        """Perform gene set enrichment analyses & plotting."""
        if col_perturbation is None:
            col_perturbation = self._columns["col_perturbation"]
        output = cr.ax.analyze_composition(
            self.adata, reference_cell_type,
            assay=assay if assay else self._assay, analysis_type=analysis_type,
            generate_sample_level=generate_sample_level, 
            col_cell_type=self._columns["col_cell_type"],
            col_perturbation=col_perturbation, est_fdr=est_fdr, plot=plot, 
            **self._columns, **self._keys, **kwargs)
        if run_label not in self.results:
            self.results[run_label] = {}
        self.results[run_label]["composition"] = output
        return output
    
    def run_dialogue(self, n_programs=3, col_cell_type=None,
                     cmap="coolwarm", vcenter=0, 
                     run_label="main", **kws_plot):
        """Analyze multicellular programs."""
        if col_cell_type is None:
            col_cell_type = self._columns["col_cell_type"]
        d_l = pt.tl.Dialogue(
            sample_id=self._columns["col_perturbation"], n_mpcs=n_programs,
            celltype_key=col_cell_type, 
            n_counts_key=self._columns["col_num_umis"])
        pdata, mcps, ws, ct_subs = d_l.calculate_multifactor_PMD(
            self.adata.copy(), normalize=True)
        mcp_cols = list(set(pdata.obs.columns).difference(
            self.adata.obs.columns))
        cols = cr.pl.square_grid(len(mcp_cols) + 2)[1]
        fig = sc.pl.umap(
            pdata, color=mcp_cols + [
                self._columns["col_perturbation"], col_cell_type],
            ncols=cols, cmap=cmap, vcenter=vcenter, **kws_plot)
        self.results[run_label]["dialogue"] = pdata, mcps, ws, ct_subs
        self.figures[run_label]["dialogue"] = fig
        return fig
            
    def plot(self, genes=None, genes_highlight=None,
             cell_types_circle=None,
             assay="default", 
             title=None,  # NOTE: will apply to multiple plots
             layers="all",
             marker_genes_dict=None,
             kws_gex=None, kws_clustering=None, 
             kws_gex_violin=None, kws_gex_matrix=None, 
             run_label="main"):
        
        # Setup  Arguments & Empty Output
        figs = {}
        if kws_clustering is None:
            kws_clustering = {"frameon": False, "legend_loc": "on_data"}
        if "legend_loc" not in kws_clustering:
            kws_clustering.update({"legend_loc": "on_data"})
        if "vcenter" not in kws_clustering:
            kws_clustering.update({"vcenter": 0})
        if "col_cell_type" in kws_clustering:  # if provide cell column
            lab_cluster = kws_clustering.pop("col_cell_type")
        else: 
            lab_cluster = self._columns["col_cell_type"] 
        if not isinstance(genes, (list, np.ndarray)) and (
            genes is None or genes == "all"):
            names = self.adata.var.reset_index()[
                self._columns["col_gene_symbols"]].copy()  # gene names
            if genes == "all": 
                genes = names.copy()
        else:
            names = genes.copy()
        if genes_highlight and not isinstance(genes_highlight, list):
            genes_highlight = [genes_highlight] if isinstance(
                genes_highlight, str) else list(genes_highlight)
        if cell_types_circle and not isinstance(cell_types_circle, list):
            cell_types_circle = [cell_types_circle] if isinstance(
                cell_types_circle, str) else list(cell_types_circle)
        genes, names = list(pd.unique(genes)), list(pd.unique(names))
        if assay == "default":
            assay = self._assay
        if assay:  # store available `.obs` columns
            cols_obs = self.adata[assay].obs.columns
        else:
            cols_obs = self.adata.obs.columns
        if names_layers["scaled"] not in self.adata.layers:
            self.adata.layers[names_layers["scaled"]] = sc.pp.scale(
                self.adata, copy=True).X  # scaling (Z-scores)
        if layers == "all":  # to include all layers
            layers = list(self.adata.layers.copy())
        if None in layers:
            layers.remove(None)
        gene_symbols = None
        if self._columns["col_gene_symbols"] != self.adata.var.index.names[0]:
            gene_symbols = self._columns["col_gene_symbols"]
            
        # Pre-Processing/QC
        if "preprocessing" in self.figures[run_label]:
            print("\n<<< PLOTTING PRE-PROCESSING >>>")
            figs["preprocessing"] = self.figures[run_label]["preprocessing"]
            
        # Gene Expression Heatmap(s)
        if kws_gex is None:
            kws_gex = {"dendrogram": True, "show_gene_labels": True}
        if "cmap" not in kws_gex:
            kws_gex.update({"cmap": COLOR_MAP})
        print("\n<<< PLOTTING GEX (Heatmap) >>>")
        for j, i in enumerate([None] + list(layers)):
            lab = f"gene_expression_{i}" if i else "gene_expression"
            if i is None:
                hm_title = "Gene Expression"
            elif i == self._layer_perturbation:
                hm_title = f"Gene Expression (Perturbation Layer)"
            else:
                hm_title = f"Gene Expression ({i})"
            try:
                # sc.pl.heatmap(
                #     self.adata[assay] if assay else self.adata, names,
                #     lab_cluster, layer=i,
                #     ax=axes_gex[j], gene_symbols=gene_symbols, 
                #     show=False)  # GEX heatmap
                figs[lab] = sc.pl.heatmap(
                    self.adata[assay] if assay else self.adata, names,
                    lab_cluster, layer=i,
                    gene_symbols=gene_symbols, show=False)  # GEX heatmap
                # axes_gex[j].set_title("Raw" if i is None else i.capitalize())
                figs[lab] = plt.gcf(), figs[lab]
                figs[lab][0].suptitle(title if title else hm_title)
                # figs[lab][0].supxlabel("Gene")
                figs[lab][0].show()
            except Exception as err:
                warnings.warn(
                    f"{err}\n\nCould not plot GEX heatmap ('{hm_title}').")
                figs[lab] = err
        
        # Gene Expression Violin Plots
        if kws_gex_violin is None:
            kws_gex_violin = {}
        if "color_map" in kws_gex_violin:
            kws_gex_violin["cmap"] = kws_gex_violin.pop("color_map")
        if "cmap" not in kws_gex_violin:
            kws_gex_violin.update({"cmap": COLOR_MAP})
        if "groupby" in kws_gex_violin or "col_cell_type" in kws_gex_violin:
            lab_cluster = kws_gex_violin.pop(
                "groupby" if "groupby" in kws_gex_violin else "col_cell_type")
        else:
            lab_cluster = self._columns["col_cell_type"]
        if lab_cluster not in cols_obs:
            lab_cluster = None   # None if cluster label N/A in `.obs`
        for i in zip(["dendrogram", "swap_axes", "cmap"], 
                     [True, False, COLOR_MAP]):
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
                    gene_symbols=gene_symbols, title=title_gexv, show=False,
                    **kws_gex_violin)  # violin plot of GEX
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster)
                figs[lab].show()
            except Exception as err:
                warnings.warn(f"{err}\n\nCould not plot GEX violins.")
                figs["gene_expression_violin"] = err
                
                
        # Gene Expression Dot Plot
        figs["gene_expression_dot"] = sc.pl.dotplot(
            self.adata[assay] if assay else self.adata, genes, lab_cluster, show=False)
        if genes_highlight is not None:
            for x in figs["gene_expression_dot"][
                "mainplot_ax"].get_xticklabels():
                # x.set_style("italic")
                if x.get_text() in genes_highlight:
                    x.set_color('#A97F03')
        
        # Gene Expression Matrix Plots
        if kws_gex_matrix is None:
            kws_gex_matrix = {}
        if "color_map" in kws_gex_matrix:
            kws_gex_matrix["cmap"] = kws_gex_matrix.pop("color_map")
        if "cmap" not in kws_gex_matrix:
            kws_gex_matrix.update({"cmap": COLOR_MAP})
        if "groupby" in kws_gex_matrix or "col_cell_type" in kws_gex_matrix:
            lab_cluster_mat = kws_gex_matrix.pop(
                "groupby" if "groupby" in kws_gex_matrix else "col_cell_type")
        else:
            lab_cluster_mat = self._columns["col_cell_type"]
        if lab_cluster_mat not in cols_obs:
            lab_cluster_mat = None   # None if cluster label N/A in `.obs`
        for i in zip(["dendrogram", "swap_axes", "cmap"], 
                     [True, False, COLOR_MAP]):
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
                if i == names_layers["scaled"]:
                    bar_title = "Expression (Mean Z-Score)"
                else: 
                    bar_title = "Expression"
                figs[lab] = sc.pl.matrixplot(
                    self.adata[assay] if assay else self.adata, genes,
                    layer=i, return_fig=True, groupby=lab_cluster_mat,
                    title=title_gexm, gene_symbols=gene_symbols,
                    **{"colorbar_title": bar_title, **kws_gex_matrix
                       },  # colorbar title overriden if already in kws_gex
                    show=False)  # violin plot of GEX
                # figs[lab].fig.supxlabel("Gene")
                # figs[lab].fig.supylabel(lab_cluster_mat)
                figs[lab].show()
        except Exception as err:
            warnings.warn(f"{err}\n\nCould not plot GEX matrix plot.")
            figs["gene_expression_matrix"] = err
        
        # UMAP
        title_umap = str(title if title else kws_clustering[
            "title"] if "title" in kws_clustering else "UMAP")
        color = kws_clustering.pop(
            "color") if "color" in kws_clustering else None  # color-coding
        if "cmap" in kws_clustering:  # in case use wrong form of argument
            kws_clustering["color_map"] = kws_clustering.pop("cmap")
        if "palette" not in kws_clustering:
            kws_clustering.update({"palette": COLOR_PALETTE})
        if "color_map" not in kws_clustering:
            kws_clustering.update({"color_map": COLOR_MAP})
        if "X_umap" in self.adata.obsm or lab_cluster in (
            self.adata[assay].obs if assay else self.adata.obs).columns:
            print("\n<<< PLOTTING UMAP >>>")
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
                        title=names, return_fig=True, 
                        gene_symbols=gene_symbols, color=names,
                        **kws_clustering)  # UMAP ~ GEX
                except Exception as err:
                    warnings.warn(f"{err}\n\nCould not plot GEX UMAP.")
                    figs["clustering_gene_expression"] = err
            if color is not None:
                print(f"\n<<< PLOTTING {color} on UMAP >>>")
                try:
                    figs[f"clustering_{color}"] = sc.pl.umap(
                        self.adata[assay] if assay else self.adata, 
                        title=title if title else None, return_fig=True, 
                        color=color, frameon=False,
                        **kws_clustering)  # UMAP ~ GEX
                except Exception as err:
                    warnings.warn(f"{err}\n\nCould not plot UMAP ~ {color}.")
                    figs[f"clustering_{color}"] = err
            if cell_types_circle and "X_umap" in (
                self.adata[assay] if assay else self.adata
                ).obsm:  # UMAP(s) with circled cell type(s)?
                fump, axu = plt.subplots(figsize=(3, 3))
                sc.pl.umap(self.adata[assay] if assay else self.adata, 
                           color=[lab_cluster], ax=axu, show=False)  # UMAP
                for h in cell_types_circle:
                    if assay:
                        location_cells = self.adata[assay][self.adata[
                            assay].obs[lab_cluster] == h, :].obsm['X_umap']
                    else:
                        location_cells = self.adata[
                            self.adata.obs[lab_cluster] == h, :].obsm['X_umap']
                    coordinates = [location_cells[:, i].mean() for i in [0, 1]]
                    circle = plt.Circle(tuple(coordinates), 1.5, color="r", 
                                        clip_on=False, fill=False)  # circle
                    axu.add_patch(circle)
                # l_1 = axu.get_legend()  # save original Legend
                # l_1.set_title(lab_cluster)
                # # Make a new Legend for the mark
                # l_2 = axu.legend(handles=[Line2D(
                #     [0],[0],marker="o", color="k", markerfacecolor="none", 
                #     markersize=12, markeredgecolor="r", lw=0, 
                #     label="selected")], frameon=False, 
                #                 bbox_to_anchor=(3,1), title='Annotation')
                #     # Add back the original Legend (was overwritten by new)
                # _ = plt.gca().add_artist(l_1)
                figs["umap_annotated"] = fump
        else:
            print("\n<<< UMAP NOT AVAILABLE TO PLOT. RUN `.cluster()`. >>>")
        return figs
        
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
            
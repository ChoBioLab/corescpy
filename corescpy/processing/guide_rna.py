import re
import os
from warnings import warn
import copy
import pandas as pd
import numpy as np
import corescpy as cr


def process_guide_rna(adata, col_guide_rna="feature_call",
                      col_guide_rna_new="perturbation",
                      col_num_umis="num_umis",
                      feature_split=None, guide_split=None,
                      file_perturbations=None,
                      key_control_patterns=None, key_control="NT",
                      remove_multi_transfected=False, filter_kws=None):
    """
    Process and filter guide RNAs, (optionally) remove cells considered
    multiply-transfected (after filtering criteria applied), and remove
    guide counts from the gene expression matrix (wrapper function).

    Args:
        adata (AnnData): AnnData object (RNA assay,
            so if multi-modal, subset before passing to this argument).
        col_guide_rna (str): Name of the column containing guide IDs.
        col_num_umis (str): Column with the UMI counts (string entries
            with (for designs with possible multiple-transfection)
            within-cell probes separated by `feature_split` and
            any probe ID suffixes at the end following `guide_split`
            (e.g., '-' if STAT1-1-2, STAT-1-1-4, etc.).
        min_n (int, optional): Minimum number of counts to retain a
            guide past filtering. The default is None.
        max_pct_control_drop (int, optional): If control
            UMI counts are less than or equal to this percentage of the
            total counts for that cell, and if a non-control sgRNA is
            also present and meets other filtering criteria, then
            consider that cell pseudo-single-transfected
            (non-control gene). Note that controls in
            multiply-transfected cells will also be ultimately dropped
            if `drop_multi_control` is True.
            Dropping with this criterion means cells with only control
            guides will be completely dropped if not meeting criteria.
            Set to 0 to ignore this filtering. Defaults to None.
        min_n_target_control_drop (int, optional): If UMI counts
            across target (non-control) guides are above this number,
            notwithstanding whether control guide percent exceeds
            `max_percent_umis_control_drop`, drop control from that
            cell. For instance, if 100 and
            `max_percent_umis_control_drop=75`, even if a cell's
            UMIs are 75% control guides, if there are at least 100
            non-control guides, consider the cell only transfected for
            the non-control guides. Set to None to ignore this
            filtering criterion. Note that controls in
            multiply-transfected cells will also be ultimately dropped
            if `drop_multi_control` is True.
            Dropping with this criterion means cells with only control
            guides will be completely dropped if not meeting criteria.
            Defaults to 100.
        drop_multi_control (bool, optional): If True, drop control
            guides from cells that, after all other filtering,
            are multiply-transfected. Defaults to True.
        min_pct_avg_n (int, optional): sgRNAs with counts below this
            percentage of the average UMI count will be considered
            noise and dropped from the list of genes for which
            that cell is considered transfected. Defaults to 40.
        min_pct_dominant (int, optional): sgRNAs with counts at or
            above this percentage of the cell total UMI count will be
            considered dominant, and all other guides will be dropped
            from the list of genes for whichmthat cell is considered
            transfected. Defaults to "highest" (will choose most
            abundant guide as the dominant guide).
        feature_split (str, optional): For designs with multiple
            guides, the character that splits guide names in
            `col_guide_rna`. For instance, "|" for
            `STAT1-1|CNTRL-1|CDKN1A`. Defaults to "|".
            If only single guides, set to None.
        guide_split (str, optional): The character that separates
            guide (rather than gene target)-specific IDs within gene.
            For instance, guides targeting STAT1 may include
            STAT1-1, STAT1-2, etc.; the argument would be "-"
            so the function can identify all of those as
            targeting STAT1. Defaults to "-".
        key_control_patterns (list, optional): List (or single string)
            of patterns in guide RNA column entries that correspond to
            a control. For instance, if control entries in the original
            `col_guide_rna` column include `NEGCNTRL` and
            `Control.D`, you should specify ['Control', 'CNTRL']
            (assuming no non-control sgRNA names contain
            those patterns). If blank entries should be interpreted as
            control guides, then include np.nan/numpy.nan in this list.
            Defaults to None -> [np.nan].
        key_control (str, optional): The name you want the control
            entries to be categorized as under the new `col_guide_rna`.
            for instance, `CNTRL-1`, `NEGCNTRL`, etc. would all be
            replaced by "Control" if that's what you specify here.
            Defaults to "Control".

    Returns:
        AnnData: An AnnData object with final, processed guide
        names converted to target genes in `.obs[<col_guide_rna_new>]`
        and the corresponding UMI counts (summed across guides
        targeting the same gene within-cell) in `.obs[<col_num_umis>]`
        (with the "_original" suffix appended to original version(s) if
        needed) and dataframe(s) in `.uns["grna_info]` with columns:

            (a) of gRNA names => target genes (or control key),
            (b) of `col_guide_rna` and `col_num_umis` string entries
            split by `feature_split` into lists,
            (c) filtered based on the specified criteria (including,
            optionally, dropping multi-transfected cells), and
            (d) re-joined into strings separated by `feature_split`.

            List versions of columns have a "_list" suffix added to
            the column name. Original, un-filtered versions
            have (an additional) suffix "_all" added. If
            multiply-transfected cells are removed, a full version of
            the dataframe can be found in `.uns["grna_info_all]`.

            Additionally, `.uns["grna_feats_n"]` contains the long
            format (i.e., each row corresponds to a target gene
            within barcode/cell) with columns for total cell count
            ("t"), guide count for that target gene ("n"), and percent
            ("%"). Percentages and total counts are not re-calculated
            upon filtering.

            Finally, guide RNA counts are removed from the gene
            expression matrix.

    Notes:
        FUTURE DEVELOPERS: The Crispr class object
        depends on names of the columns created in this function.
        If they are changed (which should be avoided), be sure to
        change throughout the package.
    """
    ann, kws_pga = adata.copy(), copy.deepcopy(kws_process_guide_rna)
    print(f"\n\n<<< PERFORMING gRNA PROCESSING & FILTERING >>>\n\n{kws_pga}")
    if guide_split is None:
        guide_split = "$"
    if key_control_patterns is None:
        key_control_patterns = [key_control]
    if isinstance(key_control_patterns, str):
        key_control_patterns = [key_control_patterns]  # ensure iterable
    if col_guide_rna_new is None:
        col_guide_rna_new = f"{col_guide_rna}_new"

    # Filter by Guide Counts

    if filter_kws is not None:  # process & filter
        tg_info, feats_n, filt = filter_by_guide_counts(
            ann, col_guide_rna, col_num_umis, col_condition=col_guide_rna_new,
            file_perturbations=file_perturbations, guide_split=guide_split,
            feature_split=feature_split, key_control=key_control,
            key_control_patterns=key_control_patterns, **filter_kws)
    else:  # just process
        tg_info, feats_n = get_guide_info(
            ann, col_guide_rna, col_num_umis=col_num_umis,
            col_condition=col_guide_rna_new, key_control=key_control,
            key_control_patterns=key_control_patterns,
            file_perturbations=file_perturbations,
            guide_split=guide_split, feature_split=feature_split
            )  # process (e.g., multi-probe names, sum & average UMIs)


    if remove_multi_transfected is True:  # remove multi-transfected
        tg_info = tg_info.dropna(subset=[f"{col_guide_rna}_list_filtered"])
        tg_info = tg_info.join(tg_info[
            f"{col_guide_rna}_list_filtered"].apply(
                lambda x: np.nan if not isinstance(x, list) and pd.isnull(
                    x) else "multi" if len(x) > 1 else "single").to_frame(
                        "multiple"))  # multiple- or single-transfected
    for x in [col_num_umis, col_guide_rna, col_guide_rna_new]:
        if f"{x}_original" in ann.obs:
            warn(f"'{x}_original' already in adata. Dropping.")
            print(ann.obs[[f"{x}_original"]])
            ann.obs = ann.obs.drop(f"{x}_original", axis=1)
    ann.obs = ann.obs.join(tg_info[
        f"{col_guide_rna}_list_all"].apply(
            lambda x: (kws_pga["feature_split"] if kws_pga[
                "feature_split"] else "").join(x) if isinstance(
                    x, (np.ndarray, list, set, tuple)) else x).to_frame(
                        col_guide_rna), lsuffix="_original"
                    )  # processed full gRNA string without guide_split...
    ann.obs = ann.obs.join(tg_info[col_num_umis + "_filtered"].to_frame(
        col_num_umis), lsuffix="_original")  # filtered UMI (summed~gene)
    ann.obs = ann.obs.join(tg_info[col_guide_rna + "_filtered"].to_frame(
        col_guide_rna_new), lsuffix="_original")  # filtered gRNA summed~gene
    nobs = copy.copy(ann.n_obs)

    # Remove Multiply-Transfected Cells (optionally)
    if remove_multi_transfected is True:
        print(ann.obs)
        print("\n\n\t*** Removing multiply-transfected cells...")
        ann.obs = ann.obs.join(tg_info[["multiple"]], lsuffix="_original")
        ann = ann[ann.obs.multiple != "multi"]
        print(f"Dropped {nobs - ann.n_obs} out of {nobs} observations "
              f"({round(100 * (nobs - ann.n_obs) / nobs, 2)}" + "%).")
        ann.obs = ann.obs.drop("multiple", axis=1)

    # Remove Filtered-Out Cells
    print("\n\n\t*** Removing filtered-out cells...")
    nno = ann.obs.shape[0]
    if any(ann.obs[col_guide_rna_new].isnull()):
        miss = ann.obs[col_guide_rna_new].isnull()
        print(f"Dropping {round(miss.mean() * 100, 2)} ({miss.mean()}" + "%)"
              f" cells without (eligible) guide RNAs of {nno} observations.")
        ann = ann[~ann.obs[col_guide_rna_new].isnull()]  # drop cells w/o gRNA

    # Remove Guide RNA Counts from Gene Expression Matrix
    k_i = [key_control]
    if not remove_multi_transfected:  # ignore multi-transfected in removal
        k_i += list(pd.Series([
            x if kws_pga["feature_split"] in x else np.nan
            for x in ann.obs[col_guide_rna].unique()]).dropna())
    ann.uns["grna_keywords"] = str(kws_pga)  # store keyword arguments
    ann.uns["grna_feats_n"] = feats_n.reset_index(1)  # avoid h5ad write issue
    # ann.uns["grna_tg_info"] = tg_info
    ann.obs = ann.obs.assign(guide_split=kws_pga["guide_split"])
    return ann


def get_guide_info(adata, col_guide_rna, col_num_umis, col_condition=None,
                   file_perturbations=None,
                   feature_split="|", guide_split="-",
                   key_control_patterns=None, key_control="Control"):
    """
    Map guide IDs to perturbation conditions & sum UMIs.

    Returns:
        pandas.DataFrame: A dataframe (a) with sgRNA names replaced
            under their target gene categories (or control) and
            (b) with `col_guide_rna` and `col_num_umis` column entries
            (strings) grouped into lists (new columns with suffix
             "list_all"). Note that the UMI counts are summed across
            sgRNAs targeting the same gene within a cell. Also versions
            of the columns (and the corresponding string versions)
            filtered by the specified criteria (with suffixes
            "_filtered" and "_list_filtered" for list versions).

    Notes:
        FUTURE DEVELOPERS: The Crispr class object initialization
        depends on names of the columns created in this function.
        If they are changed (which should be avoided), be sure to
        change throughout the package.

    Examples:
    >>> kws = dict(max_pct_control_drop=75, min_pct_avg_n=40,
    ...            min_n_target_control_drop=100,
    ...            min_pct_dominant=80, drop_multi_control=False,
    ...            feature_split="|", guide_split="-")
    """
    # Find Gene Targets & Counts of Guides - Long Data by Cell & Perturbation
    tg_info = adata.obs[[col_guide_rna, col_num_umis]].apply(
        lambda y: y.apply(lambda x: x if feature_split is None or pd.isnull(
            x) else x.split(feature_split)).explode())  # guide list -> rows
    if file_perturbations is None:  # find perturbation name given guide_split
        tg_info.loc[:, col_condition] = tg_info[col_guide_rna].apply(
            lambda x: x if guide_split is None or pd.isnull(x) else
            key_control if any((i in x) for i in key_control_patterns) else
            guide_split.join(x.split(guide_split)[:-1]))
        # condition = target genes with all control guides = key_control
        # give custom mapping if want different
        # (e.g., if isoforms s.t. condition is more specific than target gene)
        # only drop string after last separator (e.g., keep -B in HLA-B-1
        # if guide_split = "-"; so you shouldn't put "-" w/i guide ID suffix)
    else:  # custom mapping: ID -> condition if in df else use guide_split
        read = None if isinstance(file_perturbations, pd.DataFrame) else \
            pd.read_csv if os.path.splitext(file_perturbations)[
                1] == ".csv" else pd.read_excel  # to read/use mapping data
        perts = (read(file_perturbations) if read else file_perturbations
                 ).drop_duplicates().set_index(col_guide_rna)
        tg_info = tg_info.join(tg_info.groupby(col_guide_rna).apply(
            lambda x: perts[col_condition].loc[x.name] if (
                x.name in perts.index) else x.name if pd.isnull(x.name) or (
                    guide_split is None) else guide_split.join(x.name.split(
                        guide_split)[:-1])).to_frame(col_condition),
                               on=col_guide_rna)  # mapping
    tg_info.loc[:, col_num_umis] = tg_info[col_num_umis].astype(float)

    # Drop NaN Guides
    ndrop = tg_info[col_guide_rna].isnull().sum()  # number dropped
    warn(f"NaNs present in guide RNA column ({col_guide_rna}). "
         f"Dropping {ndrop} out of {tg_info.shape[0]} rows.")
    tg_info = tg_info[~tg_info[col_guide_rna].isnull()]  # drop NaNs

    # Sum Up gRNA UMIs & Calculate Percentages
    feats_n = tg_info.rename_axis("bc").set_index(col_condition, append=True)[
        col_num_umis].groupby(["bc", col_condition]).sum().rename_axis([
            "bc", "g"])  # n = sum w/i-cell of UMIs by perturbation condition
    feats_n = feats_n.to_frame("n").join(feats_n.groupby(
        "bc").sum().to_frame("t"))  # t = sum all of gRNAs in cell
    feats_n = feats_n.assign(p=feats_n.n / feats_n.t * 100)  # to %age
    feats_n = feats_n.join(feats_n.groupby("bc").apply(
        lambda x: x.shape[0]).to_frame("num_transfections"))
    feats_n = feats_n.join(feats_n.n.groupby("bc").mean().to_frame("avg"))
    return tg_info, feats_n


def filter_by_guide_counts(adata, col_guide_rna, col_num_umis,
                           col_condition=None, file_perturbations=None,
                           key_control_patterns=None, key_control="Control",
                           feature_split="|", guide_split="-",
                           min_pct_control_keep=100,
                           max_pct_control_drop=0, min_pct_avg_n=None,
                           min_n_target_control_drop=None,
                           remove_contaminated_control=False,
                           min_pct_dominant="highest", min_n=5, **kwargs):
    """
    Filter processed guide RNA names (wraps `detect_guide_targets`).

    Returns:
        pandas.DataFrame: A dataframe (a) with sgRNA names replaced
            under their target gene categories (or control) and
            (b) with `col_guide_rna` and `col_num_umis` column entries
            (strings) grouped into lists (new columns with suffix
             "list_all"). Note that the UMI counts are summed across
            sgRNAs targeting the same gene within a cell. Also versions
            of the columns (and the corresponding string versions)
            filtered by the specified criteria (with suffixes
            "_filtered" and "_list_filtered" for list versions).

    Notes:
        FUTURE DEVELOPERS: The Crispr class object initialization
        depends on names of the columns created in this function.
        If they are changed (which should be avoided), be sure to
        change throughout the package.

    Examples:
    >>> kws = dict(max_pct_control_drop=75, min_pct_avg_n=40,
    ...            min_n_target_control_drop=100,
    ...            min_pct_dominant=80, drop_multi_control=False,
    ...            feature_split="|", guide_split="-")
    """
    # Extract Guide RNA Information
    tg_info, feats_n = get_guide_info(
        adata, col_guide_rna, col_num_umis, col_condition=col_condition,
        file_perturbations=file_perturbations,
        feature_split=feature_split, guide_split=guide_split,
        key_control_patterns=key_control_patterns, key_control=key_control)
    feats_n = feats_n.join(feats_n.groupby("bc").apply(lambda x: len(
        x.reset_index().g.unique())).to_frame("num_transfections"),
                           lsuffix="_o")  # number of transfections
    filt = feats_n.assign(
        num_transfections_original=feats_n.num_transfections).copy()

    # 1. If there are two gene targets and one of the targets is a control
    #   (A) If control is < 75% of total gRNA UMI, the control can be dropped.
    #          Cell is single transfected for the gene.
    #   (B) If the control >= 75% of total gRNA UMI, but less than
    #          100% (or whatever `min_pct_control_keep` is) if you want to
    #          allow slightly contaminated control-dominant cells, the cell is
    #          multiply transfected. Cell should be removed.
    if max_pct_control_drop not in [None, False]:
        old_ix = filt.index
        ctrl_gene_drop = filt[(filt.num_transfections == 2) & (
            filt.p < max_pct_control_drop)].loc[
                :, [key_control], :].index.values  # pseudo-singly-transfected
        filt = filt.drop(ctrl_gene_drop)
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "1A"
    if max_pct_control_drop not in [None, False]:
        old_ix = filt.index
        ctrl_cell_drop = filt[(filt.num_transfections == 2) & (
            filt.p >= max_pct_control_drop) & (
                filt.p < min_pct_control_keep)].reset_index(
                    "g").index.values  # drop "contaminated" control-dominated
        filt = filt.drop(ctrl_cell_drop)
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "1B"
    filt = filt.join(filt.groupby("bc").apply(lambda x: len(x.reset_index(
        ).g.unique())).to_frame("num_transfections"),
                     lsuffix="_pre_ctrl_drop")  # new # transfections

    # 2. If there are two gene target perturbations and there is no control
    #     (A) If the one gene UMI >= 80% of total gRNA UMI, that gene is
    #            considered dominant. The cell should be labeled as
    #            singly-transfected for the dominant gene.
    #     (B) If no gene UMI >= 80%, the cell is multiple transfected.
    #            Cell should be removed.
    if min_pct_dominant not in [None, False]:
        old_ix = filt.index
        filt = filt.groupby("bc").apply(
            lambda x: x if key_control in list(x.reset_index()["g"]) or (
                x.shape[0] != 2) else x[x.n >= min_pct_dominant]
            ).reset_index(0, drop=True)
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "2"

    # 3. If there are > 2 gene targets (may or may not include control)
    #     First perform an initial filter to remove low gRNA UMI.
    #     Calculate the average per-condition gRNA UMI. For genes whose gRNA
    #     UMI <40% of average sgRNA UMI, these genes can be dropped.
    if min_pct_avg_n not in [None, False] and any(
            filt.num_transfections_original > 2):
        old_ix = filt.index
        filt = filt.join(feats_n.n.groupby("bc").mean().to_frame(
            "avg_post"))  # 3Ai
        filt = filt[(filt.n >= (min_pct_avg_n / 100) * filt.avg_post) | (
            filt.num_transfections_original < 3)]  # 3Bi
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "3"

    # Re-Calculate Remaining Number of Transfections
    filt = filt.join(filt.groupby("bc").apply(lambda x: len(x.reset_index(
        ).g.unique())).to_frame("num_transfections"),
                     lsuffix="_pre_min_pct_avg")  # new # transfections

    # 4: After initial filter:
    #       (A) If only control remains, the cell is multiply-transfected.
    #            Cell should be removed.
    #    If more than one gene target (>=2) remains:
    #       (B) The control can be dropped if present.
    #       (C) If >= 3 genes remain, and the one gene UMI >= 80% of the
    #              remaining total gRNA UMI (recalculate total gRNA UMI
    #              after all preceding steps, including removal of control
    #              if present), that gene is considered dominant. The cell
    #              should be labeled as single transfected for the
    #              dominant gene, otherwise it is multiple transfected.
    #       (D) For all other cases (including gene # = 2),
    #              the cell is multiple transfected and should be removed.

    if remove_contaminated_control:
        ctrl = filt.loc[:, key_control, :]
        old_ix = filt.index
        filt = filt.drop(ctrl[(ctrl.num_transfections == 1) & (
            ctrl.num_transfections_original >= 2)].index)  # 3Aii
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "4A"
        old_ix = filt.index
        filt = filt.drop(filt.loc[:, key_control, :][filt.loc[
            :, key_control, :].num_transfections >= 2].index)  # 3Cii
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "4B"
    filt = filt.join(filt.groupby("bc").apply(
        lambda x: x.shape[0]).to_frame("num_transfections"),
                        lsuffix="_pre_dominant_3")  # new # of transfections
    if min_pct_dominant not in [None, False] and any(
            filt.num_transfections > 2):
        old_ix = filt.index
        filt = filt.join(filt.n.groupby("bc").sum().to_frame("t_remaining"))
        filt.loc[:, "p_remaining"] = filt.n / filt.t_remaining
        filt = filt[(filt.num_transfections <= 2) | (
            filt.p_remaining >= min_pct_dominant)]
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "4C"

    # Finally, Drop Unless Singly- or Pseudo-Singly-Tranfected
    old_ix = filt.index
    filt = filt.join(filt.groupby("bc").apply(
        lambda x: len(x.reset_index().g.unique())).to_frame(
            "num_transfections"), lsuffix="_pre_final")  # new # transfections
    filt = filt[filt.num_transfections == 1]
    feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "4D"
    return tg_info, feats_n, filt


def process_guide_rna_custom(adata, file_perturbations,
                             col_guide_rna, col_condition, col_target_genes,
                             col_num_umis, key_control=None,
                             guide_split=None, feature_split=None,
                             **kws_process_guide_rna):
    """
    Process guide RNAs when using a custom file mapping.

    Returns:
        tuple: Tuple of (filtered `AnnData`, unfiltered `AnnData`)
            Filtered and processed `AnnData` object and the unfiltered
            (i.e., all rows present) `AnnData` object with certain
            metadata from `file_perturbations` added.

    Notes:

        Process guide RNAs using a custom file mapping guide IDs to
        perturbation condition to target gene. A common use
        case is if, when calculating filtering conditions (e.g.,
        percent of UMI counts, total # of UMIs across guide IDs
        corresponding to the same perturbation condition),
        you want to group by a perturbation condition column
        (`col_condition`) that is different from the
        target genes (`col_target_genes`), such as isoforms
        (e.g., consider MARK3-P1 & MARK3-P2 different rather
        than both in the condition targeting MARK3) and/or if you have
        issues with auto-detecting target genes (e.g., if you have "-"
        as your `guide_split` but also in gene names, such as HLA-B).
        `file_perturbations` should be a dataframe or .xlsx/.csv file
        with the index (if a dataframe) or first column (if a file) as
        the individual guide IDs (most specific),
        `col_condition` as the column you want to group by for
        filtering, and `col_target_genes` as the column mapping those
        to the target gene (most general, entries
        should all be in `adata.var_names`).

    Example:

        >>> DIR = "~/corescpy/examples/data/crispr-screening"
        >>> file_arg = {"HH03": dict(directory=f"{DIR}/HH03")}
        >>> file_pert = f"{DIR}/04172024_CRISPRi_IBD_guides.xlsx"
        >>> kws_pga = {
        ...    "feature_split": "|", "guide_split": "-",
        ...    "drop_multi_control": False,
        ...    "key_control_patterns": ["CTRL"],
        ...    "remove_multi_transfected": True,
        ...    "min_n_target_control_drop": None,
        ...    "max_pct_control_drop": 75, "min_pct_dominant": 80,
        ...    "min_pct_avg_n": 40, "min_n": 5}
        >>> adata = cr.pp.create_object(file_arg)  # initial object
        >>> adata, adata_unfiltered = cr.pp.process_guide_rna_custom(
        ...    adata, file_pert, "feature_call", "perturbation",
        ...    "target_gene", "num_umis", key_control="NT", **kws_pga)
    """
    ann = adata.copy()
    perts = file_perturbations if isinstance(
        file_perturbations, pd.DataFrame) else pd.read_excel(
            file_perturbations, index_col=0) if os.path.splitext(
                file_perturbations)[1] == ".xlsx" else pd.read_csv(
                    file_perturbations, index_col=0)  # perturbations data
    # col_id = perts.columns.difference([col_guide_rna])[0]
    col_id = col_condition
    if ann.obs[col_guide_rna].isnull().any():
        warn(f"Dropping NaN's from {col_guide_rna} column")
        ann = ann[~ann.obs[col_guide_rna].isnull()]  # drop NaN guide IDs
    grnas = ann.obs[col_guide_rna].dropna().apply(
        lambda x: feature_split.join([perts.loc[i][col_id] if (
            i in perts.index) else guide_split.join(i.split(
                guide_split)[:-1]) if guide_split is not None and (
                    guide_split in i) else i
                for i in x.split(feature_split)])).to_frame(col_guide_rna)
    grnas = grnas.join(ann.obs[col_guide_rna].dropna().apply(
        lambda x: feature_split.join([perts.loc[i][col_target_genes] if (
            i in perts.index) else guide_split.join(
                i.split(guide_split)[:-1]) if (
                    guide_split is not None) and guide_split in i else i
                for i in x.split(feature_split)])).to_frame(col_target_genes))
    grnas = grnas.join(ann.obs[col_guide_rna].to_frame(
        col_guide_rna + "_flat_ix"))
    ann_unfilt = ann.copy()  # to hold un-filtered object but w/ gRNA columns
    ann_unfilt.obs = ann_unfilt.obs.join(grnas, lsuffix="_original")
    kws_pga = cr.tl.merge({
        "col_guide_rna": col_guide_rna, "col_num_umis": col_num_umis,
        "key_control": key_control, "col_guide_rna_new": col_condition,
        "feature_split": feature_split, "guide_split": guide_split},
                          kws_process_guide_rna)   # processing rguments
    grnas = grnas.join(grnas[col_guide_rna].apply(lambda x: x.split(
        feature_split)).to_frame(col_guide_rna + "_list"))
    ann.obs = ann.obs.join(grnas[perts.columns], lsuffix="_original")

    ann = cr.pp.process_guide_rna(ann, **kws_pga, tg_info=grnas)  # run wrap

    ann.obs = ann.obs.join(ann.obs[col_condition].dropna().apply(
        lambda x: feature_split.join([perts[perts[
            col_condition] == i][col_target_genes].iloc[0] if i in list(
                perts[col_condition]) else guide_split.join(i.split(
                    guide_split)[:-1]) if guide_split in i else i
                for i in x.split(feature_split)])).to_frame(
                    col_target_genes))
    return ann, ann_unfilt

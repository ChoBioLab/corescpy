import os
from warnings import warn
import copy
import pandas as pd


def process_guide_rna(adata, col_guide_rna="feature_call",
                      col_guide_rna_new="perturbation",
                      col_target_genes=None,
                      col_num_umis="num_umis",
                      feature_split=None, guide_split=None,
                      file_perturbations=None,
                      key_control_patterns=None, key_control="NT",
                      remove_multi_transfected=None, kws_filter=None):
    """
    Process and filter guide RNAs, (optionally) remove cells considered
    multiply-transfected (after filtering criteria applied), and remove
    guide counts from the gene expression matrix (wrapper function).

    Args:
        adata (AnnData): AnnData object (RNA assay,
            so if multi-modal, subset before passing to this argument).
        col_guide_rna (str): Name of the column containing guide IDs.
        col_guide_rna_new (str): Desired name for the column in which
            to store the processed/assigned guide ID-perturbation
            condition mapping (or, if `file_perturbations` is
            specified, the column in the dataframe/file containing
            this information). This argument translates to
            `col_condition` and/or `col_target_genes` in other
            areas (e.g., functions arguments, class attributes).
        col_num_umis (str): Column with the UMI counts (string entries
            with (for designs with possible multiple-transfection)
            within-cell probes separated by `feature_split` and
            any probe ID suffixes at the end following `guide_split`
            (e.g., '-' if STAT1-1-2, STAT-1-1-4, etc.).
        col_target_genes (str, optional): Name of the column
            containing gene symbols (if None, will be inferred
            from `col_guide_rna`). Defaults to None.
        file_perturbations (str or DataFrame): Path to a file
            containing perturbations (e.g., cell-type perturbations)
            or a DataFrame containing perturbations.
        feature_split (str, optional): For designs with multiple
            guides, the character that splits guide names in
            `col_guide_rna`. For instance, "|" for
            `STAT1-1|CNTRL-1|CDKN1A`. Defaults to None (only one guide
            per cell).
        guide_split (str, optional): The character that separates
            guide (rather than gene target)-specific IDs within gene.
            For instance, guides targeting STAT1 may include
            STAT1-1, STAT1-2, etc.; the argument would be "-"
            so the function can identify all of those as
            targeting STAT1. It will only remove the
            last "<guide_split>..." found (e.g., STAT1-2-3 -> STAT1-2).
            This protects against situations where a dash is used as
            the ID separator but also in a gene name (e.g., HLA-B).
            To specify a custom mapping from guide ID to perturbation
            condition to target gene, specify `file_perturbations`.
            Defaults to None (i.e., all guides assumed to be same as
            perturbation condition which is the same as target gene).
        key_control_patterns (list, optional): List (or single string)
            of patterns in guide RNA column entries that correspond to
            a control. For instance, if control entries in the original
            `col_guide_rna` column include `NEGCNTRL` and
            `Control.D`, you should specify ['Control', 'CNTRL']
            (assuming no non-control sgRNA names contain
            those strings). If blank entries should be interpreted as
            control guides, first convert them to strings that can be
            specified in this argument.
        key_control (str, optional): The name you want the control
            entries to be categorized as under the new `col_guide_rna`.
            for instance, `CNTRL-1`, `NEGCNTRL`, etc. would all be
            replaced by "Control" if that's what you specify here.
            Defaults to "Control".
        kws_filter (dict, optional): Keyword arguments for filtering
            (see `filter_by_guide_counts()` for description of the
            arguments that can be specified here, i.e., the ones in
            that function that don't overlap with the ones explicitly
            specified in this function, such as `col_guide_rna`).
            Defaults to None (will just run `get_guide_info()` to
            return information such as summed guide UMI counts).
        remove_multi_transfected (bool, optional): If True, remove
            cells determined as multi-transfected according to
            criteria specified in `kws_filter`. If False,
            multi-transfected cells with have NaNs in the
            `col_guide_rna_new` column. If None, True if `kws_filter`
            is not None; otherwise, False (if unspecified).
            Defaults to None (will infer from `kws_filter`).

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
    print(f"\n\n<<< PERFORMING gRNA PROCESSING/FILTERING >>>\n\n{kws_filter}")
    ann = adata.copy()
    ann.raw = adata.copy()
    if guide_split is None:
        guide_split = "$"
    if key_control_patterns is None:
        key_control_patterns = [key_control]
    if isinstance(key_control_patterns, str):
        key_control_patterns = [key_control_patterns]  # ensure iterable
    if col_guide_rna_new is None:
        col_guide_rna_new = f"{col_guide_rna}_condition"
    if col_target_genes == col_guide_rna_new:
        col_target_genes = None  # don't consider separately if same
    if remove_multi_transfected is None:
        remove_multi_transfected = kws_filter is not None

    # Filter by Guide Counts
    if kws_filter is not None:  # process & filter
        tg_info, feats_n, filt, perts = filter_by_guide_counts(
            ann, col_guide_rna, col_num_umis, col_condition=col_guide_rna_new,
            file_perturbations=file_perturbations, guide_split=guide_split,
            feature_split=feature_split, key_control=key_control,
            key_control_patterns=key_control_patterns, **kws_filter)
    else:  # just process
        tg_info, feats_n, perts = get_guide_info(
            ann, col_guide_rna, col_num_umis=col_num_umis,
            col_condition=col_guide_rna_new, key_control=key_control,
            key_control_patterns=key_control_patterns,
            file_perturbations=file_perturbations,
            guide_split=guide_split, feature_split=feature_split
            )  # process (e.g., multi-probe names, sum & average UMIs)
    cols_fl = ["n", "t", "p"] + [col_target_genes if col_target_genes else []]
    filt_flat = filt.rename_axis(["bc", col_guide_rna_new])
    if col_target_genes is None:
        if perts is not None:
            tgs = perts.reset_index().set_index(
                col_guide_rna_new)[col_target_genes]  # condition-gene mapping
            filt_flat = filt_flat.join(filt_flat.groupby(
                col_guide_rna_new).apply(lambda x: tgs.loc[x.name].unique()[
                    0] if x.name in tgs.index else x.name).to_frame(
                        col_target_genes))  # join custom target gene column
        else:
            warn(f"Can't create target genes column ({col_target_genes}) "
                 "if `file_perturbations` not specified. Setting equal to"
                 f"the perturbation condition column ({col_guide_rna_new}).")
            filt_flat.loc[:, col_target_genes] = filt_flat[col_guide_rna_new]
    filt_flat = filt_flat[cols_fl].reset_index(1).rename({
        "n": col_num_umis, "t": "total_umis_cell", "p": "percent_umis"
        }, axis=1).astype(str).apply(lambda y: y.groupby("bc").apply(
            lambda x: x if feature_split is None else feature_split.join(
                x.to_list())))  # flatten to 1 row/cell
    tg_cgrna_flat = tg_info[[col_guide_rna, col_num_umis]].rename_axis(
        "bc").astype(str).apply(lambda y: y.groupby("bc").apply(
            lambda x: x if feature_split is None else feature_split.join(
                x.to_list())))  # one row per cell; unique guide IDs
    for x in [col_num_umis, col_guide_rna, col_guide_rna_new,
              col_guide_rna + "_processed"]:
        if f"{x}_original" in ann.obs:
            warn(f"'{x}_original' already in adata. Dropping.")
            print(ann.obs[[f"{x}_original"]])
            ann.obs = ann.obs.drop(f"{x}_original", axis=1)
    ann.obs = ann.obs.join(tg_cgrna_flat, rsuffix="_processed").join(
        filt_flat, lsuffix="_original")  # join processed/filtered columns
    if remove_multi_transfected is True:  # remove multi-transfected
        ann.raw = ann.copy()
        nobs = copy.copy(ann.n_obs)
        ann = ann[~ann.obs[col_guide_rna_new].isnull()]
        print(f"Dropped {nobs - ann.n_obs} out of {nobs} observations "
              f"({round(100 * (nobs - ann.n_obs) / nobs, 2)}" + "%)"
              " during guide RNA filtering.")
    ann.uns["kws_filter"] = str(kws_filter)  # store keyword arguments
    ann.uns["grna_feats_n"] = feats_n.reset_index(1)  # avoid h5ad write issue
    ann.obs = ann.obs.assign(guide_split=guide_split).assign(
        feature_split=feature_split)
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
        perts = None
    else:  # custom mapping: ID -> condition if in df else use guide_split
        read = None if isinstance(file_perturbations, pd.DataFrame) else \
            pd.read_csv if os.path.splitext(file_perturbations)[
                1] == ".csv" else pd.read_excel  # to read/use mapping data
        perts = (read(file_perturbations) if read else file_perturbations
                 ).drop_duplicates().set_index(col_guide_rna)
        # TODO: Add error if non-unique mapping to condition or target gene
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
    feats_n = feats_n.join(feats_n.groupby("bc").apply(lambda x: len(
        x.reset_index().g.unique())).to_frame("num_transfections"))
    feats_n = feats_n.join(feats_n.n.groupby("bc").mean().to_frame("avg"))
    return tg_info, feats_n, perts


def filter_by_guide_counts(adata, col_guide_rna, col_num_umis,
                           col_condition=None, file_perturbations=None,
                           key_control_patterns=None, key_control="Control",
                           feature_split="|", guide_split="-",
                           min_pct_control_keep=100,
                           max_pct_control_drop=None, min_pct_avg_n=None,
                           min_n_target_control_drop=None,
                           remove_contaminated_control=False,
                           min_pct_dominant=None, min_n=0, **kwargs):
    """
    Filter processed guide RNA names (wraps `detect_guide_targets`).

    Args:
        adata
        min_n (int, optional): Minimum number of counts to retain a
            guide (performed after all other filtering).
            The default is 0.
        max_pct_control_drop (float, optional): In a cell transfected
            for exactly one distinct (i.e., one perturbation condition)
            non-control guide and one control guide
            (exactly two distinct guies), if
            control UMI counts are less than or equal to this
            percentage of the total counts for that cell, and if a
            non-control sgRNA is also present and meets other filtering
            criteria, then consider that cell pseudo-single-transfected
            (non-control gene). If a cell has exactly two distinct
            perturbation condition guides, one of which is control and
            has less than `min_pct_control_keep`, the whole cell will
            be dropped as a "contaminated control-dominant" cell.
            Defaults to None (will not filter on this criterion).
        min_pct_control_keep (float, optional): In a cell transfected
            for exactly one distinct (i.e., one perturbation condition)
            non-control guide and one control guide
            (exactly two distinct guies), the minimum percentage
            of guide UMIs needed to be made up by a control guide in
            a cell transfected for the cell to be considered
            pseudo-singly-transfected for control. If 100%, controls in
            this situation will always be dropped, and the whole cell
            will be dropped if control count is
            > `max_pct_control_drop` but <= `min_pct_control_keep`.
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
        min_pct_avg_n (int, optional): sgRNAs in cells transfected for
            exactly two distinct (of different perturbation
            conditions) non-control guides with counts below this
            percentage of the average UMI count (as re-calculated
            after performing initial control filtering) will be
            considered noise and dropped from the list of genes for
            which the cell is considered transfected. Defaults to None.
        min_pct_dominant (int, optional): sgRNAs with counts at or
            above this percentage of the cell total UMI count will be
            considered dominant, and all other guides will be dropped
            from the list of genes for whichmthat cell is considered
            transfected. Defaults to "highest" (will choose most
            abundant guide as the dominant guide).

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
    tg_info, feats_n, perts = get_guide_info(
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

    # 2. If there are 2 gene target perturbations and there is no control
    #     (A) If the one gene UMI >= (min_pct_dominant)% of total gRNA UMI,
    #             that gene is considered dominant. The cell should be labeled
    #             as singly-transfected for the dominant gene.
    #     (B) If no gene UMI >= (min_pct_dominant)%, the cell is
    #             multiply-transfected. Cell should be removed.
    if min_pct_dominant not in [None, False]:
        old_ix = filt.index
        filt = filt.groupby("bc").apply(
            lambda x: x if key_control in list(x.reset_index()["g"]) or (
                len(x.reset_index().g.unique()) != 2) else
            x[x.p >= min_pct_dominant]).reset_index(0, drop=True)
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "2"

    # 3. If there are >= 3 gene targets (may or may not include control)
    #     (A) First perform an initial filter to remove low gRNA UMI.
    #     (i) Calculate the average per-condition gRNA UMI.
    #     (B) Drop genes whose gRNA UMI <(min_pct_avg_n)% of average UMI.
    if min_pct_avg_n not in [None, False] and any(
            filt.num_transfections_original > 2):
        old_ix = filt.index
        filt = filt.join(feats_n.n.groupby("bc").mean().to_frame(
            "avg_post"))  # 3i (re-calculate average cell UMI)
        filt = filt[(filt.n >= (min_pct_avg_n / 100) * filt.avg_post) | (
            filt.num_transfections_original < 3)]  # 3B
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "3B"
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
            filt.num_transfections >= 3):
        old_ix = filt.index
        filt = filt.join(filt.n.groupby("bc").sum().to_frame("t_remaining"))
        filt.loc[:, "p_remaining"] = 100 * filt.n / filt.t_remaining
        filt = filt[(filt.num_transfections <= 2) | (
            filt.p_remaining >= min_pct_dominant)]
        feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "4C"

    # Finally, Keep Singly-/Pseudo-Singly-Tranfected; Enforce Minimum UMI
    old_ix = filt.index
    filt = filt.join(filt.groupby("bc").apply(
        lambda x: len(x.reset_index().g.unique())).to_frame(
            "num_transfections"), lsuffix="_pre_final")  # new # transfections
    filt = filt[filt.num_transfections == 1]
    filt = filt[filt.n >= min_n]
    feats_n.loc[old_ix.difference(filt.index), "reason_drop"] = "4D"
    return tg_info, feats_n, filt, perts

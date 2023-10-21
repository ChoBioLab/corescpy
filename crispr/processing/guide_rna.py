import re
from  warnings import warn
import copy
import crispr as cr
import pandas as pd
import numpy as np


def process_guide_rna(adata, col_guide_rna="guide_id", 
                      col_guide_rna_new="condition", 
                      col_num_umis="UMI count",
                      key_control="NT", 
                      conserve_memory=False,
                      remove_multi_transfected=False,
                      **kws_process_guide_rna):
    """
    Process and filter guide RNAs, (optionally) remove cells considered 
    multiply-transfected (after filtering criteria applied), and remove
    guide counts from the gene expression matrix (wrapper function).

    Args:
        adata (AnnData): AnnData object (RNA assay, 
            so if multi-modal, subset before passing to this argument).
        col_guide_rna (str): Name of the column containing guide IDs.
        col_num_umis (str): Column with the UMI counts (string entried, 
            with (for designs with possible multiple-transfection) 
            within-cell probes separated by `feature_split` and
            any probe ID suffixes at the end following `guide_split`.
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
            Set to 0 to ignore this filtering. Defaults to 75.
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
            transfected. Defaults to 80.
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
    print("\n\n<<< PERFORMING gRNA PROCESSING AND FILTERING >>>\n")
    ann, kws_pga = adata.copy(), copy.deepcopy(kws_process_guide_rna)
    
    # Filter by Guide Counts
    tg_info, feats_n = filter_by_guide_counts(
        ann, col_guide_rna, col_num_umis=col_num_umis, 
        key_control=key_control, **kws_pga
        )  # process (e.g., multi-probe names) & filter by # gRNA
    
    # Add Results to AnnData
    for x in ["feature_split", "guide_split"]:
        if x not in kws_pga:
            kws_pga[x] = None
    try:
        tg_info = tg_info.loc[ann.obs.index]
    except Exception as err:
        warn(f"{err}\n\nCouldn't re-order tg_info to mirror adata index!")
    tg_info_all = None  # fill later if needed
    if remove_multi_transfected is True:  # remove multi-transfected
        tg_info_all = tg_info.copy() if conserve_memory is False else None
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
            lambda x: kws_pga["feature_split"].join(x) if isinstance(
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
    print(f"\n\n\t*** Removing filtered-out cells...")
    ann = ann[~ann.obs[col_guide_rna_new].isnull()]
    print(f"Dropped {nobs - ann.n_obs} out of {nobs} observations "
          f"({round(100 * (nobs - ann.n_obs) / nobs, 2)}" + "%).")
    
    # Remove Guide RNA Counts from Gene Expression Matrix
    k_i = [key_control]
    if not remove_multi_transfected:  # ignore multi-transfected in removal
        k_i += list(pd.Series([
            x if kws_pga["feature_split"] in x else np.nan 
            for x in ann.obs[col_guide_rna].unique()]).dropna())
    ann = remove_guide_counts_from_gex(ann, col_guide_rna, key_ignore=k_i)
    ann.uns["grna_keywords"], ann.uns["grna_feats_n"] = kws_pga, feats_n
    ann.uns["grna_info"], ann.uns["grna_info_all"] = tg_info, tg_info_all
    ann.obs = ann.obs.assign(guide_split=kws_pga["guide_split"]
                             )  # make sure guide split in `.obs`
    return ann


def remove_guide_counts_from_gex(adata, col_target_genes, key_ignore=None):
    """Remove guide RNA counts from gene expression matrix."""
    guides = list(adata.obs[col_target_genes].dropna())  # guide names
    if key_ignore is not None:
        guides = list(set(guides).difference(
            [key_ignore] if isinstance(key_ignore, str) else key_ignore))
    guides_in_varnames = list(set(adata.var_names).intersection(set(guides)))
    if len(guides_in_varnames) > 0:
        print(f"\n\t*** Removing {', '.join(guides)} guides "
            "from gene expression matrix...")
        adata._inplace_subset_var(list(set(adata.var_names).difference(
            set(guides_in_varnames))))  # remove guide RNA counts
    return adata


def detect_guide_targets(col_guide_rna_series,
                         feature_split="|", guide_split="-",
                         key_control_patterns=None,
                         key_control="Control", **kwargs):
    """Detect guide gene targets (see `filter_by_guide_counts`)."""
    if kwargs:
        print(f"\nUn-Used Keyword Arguments: {kwargs}\n\n")
    if key_control_patterns is None:
        key_control_patterns = [
            key_control]  # if already converted, pattern=key itself
    if isinstance(key_control_patterns, str):
        key_control_patterns = [key_control_patterns]
    targets = col_guide_rna_series.str.strip(" ").replace("", np.nan)
    if key_control_patterns and pd.Series(
        key_control_patterns).isnull().any():  # if NAs = control sgRNAs
        targets = targets.replace(np.nan, key_control)  # NaNs -> control key
        key_control_patterns = list(pd.Series(key_control_patterns).dropna())
    else:  # if NaNs mean unperturbed cells
        if any(pd.isnull(targets)):
            warn(f"Dropping rows with NaNs in `col_guide_rna`.")
        targets = targets.dropna()
    if feature_split is not None or guide_split is not None:
        targets, nums = [targets.apply(
            lambda x: [re.sub(p, ["", r"\1"][j], str(i)) if re.search(
                p, str(i)) else [i, ""][j] for i in list(
                    x.split(feature_split) if feature_split  # if multi
                    else [x])]  # if single gRNA
            if p not in ["", None] else  # ^ if need to remove guide suffixes
            list(x.split(feature_split) if feature_split else [x] if p else p
                )  # ^ no suffixes: split x or [x] (j=0), "" for suffix (j=1)
            ) for j, p in enumerate(list(
                [f"{guide_split}.*", rf'^.*?{re.escape(guide_split)}(.*)$']
            if guide_split else [None, ""]))
                            ]  # each entry -> list of target genes
    if key_control_patterns:  # if need to search for control key patterns
        targets = targets.apply(
            lambda x: [i if i == key_control else key_control if any(
                    (k in i for k in key_control_patterns)) else i 
                for i in x])  # find control keys among targets
    # targets = targets.apply(
    #     lambda x: [[x[0]] if len(x) == 2 and x[1] == "" else x
    #     for i in x])  # in case all single-transfected
    grnas = targets.to_frame("t").join(nums.to_frame("n")).apply(
        lambda x: [i + str(guide_split if guide_split else "") + "_".join(
            np.array(x["n"])[np.where(np.array(x["t"]) == i)[0]]) 
                   for i in pd.unique(x["t"])],  # sum gRNA counts/gene target 
        axis=1).apply(lambda x: feature_split.join(x)).to_frame(
            "ID")  # e.g., STAT1-1|STAT1-2|NT-1-2 => STAT1-1_2 counts
    # DO NOT change the name of grnas["ID"]
    return targets, grnas


def filter_by_guide_counts(adata, col_guide_rna, col_num_umis, 
                           max_pct_control_drop=75,
                           min_n_target_control_drop=100,
                           min_pct_avg_n=40,
                           min_pct_dominant=80,
                           drop_multi_control=True,
                           feature_split="|", guide_split="-",
                           key_control_patterns=None,
                           key_control="Control", **kwargs):
    """
    Filter processed guide RNA names (wraps `detect_guide_targets`).

    Returns:
        pandas.DataFrame: A dataframe (a) with sgRNA names replaced 
            under their target gene categories (or control) and
            (b) with `col_guide_rna` and `col_num_umis` column entries
            (strings) grouped into lists 
            (new columns with suffix "list_all"). 
            Note that the UMI counts are summed across sgRNAs 
            targeting the same gene within a cell. Also versions of the 
            columns (and the corresponding string versions) 
            filtered by the specified  
            criteria (with suffixes "_filtered" and "_list_filtered" 
            for list versions).
            
    Notes:
        FUTURE DEVELOPERS: The Crispr class object initialization 
        depends on names of the columns created in this function. 
        If they are changed (which should be avoided), be sure to 
        change throughout the package.
    """
    # Extract Guide RNA Information
    ann = adata.copy()
    if guide_split is None:
        guide_split = "$"
    if key_control_patterns is None:
        key_control_patterns = [np.nan]
    guides = ann.obs[col_guide_rna].copy()  # guide names
    
    # If `guide_split` in Any Gene Names, Temporarily Substitute
    grs = None
    if guide_split is not None:
        split_char = [guide_split in g for g in ann.var_names]
        if any(split_char):
            grs = "==="
            bad_symb = np.array(ann.var_names)[np.where(split_char)[0]]
            if grs in guide_split:
                raise ValueError(f"{grs} is a reserved name and cannot be "
                                 "contained within `guide_split`.")
            warn(f"`guide_split` ({guide_split}) found in at least "
                 f"one gene name ({', '.join(bad_symb)}). Using {grs}. "
                 "as temporary substitute. Will attempt to replace later, "
                 "but note that there are risks in having a `guide_split` "
                 "as a character also found in gene names.")
            guides = guides.apply(lambda x: re.sub(bad_symb[np.where(
                [i in str(x) for i in bad_symb])[0][0]], re.sub(
                guide_split, grs, bad_symb[np.where(
                    [i in str(x) for i in bad_symb])[0][0]]), 
                str(x)) if any((i in str(x) for i in bad_symb)) else x)
    
    # Find Gene Targets & Counts of Guides
    targets, grnas = detect_guide_targets(
        guides, feature_split=feature_split, guide_split=guide_split,
        key_control_patterns=key_control_patterns, 
        key_control=key_control, **kwargs)  # target genes
    if grs is not None:  # if guide_split was in any gene name
        targets = targets.apply(lambda x: [
            re.sub(grs, guide_split, i) for i in x])  # replace grs in list
        grnas.loc[:, "ID"] = grnas["ID"].apply(
            lambda x: re.sub(grs, guide_split, str(x)))  # e.g., back to HLA-B
    tg_info = grnas["ID"].to_frame(
        col_guide_rna + "_flat_ix").join(
            targets.to_frame(col_guide_rna + "_list"))
    if col_num_umis is not None:
        tg_info = tg_info.join(ann.obs[[col_num_umis]].apply(
            lambda x: [float(i) for i in list(
                str(x[col_num_umis]).split(feature_split)
                if feature_split else [float(x[col_num_umis])])], 
            axis=1).to_frame(col_num_umis + "_list"))
        tg_info = tg_info.join(tg_info[col_num_umis + "_list"].dropna().apply(
            sum).to_frame(col_num_umis + "_total"))  # total UMIs/cell
    tg_info = ann.obs[col_guide_rna].to_frame(col_guide_rna).join(tg_info)
    if tg_info[col_guide_rna].isnull().any() and (~any(
        [pd.isnull(x) for x in key_control_patterns])):
        warn(f"NaNs present in guide RNA column ({col_guide_rna}). "
             f"Dropping {tg_info[col_guide_rna].isnull().sum()} "
             f"out of {tg_info.shape[0]} rows.")
        tg_info = tg_info[~tg_info[col_guide_rna].isnull()]

    # Sum Up gRNA UMIs
    cols = [col_guide_rna + "_list", col_num_umis + "_list"]
    feats_n = tg_info[cols].dropna().apply(lambda x: pd.Series(
        dict(zip(pd.unique(x[cols[0]]), [sum(np.array(x[cols[1]])[
            np.where(np.array(x[cols[0]]) == i)[0]]) for i in pd.unique(
                x[cols[0]])]))), axis=1).stack().rename_axis(["bc", "g"])
    feats_n = feats_n.to_frame("n").join(feats_n.groupby(
        "bc").sum().to_frame("t"))  # sum w/i-cell # gRNAs w/ same target gene 
    feats_n = feats_n.assign(p=feats_n.n / feats_n.t * 100)  # to %age
    
    # Other Variables
    feats_n = feats_n.join(feats_n.reset_index("g").groupby("bc").apply(
            lambda x: 0 if all(x["g"] == key_control) else x[
                x["g"] != key_control]["n"].sum()
            ).to_frame("n_non_ctrl"))  # overridden by dominant guide?
    feats_n = feats_n.join(feats_n.groupby(["bc", "g"]).apply(
        lambda x: "retain" if (x.name[1] != key_control) 
        else ("low_control" if float(x["p"]) <= max_pct_control_drop 
              else "high_noncontrol" if min_n_target_control_drop and float(
                  x["n_non_ctrl"]) >= min_n_target_control_drop
              else "retain")).to_frame("drop_control"))  # control drop labels
    feats_n = feats_n.join(feats_n.n.groupby("bc").mean().to_frame(
        "n_cell_avg"))  # average UMI count within-cell
    feats_n = feats_n.reset_index("g")
    feats_n = feats_n.assign(control=feats_n.g == key_control
                             )  # control guide dummy-coded column
    feats_n = feats_n.assign(target=feats_n.g != key_control).set_index(
        "g", append=True)  # target guide dummy-coded column
    if min_pct_dominant is not None:
        feats_n = feats_n.assign(dominant=feats_n.p >= min_pct_dominant)
        feats_n = feats_n.assign(
            dominant=feats_n.dominant & feats_n.target
            )  # only non-control guides considered dominant
    else:
        feats_n = feats_n.assign(dominant=False)  # no filtering based on this
    # CHECK: 
    # feats_n[feats_n.p >= min_pct_dominant].dominant.all()
    feats_n = feats_n.assign(
        low_umi=feats_n.n < feats_n.n_cell_avg * min_pct_avg_n / 100 
        if min_pct_avg_n is not None else False
        )  # low % of mean UMI count (if filtering based on that)?
    feats_n = feats_n.join(feats_n.dominant.groupby("bc").apply(
            lambda x: pd.Series(
                [True if (x.any()) and (i is False) else False for i in x], 
                index=x.reset_index("bc", drop=True).index)
            ).to_frame("drop_nondominant"))  # overridden by dominant guide?
    # CHECK: 
    # feats_n.loc[feats_n[(feats_n.p < min_pct_dominant) & (
    #     feats_n.drop_nondominant)].reset_index().bc.unique()].groupby(
    #         "bc").apply(lambda x: x.dominant.any()).all()
    # feats_n[(feats_n.p >= 80)].drop_nondominant.sum() == 0
    filt = feats_n.copy()  # start with full feats_n

    # Filtering Phase I (If Dominant Guide, Drop Others; Drop Low Controls)
    filt = filt[filt.drop_control.isin(["retain"])
                ]  # low control or high non-control = not control-transfected 
    filt = filt[~filt.drop_nondominant]  # drop if another guide dominates
    
    # Filtering Phase II (Filter Low Targeting Guides in Multiply-Transfected)
    filt = filt.join(filt.reset_index("g").g.groupby("bc").apply(
        lambda x: len(x[x != key_control].unique()) > 1).to_frame(
            "multi_noncontrol"))  # multi-transfected with non-control guides?
    filt = filt.assign(low_umi_multi=filt.low_umi & filt.multi_noncontrol)
    filt = filt[~filt.low_umi_multi]  # drop low guides, multi-NC-transfected
    
    # Filtering Phase III (Remove Control from Multi-Transfected)
    filt = filt.join(filt.reset_index("g").g.groupby("bc").apply(
        lambda x: "multi" if len(x.unique()) > 1 else "single").to_frame(
            "transfection"))  # after filtering, single or multi-guide?
    if drop_multi_control is True:  # drop control (multi-transfected cells)
        filt = filt.assign(multi=filt.transfection == "multi")
        filt = filt.assign(multi_control=filt.control & filt.multi)
        filt = filt[~filt.multi_control]  # drop
        filt = filt.drop(["multi", "multi_control"], axis=1)
    filt = filt.drop("transfection", axis=1)
    filt = filt.join(filt.reset_index("g").g.groupby("bc").apply(
        lambda x: "multi" if len(x.unique()) > 1 else "single").to_frame(
            "transfection"))  # after filtering, single or multi-guide?
    
    # Join Counts/%s/Filtering Categories w/ AnnData-Indexed Guide Information
    filt = filt.n.to_frame("u").groupby("bc").apply(
        lambda x: pd.Series({cols[0]: list(x.reset_index("g")["g"]), 
                            cols[1]: list(x.reset_index("g")["u"])}))
    tg_info = tg_info.join(filt, lsuffix="_all", rsuffix="_filtered")  # join
    tg_info = tg_info.dropna().loc[ann.obs.index.intersection(
        tg_info.dropna().index)]  # re-order according to adata index
    
    # Re-Make String Versions of New Columns with List Entries
    for q in [col_guide_rna, col_num_umis]:  # string versions of list entries 
        tg_info.loc[:, q + "_filtered"] = tg_info[q + "_list_filtered"].apply( 
            lambda x: x if not isinstance(x, list) else feature_split.join(
                str(i) for i in x))  # join processed names by `feature_split`
        
    # DON'T CHANGE THESE!
    rnd = {"g": "Gene", "t": "Total Guides in Cell", 
           "p": "Percent of Cell Guides", "n": "Number in Cell"}
    # Crispr.get_guide_counts() depends on the names in "rnd"
    
    feats_n = feats_n.reset_index().rename(rnd, axis=1).set_index(
        [feats_n.index.names[0], rnd["g"]])
    tg_info = tg_info.assign(feature_split=feature_split)
    return tg_info, feats_n
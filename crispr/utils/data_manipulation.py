import anndata as ad
import pandas as pd
import numpy as np

def create_pseudobulk(adata: ad.AnnData,
    groupby: str | list[str],
    *,
    n_samples_per_group: int,
    n_cells: int,
    random_state: None | int | np.random.RandomState = None,
    layer: str = None) -> ad.AnnData:
    """'
    From SC Best Practices (Theis Lab).
    
    Sum sample of X per condition.

    Drops conditions which don't have enough samples.

    Parameters
    ----------
    adata
        AnnData to sum expression of
    groupby
        Keys in obs to groupby
    n_samples_per_group
        Number of samples to take per group
    n_cells
        Number of cells to take per sample
    random_state
        Random state to use when sampling cells
    layer
        Which layer of adata to use

    Returns
    -------
    AnnData with same var as original, obs with columns from groupby, and X.
    """
    from scipy import sparse
    from sklearn.utils import check_random_state

    # Checks
    if isinstance(groupby, str):
        groupby = [groupby]
    random_state = check_random_state(random_state)

    indices = []
    labels = []

    grouped = adata.obs.groupby(groupby)
    for k, inds in grouped.indices.items():
        # Check size of group
        if len(inds) < (n_cells * n_samples_per_group):
            continue

        # Sample from group
        condition_inds = random_state.choice(
            inds, n_cells * n_samples_per_group, replace=False
        )
        for i, sample_condition_inds in enumerate(np.split(
            condition_inds, 3)):
            if isinstance(k, tuple):
                labels.append((*k, i))
            else:  # only grouping by one variable
                labels.append((k, i))
            indices.append(sample_condition_inds)

    # obs of output AnnData
    new_obs = pd.DataFrame.from_records(
        labels,
        columns=[*groupby, "sample"],
        index=["-".join(map(str, l)) for l in labels],
    )
    n_out = len(labels)

    # Make indicator matrix
    indptr = np.arange(0, (n_out + 1) * n_cells, n_cells)
    indicator = sparse.csr_matrix(
        (
            np.ones(n_out * n_cells, dtype=bool),
            np.concatenate(indices),
            indptr,
        ),
        shape=(len(labels), adata.n_obs),
    )

    return ad.AnnData(
        X=indicator @ sc.get._get_obs_rep(adata, layer=layer),
        obs=new_obs,
        var=adata.var.copy(),
    )
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def sparsify_hard_map(
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    n_rows: int,
    n_cols: int,
    row_names: List[str] | None = None,
    col_names: List[str] | None = None,
):
    ordr = np.argsort(col_idx)
    row_idx = row_idx[ordr]
    col_idx = col_idx[ordr]

    T_sparse = coo_matrix((np.ones(n_cols), (row_idx, col_idx)), shape=(n_rows, n_cols))

    T_sparse = pd.DataFrame.sparse.from_spmatrix(
        T_sparse,
        index=row_names,
        columns=col_names,
    )
    return T_sparse


def soft_T_to_hard(
    T: np.ndarray | pd.DataFrame,
    S_to: np.ndarray | pd.DataFrame | None = None,
    S_from: np.ndarray | pd.DataFrame | None = None,
    pos_by_argmax=True,
    pos_by_weight=False,
    **kwargs
):

    n_rows, n_cols = T.shape

    if isinstance(T, np.ndarray):
        row_names = None
        col_names = None
    elif isinstance(T, pd.DataFrame):
        row_names = T.index
        col_names = T.columns
    else:
        raise NotImplementedError

    col_idx = np.arange(n_cols)

    # assign hard positions by argmax
    if pos_by_weight:
        assert S_from, "Single cell coordinates not available for pos_by_weight"
        assert S_to, "Spatial coordinates not available for pos_by_weight"

        if isinstance(S_to, pd.DataFrame):
            S_to = S_to.values

        if isinstance(S_from, pd.DataFrame):
            S_from = S_from.values

        # build kd tree of spatial coordinates in "to"
        kd = cKDTree(S_to)
        _, idxs = kd.query(S_from, k=1)

        row_idx = idxs.flatten()

    # assign hard positions by argmax
    if pos_by_argmax:
        row_idx = np.argmax(T, axis=0).flatten()

    # save hard map as sparse matrix
    T_hard = sparsify_hard_map(
        row_idx,
        col_idx,
        n_rows,
        n_cols,
        row_names=row_names,
        col_names=col_names,
    )
    return T_hard

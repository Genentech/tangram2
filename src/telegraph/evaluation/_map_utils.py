import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def soft_T_to_hard(
    cls,
    T: np.ndarray | pd.DataFrame,
    out,
    hard_map=False,
    pos_by_argmax=True,
    pos_by_weight=False,
    S_to=None,
    S_from=None,
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

    if hard_map and (pos_by_argmax or pos_by_weight):
        col_idx = np.arange(n_cols)

        # assign hard positions by argmax
        if pos_by_weight:
            # build kd tree of spatial coordinates in "to"
            kd = cKDTree(S_to)
            _, idxs = kd.query(S_from, k=1)

            row_idx = idxs.flatten()

        # assign hard positions by argmax
        if pos_by_argmax:
            row_idx = np.argmax(T, axis=0).flatten()

        # save hard map as sparse matrix
        cls.hard_update_out_dict(
            out,
            row_idx,
            col_idx,
            n_rows,
            n_cols,
            row_names=row_names,
            col_names=col_names,
        )

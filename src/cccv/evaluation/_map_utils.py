import numpy as np
from scipy.spatial import cKDTree


def soft_T_to_hard(
    cls,
    T: np.ndarray,
    out,
    hard_map=False,
    pos_by_argmax=True,
    pos_by_weight=False,
    S_to=None,
    S_from=None,
    return_sparse=False,
):

    n_rows, n_cols = T.shape

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
            return_sparse,
        )

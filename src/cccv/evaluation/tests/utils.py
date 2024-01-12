from typing import Any, Dict

import anndata as ad
import numpy as np
import pandas as pd


def check_out(method, res_dict):
    assert all(
        [x in res_dict for x in method.outs]
    ), "out objects not found in results dictionary"


def make_fake_adata(n_obs, n_features, n_labels, random_seed=42):
    rng = np.random.default_rng(random_seed)

    X = rng.random((n_obs, n_features))
    S = rng.random((n_obs, 2))
    labels = rng.integers(0, n_labels + 1, size=n_obs)
    labels = [f"label_{x}" for x in labels]

    var_names = ["feature_{}".format(x) for x in range(n_features)]
    obs_names = ["obs_{}".format(x) for x in range(n_obs)]

    adata = ad.AnnData(
        X,
        var=pd.DataFrame(var_names, index=var_names, columns=["features"]),
        obs=pd.DataFrame(labels, index=obs_names, columns=["labels"]),
    )

    adata.obsm["spatial"] = S

    return adata


def make_fake_X(
    n_to=10,
    n_from=12,
    n_features_to=15,
    n_features_from=15,
    n_labels_to=5,
    n_labels_from=3,
    **kwargs,
):
    X_to = make_fake_adata(n_to, n_features_to, n_labels_to)
    X_from = make_fake_adata(n_from, n_features_from, n_labels_from)

    res_dict = dict(X_to=X_to, X_from=X_from)
    return res_dict


def make_fake_map_input(*args, **kwargs):
    return make_fake_X(*args, **kwargs)


def make_fake_T(
    n_to=10,
    n_from=12,
    t_row_sum: float | None = 1,
    t_col_sum: float | None = None,
    res_dict: Dict[str, Any] | None = None,
    return_sparse: bool = False,
    **kwargs,
):

    if res_dict is None:
        res_dict = {}

    if "X_to" in res_dict:
        to_names = res_dict["X_to"].obs.index.values.tolist()
        n_to = len(to_names)
    else:
        to_names = [f"to_{x}" for x in range(n_to)]

    if "X_from" in res_dict:
        from_names = res_dict["X_from"].obs.index.values.tolist()
        n_from = len(from_names)
    else:
        from_names = [f"from_{x}" for x in range(n_from)]

    if return_sparse:
        col_idx = np.arange(n_from).astype(int)
        row_idx = np.random.choice(n_to, replace=True, size=n_from).astype(int)

        ordr = np.argsort(col_idx)

        row_idx = row_idx[ordr]
        col_idx = col_idx[ordr]

        T = coo_matrix((np.ones(n_cols), (row_idx, col_idx)), shape=(n_to, n_from))

    else:

        T = np.random.random((n_to, n_from))

        match (t_row_sum, t_col_sum):
            case (_, None):
                sum_axis = 1
            case (None, _):
                sum_axis = 0
            case (None, None):
                sum_axis = None
            case (_, _):
                raise AssertionError("Can't specify sum for both row/col")

        if sum_axis is not None:
            T_div = T.sum(axis=sum_axis, keepdims=True)
            T = np.divide(T, T_div)

        res_dict["T"] = T
        res_dict["from_names"] = from_names
        res_dict["to_name"] = to_names

    return res_dict


def make_fake_S(
    n_to=10,
    n_from=12,
    res_dict: Dict[str, Any] | None = None,
    return_sparse: bool = False,
    **kwargs,
):
    if res_dict is None:
        res_dict = {}

    if "T" in res_dict:
        n_to, n_from = res_dict["T"].shape

    S_to = np.random.unform(0, 10, size=(n_to, 2))
    S_from = np.random.unform(0, 10, size=(n_from, 2))

    res_dict["S_to"] = S_to
    res_dict["S_from"] = S_from

    return res_dict


def make_fake_D(
    n_to=10,
    n_from=12,
    n_grp_to=3,
    n_grp_from=4,
    res_dict: Dict[str, Any] | None = None,
    **kwargs,
):

    if res_dict is None:
        res_dict = {}
    else:
        if "T" in res_dict:
            n_to, n_from = res_dict["T"].shape
        else:
            if "X_to" in res_dict:
                n_to = res_dict["X_to"].shape[0]
            elif "X_to_pred" in res_dict:
                n_to = res_dict["X_to_pred"].shape[0]

            if "X_from" in res_dict:
                n_from = res_dict["X_from"].shape[0]

    def _gen_D(n_row, n_col, row_prefix="row", col_prefix="cat"):
        mat = np.concatenate(
            [np.random.randint(0, 2, size=(n_row, 1)) for x in range(n_col)], axis=1
        )

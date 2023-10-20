from functools import reduce
from typing import Any, Dict, List, TypeVar

import anndata as ad
import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import spmatrix

W = TypeVar("W")


def identity_fun(x: W, *args, **kwargs) -> W:
    return x


def read_input_object(path: str, return_array: bool = False, layer=None):
    if path.endswith(".h5ad"):
        obj = ad.read_h5ad(path)
        if layer is not None:
            obj.X = obj.layers[layer]

        if return_array:
            obj = obj.X
            if isinstance(obj, spmatrix):
                obj = obj.todense()

    elif path.endswith((".csv", ".tsv")):
        obj = pd.read_csv(path, header=0, index_col=0)
        if return_array:
            obj = obj.values
    elif path.endswith(".npy"):
        obj = np.load(path)
    else:
        raise NotImplementedError

    return obj


def read_data(data_dict: Dict[str, str]) -> Dict[str, Any]:
    input_dict = dict()
    rename_map = dict(sp="X_to", sc="X_from")

    for name in data_dict.keys():
        pth = data_dict[name]["path"]
        layer = data_dict[name].get("layer", None)
        return_array = data_dict[name].get("asarray", False)
        obj = read_input_object(pth, return_array=return_array, layer=layer)

        input_dict[name] = obj

    for old_name, new_name in rename_map.items():
        if old_name in input_dict:
            input_dict[new_name] = input_dict.pop(old_name)

    return input_dict


def ad2np(func):
    # this is ugly af; remnant from when @andera29
    # thought it would be a good idea to make the
    # methods agnostic to the anndata package

    def wrapper(
        cls,
        input_dict: Dict[str, Any],
        to_spatial_key: str = "spatial",
        from_spatial_key: str | None = None,
        *args,
        **kwargs,
    ):
        arr_X_to = input_dict["X_to"].X
        if isinstance(arr_X_to, spmatrix):
            arr_X_to = arr_X_to.toarray()
        arr_X_from = input_dict["X_from"].X
        if isinstance(arr_X_from, spmatrix):
            arr_X_from = arr_X_from.toarray()

        S_to = input_dict["X_to"].obsm[to_spatial_key]
        if from_spatial_key is not None:
            S_from = input_dict["X_from"].obsm[from_spatial_key]
        else:
            S_from = None

        input_dict["__X_to"] = input_dict.pop("X_to").copy()
        input_dict["__X_from"] = input_dict.pop("X_from").copy()

        input_dict["X_to"] = arr_X_to
        input_dict["X_from"] = arr_X_from
        input_dict["S_from"] = S_from
        input_dict["S_to"] = S_to

        out = func(cls, input_dict, *args, **kwargs)

        del input_dict["X_to"]
        del input_dict["X_from"]

        input_dict["X_to"] = input_dict.pop("__X_to")
        input_dict["X_from"] = input_dict.pop("__X_from")

        out["to_names"] = indput_dict["X_to"].obs.index.values.tolist()
        out["from_names"] = indput_dict["X_from"].obs.index.values.tolist()

        return out

    return wrapper


@njit
def mat_cosine_similarity(V1, V2, axis=0):
    n_1 = np.sum(V1 * V1, axis=axis) ** 0.5
    n_2 = np.sum(V2 * V2, axis=axis) ** 0.5
    norms_sq = n_1 * n_2
    ewise = V1 * V2
    dot_unorm = np.sum(ewise, axis=axis)
    cs = 1 - dot_unorm / norms_sq
    return cs


def matrix_correlation(O, P):
    (n, t) = O.shape  # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (
        np.einsum("nt->t", O, optimize="optimal") / np.double(n)
    )  # compute O - mean(O)
    DP = P - (
        np.einsum("nm->m", P, optimize="optimal") / np.double(n)
    )  # compute P - mean(P)

    cov = np.einsum("nm,nt->mt", DP, DO, optimize="optimal")

    varP = np.einsum("nm,nm->m", DP, DP, optimize="optimal")
    varO = np.einsum("nt,nt->t", DO, DO, optimize="optimal")
    tmp = np.einsum("m,t->mt", varP, varO, optimize="optimal")

    return cov / np.sqrt(tmp)


def get_ad_value(adata: ad.AnnData, key: str):
    if key in adata.obs:
        out = adata.obs[key].values
    elif key in adata.obsm:
        out = adata.obsm[key]
    elif key in adata.obsp:
        out = adata.obsp[key]
    elif key in adata.var:
        out = adata.var[key].values
    elif key in adata.uns:
        out = adata.uns[key]
    else:
        out = None

    return out


def expand_key(
    d: Dict[Any, Any],
    fill_keys: List[str],
    expand_key: str = "all",
) -> Dict[Any, Any]:
    if expand_key in d:
        exp_d = {x: d[expand_key] for x in fill_keys}
        return exp_d
    else:
        return d


def recursive_get(d, *keys):
    return reduce(lambda c, k: c.get(k, {}), keys, d)

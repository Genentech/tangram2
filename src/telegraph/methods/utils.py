import gzip
import os.path as osp
from functools import reduce
from typing import Any, Dict, List, Literal, Tuple, TypeVar

import anndata as ad
import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import spmatrix

W = TypeVar("W")


def array_to_sparse_df(
    arr: np.ndarray, index: None | List[str] = None, columns: None | List[str] = None
):
    arr = pd.DataFrame(
        arr,
        index=index,
        columns=columns,
    )

    arr = arr.astype(pd.SparseDtype("float", 0))

    return arr


def merge_default_dict_with_kwargs(default_dict, kwargs):
    out_dict = dict()
    for key, value in default_dict.items():
        if key in kwargs:
            out_dict[key] = kwargs[key]
        else:
            out_dict[key] = value
    return out_dict


def update_default_groups(raw_groups: List[Tuple[str, str]], uni_labels: np.ndarray):
    groups = raw_groups
    new_groups = []
    for group in groups:
        grp_a, grp_b = group
        new_grp_a = [x for x in uni_labels if grp_a in x]
        new_group = [
            (x, x.replace(grp_a, grp_b))
            for x in new_grp_a
            if x.replace(grp_a, grp_b) in uni_labels
        ]
        new_groups += new_group

    return new_groups


def ifnonereturn(obj, return_object: None):
    if obj is None:
        return return_object
    return obj


def get_from_dict_with_fuzzy(
    key: str,
    base_dict: Dict[str, Any],
    use_fuzzy_match: bool = False,
    verbose: bool = False,
):
    # use fuzzy match if specified
    if use_fuzzy_match:
        match_key = get_fuzzy_key(key, base_dict, verbose=verbose)
    else:
        match_key = key

    return base_dict.get(match_key, None)


def check_in_out(func):
    """compatiblity check

    This decorator checks whether the necessary
    input is provided to a method


    """

    def inner(cls, input_dict: Dict[str, Any], **kwargs):

        vars_not_in_input = [x for x in cls.ins if x not in input_dict]

        for obj in cls.ins:
            if isinstance(obj, (list, tuple)):
                any_in_input = any([x in input_dict for x in obj])
                if not any_in_input:
                    raise ValueError(
                        "None of {} were in the input".format(", ".join(obj))
                    )
            else:
                if obj not in input_dict:
                    raise ValueError("{} was not in the input".format(obj))

        return func(cls, input_dict, **kwargs)

    return inner


def design_matrix_to_labels(design_matrix: pd.DataFrame) -> np.ndarray:
    # design matrix is output from any group method : [n_obs] x [n_covariates]
    # creates an array of length [n_obs], each observation has one label
    labels = np.array(
        list(
            map(
                lambda x: "_".join(design_matrix.columns.values[x].tolist()),
                design_matrix.values == 1,
            )
        )
    )
    return labels


def list_or_path_get(obj: str | None | List[str] = None):
    # checks if an object is a list or a path to a file
    # that holds a list. If path, we read the file
    # and returns it as a list. If already a list, tuple, or
    # pandas series we return the object (unmodified)

    if obj is not None:
        # if a string and a path read
        if isinstance(obj, str) and osp.isfile(obj):
            # read csv file
            if obj.endswith(".csv"):
                obj = pd.read_csv(obj, index_col=0)
                obj = np.reshape(obj.values, -1)
            # read text file
            elif obj.endswith(".txt"):
                with open(obj, "r") as f:
                    obj = f.readlines()
                    obj = [x.rstrip("\n") for x in obj]
            else:
                NotImplementedError

            return obj
        # return list-like objects
        elif isinstance(obj, (list, tuple, pd.Series)):
            return obj

        else:
            raise NotImplementedError

    else:
        return None


def get_fuzzy_key(
    fuzzy_key: str,
    d: Dict[str, Any],
    allow_fuzzy_ratio: int = 75,
    verbose: bool = False,
) -> str:
    # function to support fuzzy dictionary matching

    if fuzzy_key in d:
        return fuzzy_key

    # get keys in dictionary
    true_keys = list(d.keys())
    # compute fuzzy score for each key in dictionary and the
    # specified key
    fuzzy_rank = [(key, fuzz.ratio(fuzzy_key, key)) for key in true_keys]
    # sort key w.r.t. fuzzy score, highest first
    fuzzy_rank.sort(key=lambda x: -x[1])
    # get highest scoring key and its score
    top_key, top_score = fuzzy_rank[0]

    # if key match fuzzy_key then no correction
    if top_key == fuzzy_key:
        return fuzzy_key
    # if score is higher than required ratio
    elif top_score >= allow_fuzzy_ratio:
        if verbose:
            print("Using {} instead of {}".format(top_key, fuzzy_key))
        return top_key
    # if no exact or fuzzy match
    else:
        if verbose:
            print("No match for key")
        return fuzzy_key


def to_csv_gzip(df: pd.DataFrame, filename, **kwargs) -> None:
    # function to support saving data frames in compressed format
    with gzip.open(
        filename,
        mode="wb",
    ) as f:
        df.to_csv(f, **kwargs)


def identity_fun(x: W, *args, **kwargs) -> W:
    # dummy function, leaves input unchanged
    return x


def ifnonereturn(obj, return_object: None):
    if obj is None:
        return return_object
    return obj


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

        out["to_names"] = input_dict["X_to"].obs.index.values.tolist()
        out["from_names"] = input_dict["X_from"].obs.index.values.tolist()

        return out

    return wrapper


@njit
def mat_cosine_similarity(V1, V2, axis=0):
    # efficient implementation of cosine similarity
    n_1 = np.sum(V1 * V1, axis=axis) ** 0.5
    n_2 = np.sum(V2 * V2, axis=axis) ** 0.5
    norms_sq = n_1 * n_2
    ewise = V1 * V2
    dot_unorm = np.sum(ewise, axis=axis)
    cs = 1 - dot_unorm / norms_sq
    return cs


def matrix_correlation(
    O: np.ndarray | spmatrix, P: np.ndarray | spmatrix
) -> np.ndarray:
    # efficient implementation of columnwise
    # correlation between two input matrices
    # shamelessly stolen from: https://github.com/ikizhvatov/efficient-columnwise-correlation/blob/master/columnwise_corrcoef_perf.py

    if isinstance(O, spmatrix):
        O = O.toarray()
    if isinstance(P, spmatrix):
        P = P.toarray()

    (n, t) = O.shape  # n traces of t samples
    (n_bis, m) = P.shape  # n predictions for each of m candidates

    DO = O - (
        np.einsum("nt->t", O, optimize="optimal") / np.double(n)
    )  # compute O - mean(O)
    DP = P - (
        np.einsum("nm->m", P, optimize="optimal") / np.double(n)
    )  # compute P - mean(P)

    # compute covariance
    cov = np.einsum("nm,nt->mt", DP, DO, optimize="optimal")

    # compute variance
    varP = np.einsum("nm,nm->m", DP, DP, optimize="optimal")
    varO = np.einsum("nt,nt->t", DO, DO, optimize="optimal")
    tmp = np.einsum("m,t->mt", varP, varO, optimize="optimal")

    return cov / np.sqrt(tmp)


def get_ad_value(adata: ad.AnnData, key: str, to_np: bool = True, **kwargs):
    # helper function to grab information
    # from anndata objects. Check if the key is
    # present in any of multiple different slots

    # is in .obs?
    if key in adata.obs:
        out = adata.obs[key]
        out = out.values if to_np else out
    # is in .obsm?
    elif key in adata.obsm:
        out = adata.obsm[key]

    # is in .obsp?
    elif key in adata.obsp:
        out = adata.obsp[key]
    # is in .var (features)
    elif key in adata.var:
        out = adata.var[key]
        out = out.values if to_np else out
    # is in .uns?
    elif key in adata.uns:
        out = adata.uns[key]

    # return None if no match
    else:
        out = None

    return out


def listify(obj: W) -> List[W]:
    # check if object is list
    # if yes return unchanged object
    # if no return object in a list format
    if not isinstance(obj, (tuple, list)):
        return [obj]
    return obj


def df2ad(in_df: pd.DataFrame) -> ad.AnnData:
    # convert a pandas dataframe to anndata
    out_ad = ad.AnnData(in_df, obs=in_df.index, var=in_df.columns)
    return out_ad


def _adata_to_input_dict(
    adata: ad.AnnData,
    categorical_labels: List[str] | None = None,
    continuous_labels: List[str] | None = None,
    layer: str | None = None,
):

    X = adata.to_df(layer=layer)
    if isinstance(X, spmatrix):
        X = X.toarray()

    input_dict = dict(X=X)

    D = list()

    if categorical_labels is not None:
        col_names = listify(categorical_labels)

        for col in col_names:
            if col in adata.obs:
                labels = adata.obs[col]
                labels = pd.get_dummies(labels).astype(float)
                D.append(labels)

    if continuous_labels is not None:
        col_names = listify(continuous_labels)

        for col in col_names:
            if col in adata.obs:
                labels = adata.obs[[col]]
                D.append(labels)

    if len(D) > 0:
        D = pd.concat(D, axis=1)
        input_dict["D"] = D

    return input_dict


def adatas_to_input(
    adatas: Dict[Literal["to", "from"], ad.AnnData],
    categorical_labels: Dict[Literal["to", "from"], List[str]] | None = None,
    continuous_labels: Dict[Literal["to", "from"], List[str]] | None = None,
    layers: Dict[Literal["to", "from"], str] | None = None,
):

    input_dict = dict()
    for name, adata in adatas.items():
        cat_labels = (
            None if categorical_labels is None else categorical_labels.get(name)
        )
        con_labels = None if continuous_labels is None else continous_labels.get(name)
        layer = None if layers is None else layers.get(name)
        _input_dict = _adata_to_input_dict(adata, cat_labels, con_labels, layer)

        input_dict[f"X_{name}"] = _input_dict["X"]
        if "D" in _input_dict:
            input_dict[f"D_{name}"] = _input_dict["D"]

    return input_dict


def merge_input_dicts(*input_dicts):

    union_keys = [list(x.keys()) for x in input_dicts]
    union_keys = [x for y in union_keys for x in y]
    union_keys = [x for x in union_keys if union_keys.count(x) == len(input_dicts)]
    union_keys = list(set(union_keys))
    union_keys = [x for x in union_keys if x.startswith(("T", "X", "D"))]

    new_input_dict = dict()

    for key in union_keys:
        obj_list = list()
        for input_dict in input_dicts:
            obj = input_dict[key]
            obj = obj.iloc[:, ~obj.columns.duplicated()]
            obj = obj.to_df() if isinstance(obj, ad.AnnData) else obj
            obj_list.append(obj)

        # same operation now, but perhaps want to change in future
        if key.startswith(("X", "T", "D")):
            obj_list = pd.concat(obj_list, axis=0).fillna(0)
            new_input_dict[key] = obj_list
        else:
            raise ValueError("Merge not available for {}".format(key))

    return new_input_dict

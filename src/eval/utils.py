from functools import reduce
from typing import Any, Dict, List

import anndata as ad
import numpy as np
from numba import njit
from scipy.sparse import spmatrix


def ad2np(func):
    def wrapper(
        cls,
        ad_to: ad.AnnData,
        ad_from: ad.AnnData,
        to_spatial_key: str = "spatial",
        from_spatial_key: str | None = None,
        *args,
        **kwargs,
    ):
        X_to = ad_to.X
        if isinstance(X_to, spmatrix):
            X_to = X_to.toarray()
        X_from = ad_from.X
        if isinstance(X_from, spmatrix):
            X_from = X_from.toarray()

        S_to = ad_to.obsm[to_spatial_key]
        if from_spatial_key is not None:
            S_from = ad_to.obsm[from_spatial_key]
        else:
            S_from = None

        return func(cls, X_to, X_from, S_to=S_to, S_from=S_from, *args, **kwargs)

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
    d: Dict[Any, Any], fill_keys: List[str], expand_key: str = "all",
) -> Dict[Any, Any]:
    if expand_key in d:
        exp_d = {x: d[expand_key] for x in fill_keys}
        return exp_d
    else:
        return d


def recursive_get(d, *keys):
    return reduce(lambda c, k: c.get(k, {}), keys, d)

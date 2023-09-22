import numpy as np
from numba import njit
import anndata as ad

# @njit(nopython=True)
# def matrix_correlation(x: np.ndarray, y: np.ndarray, axis: int = 1):
#     mx = np.mean(x, axis=axis, keepdims=True)
#     my = np.mean(y, axis=axis, keepdims=True)
#     xm, ym = x - mx, y - my
#     r_num = np.add.reduce(xm * ym, axis=axis)
#     r_den = np.sqrt((xm * xm).sum(axis=axis) * (ym * ym).sum(axis=axis))
#     r = r_num / r_den
#     return r


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

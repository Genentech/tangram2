from functools import reduce
from typing import Dict, List, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.stats import multinomial as mul
from scipy.stats import multivariate_hypergeom as mhg


def _check_vals(
    x: float | int | List[float] | List[int] | None,
    name: str = "value",
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> None:
    if x is None:
        return None

    if not hasattr(x, "__len__"):
        x = [x]

    for xi in x:
        if lower_bound is not None:
            assert xi >= lower_bound, f"{name} is less than {lower_bound}"
        if upper_bound is not None:
            assert xi <= upper_bound, f"{name} is greater than {upper_bound}"


def nbrs_to_coo(
    kdres: List[List[int]],
) -> coo_matrix:
    flat_list = lambda x, y: x + y

    row_idx = np.array(
        reduce(
            flat_list, [[k] * len(nbrs) for k, nbrs in enumerate(kdres) if len(nbrs)]
        )
    )
    col_idx = np.array(reduce(flat_list, kdres))
    dta_val = np.ones_like(col_idx)

    coo = coo_matrix((dta_val, (row_idx.astype(int), col_idx.astype(int))))
    return coo


def plot_visify(
    og_ad: ad.AnnData,
    vs_ad: ad.AnnData,
    spatial_key: str = "str",
    feature: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    og_crd = og_ad.obsm[spatial_key]
    vs_crd = vs_ad.obsm[spatial_key]

    if feature is not None:
        og_val = np.asarray(og_ad.obs_vector(feature)).flatten()
        vs_val = np.asarray(vs_ad.obs_vector(feature)).flatten()
    else:
        og_val = np.asarray(og_ad.X.sum(axis=1)).flatten()
        vs_val = np.asarray(vs_ad.X.sum(axis=1)).flatten()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    og_ax = ax[0]
    vs_ax = ax[1]

    og_ax.scatter(og_crd[:, 0], og_crd[:, 1], c=og_val, s=1)
    vs_ax.scatter(vs_crd[:, 0], vs_crd[:, 1], c=vs_val, s=10)

    og_ax.set_title("Original Data")
    vs_ax.set_title("Visified Data")

    for axx in [og_ax, vs_ax]:
        axx.set_aspect("equal")
        for sp in axx.spines.values():
            sp.set_visible(False)

    return fig, ax


def add_mul_noise(
    data: np.ndarray, p_noise: float, return_all: bool = False
) -> Dict[str, np.ndarray]:
    if p_noise == 0:
        return dict(perturbed=data)

    X = data.copy().astype(int)
    p_vals_bg = X.sum(axis=0).flatten()
    p_vals_bg = p_vals_bg / p_vals_bg.sum()

    n_tot = X.sum(axis=1).flatten()
    n_og = (n_tot * (1 - p_noise)).astype(int)
    n_ns = (n_tot - n_og).astype(int)

    X_ns = np.array([mul.rvs(n_ns[ii], p=p_vals_bg) for ii in range(n_ns.shape[0])])
    X_og = mhg.rvs(X, n_og)

    X_prt = (X_ns + X_og).astype(float)
    res_dict = dict(perturbed=X_prt)
    if return_all:
        res_dict["noise"] = X_ns
        res_dict["og"] = X_og

    return res_dict


def square_sample_crd(min_xy: np.ndarray, max_xy: np.ndarray, n_samples: int = 100):
    xx = np.random.uniform(min_xy[0], max_xy[0], size=n_samples)
    yy = np.random.uniform(min_xy[1], max_xy[1], size=n_samples)
    xy = np.hstack((xx[:, None], yy[:, None]))
    return xy


def r_by_n(
    crd_ref: np.ndarray,
    crd_query: np.ndarray,
    n_neighs: int,
    outlier_removal: bool = False,
):

    kd_ref = cKDTree(crd_ref)
    ndists, _ = kd_ref.query(crd_query, k=n_neighs + 1)
    max_kdists = ndists.max(axis=1)
    if outlier_removal:
        iq3 = np.percentile(max_kdists, 75)
        iq1 = np.percentile(max_kdists, 25)
        iqr = iq3 - iq1
        in_lower = max_kdists >= iq1 - iqr * 1.5
        in_upper = max_kdists <= iq3 + iqr * 1.5
        max_kdists = max_kdists[in_lower & in_upper]

    # this is more robust to weird outliers
    av_max_kdist = np.median(max_kdists)

    return av_max_kdist

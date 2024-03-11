from typing import List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, spmatrix
from scipy.spatial import cKDTree as KDTree
from scipy.stats import multivariate_hypergeom as mhg
from utils import add_mul_noise, nbrs_to_coo, r_by_n, square_sample_crd


def visify(
    adata: ad.AnnData,
    spot_dist: float | None = None,
    spot_diameter: float = 55,
    spatial_key: str = "spatial",
    downsample: bool = False,
    return_mappable: bool = False,
    add_indicator: bool = False,
    p_lwr: float = 0.85,
    p_upr: float = 1,
    expected_cells_per_spot: int | None = None,
    n_query: int = 10000,
    p_mul: List[float] | None = None,
    verbose: bool = True,
    random_seed: int = 42,
) -> Tuple[ad.AnnData, ad.AnnData]:

    np.random.seed(random_seed)

    crd = adata.obsm[spatial_key]
    kd_sp = KDTree(adata.obsm[spatial_key])
    min_xy = adata.obsm[spatial_key].min(axis=0)
    max_xy = adata.obsm[spatial_key].max(axis=0)

    if expected_cells_per_spot is None:
        spot_radius = spot_diameter / 2
    else:
        # to make sure we sample in the whole array area
        box_crd = square_sample_crd(min_xy, max_xy, n_query)
        spot_radius = r_by_n(crd, box_crd, expected_cells_per_spot)

    if spot_dist is None:
        # this is taken from visium where the spot diameter is 55um and
        # the spot-center to spot-center distance is 100um
        radius_by_dist = 100 / (55 / 2)
        spot_dist = radius_by_dist * spot_radius
    else:
        assert (
            spot_radius * 2 < spot_dist
        ), "spot radius is too large for the specified inter-spot distance"

    if verbose:
        print("Distance between spots is {} l.u".format(spot_dist))

    xs = np.arange(min_xy[0] + spot_radius, max_xy[0] - spot_radius, spot_dist)
    ys = np.arange(min_xy[1] + spot_radius, max_xy[1] - spot_radius, spot_dist)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.flatten()
    yy = yy.flatten()
    grd_crd = np.hstack((xx[:, None], yy[:, None]))
    if verbose:
        print("Number of spots: {}".format(len(grd_crd)))

    kd_grd = KDTree(grd_crd)

    kdres = kd_grd.query_ball_tree(kd_sp, r=spot_radius)
    obs_num = [len(x) for x in kdres]
    obs_num_av = np.mean(obs_num)
    obs_num_std = np.std(obs_num)
    if verbose:
        print(
            "Average number of cells per spot: {}+/-{}".format(obs_num_av, obs_num_std)
        )

    xmat_sum = np.zeros((len(grd_crd), adata.shape[1]))

    X = adata.X.copy()
    if isinstance(X, spmatrix):
        X = X.todense()
    X = np.asarray(X)

    for ii in range(len(grd_crd)):
        for nix in kdres[ii]:
            xmat_sum[ii] += X[nix].flatten()

    obs_name = ["spot_{}".format(x) for x in range(len(xmat_sum))]

    vs_ad = ad.AnnData(
        xmat_sum,
        obs=pd.DataFrame(
            [],
            index=obs_name,
        ),
        var=adata.var,
    )
    vs_ad.obsm[spatial_key] = grd_crd

    obs_map = nbrs_to_coo(kdres)
    n_rows = vs_ad.shape[0]
    n_cols_og = adata.shape[0]
    # this is awkward but anndata can't save sparse obsm or uns
    vs_ad.uns["visify_cell_map_og"] = dict(
        row_self=obs_map.row,
        row_target=obs_map.col,
        shape=np.array((n_rows, n_cols_og)),
    )

    vs_ad.uns["info"] = dict(average_cells_per_spot=obs_num_av)

    if p_mul is not None:
        for p_val in p_mul:
            noise_res = add_mul_noise(xmat_sum, p_noise=p_val)
            vs_ad.layers[f"mul_noise_{p_val:0.2f}"] = csr_matrix(noise_res["perturbed"])

    if downsample:
        xmat_down = np.zeros_like(xmat_sum)
        for ii in range(xmat_down.shape[0]):
            xvec_fll = xmat_sum[ii]
            xtot = np.sum(xvec_fll)
            p = np.random.uniform(p_lwr, p_upr)
            xsize = int(xtot * p)
            xvec_dwn = mhg.rvs(xvec_fll.astype(int), xsize)
            xmat_down[ii, :] = xvec_dwn

        vs_ad.layers[f"downsampled_{p_lwr:0.2f}_{p_upr:0.2f}"] = csr_matrix(xmat_down)

    vis_res = dict(vs=vs_ad)

    if return_mappable:
        mapped_obs, mapped_idx = np.unique(obs_map.col, return_index=True)
        mapped_obs = mapped_obs.astype(int)
        mp_ad = adata[mapped_obs].copy()
        n_cols_mp = mp_ad.shape[0]

        visify_cell_map = dict(
            row_self=np.arange(len(mp_ad)),
            row_target=obs_map.row[mapped_idx],
            shape=np.array((n_cols_mp, n_rows)),
        )
        mp_ad.uns["visify_cell_map"] = visify_cell_map
        mp_ad.uns["info"] = dict(average_cells_per_spot=obs_num_av)

        vis_res["mp"] = mp_ad
        vis_res["vs"].uns["visify_cell_map_mp"] = dict(
            row_target=np.arange(len(mp_ad)),
            row_self=obs_map.row[mapped_idx],
            shape=np.array((n_rows, n_cols_mp)),
        )

    if add_indicator:
        mapped_obs = np.unique(obs_map.col).astype(int)
        mapped_ind = np.zeros(adata.shape[0])
        mapped_ind[mapped_obs] = 1
        adata.obs["mappable"] = mapped_ind.astype(bool)

        vis_res["adata"] = adata

    return vis_res

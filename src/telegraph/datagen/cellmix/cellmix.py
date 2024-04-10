from functools import reduce
from random import random, shuffle
from typing import List

import anndata as ad
import numpy as np
import pandas as pd
from glum import GeneralizedLinearRegressor as glm
from scipy.sparse import spmatrix
from scipy.spatial import cKDTree
from scipy.stats import multivariate_hypergeom as mhg


def _update_adatas(ad_sc, ad_sp, x_add_sc, sc_var_names):

    ad_sc_add = ad.AnnData(
        x_add_sc,
        obs=pd.DataFrame([], index=ad_sc.obs_names),
        var=pd.DataFrame([], index=sc_var_names),
    )

    ad_sc = ad.concat(
        (ad_sc, ad_sc_add), axis=1, join="outer", uns_merge="only", merge="unique"
    )

    x_add_sp = np.zeros((len(ad_sp), x_add_sc.shape[1]))

    spot_id = ad_sc.obs["spot_id"]

    for ii in np.unique(spot_id):
        x_add_sp[ii] = np.sum(x_add_sc[spot_id == ii], axis=0)

    ad_sp_add = ad.AnnData(
        x_add_sp,
        obs=pd.DataFrame([], index=ad_sp.obs_names),
        var=pd.DataFrame([], index=sc_var_names),
    )

    ad_sp = ad.concat(
        (ad_sp, ad_sp_add), axis=1, join="outer", uns_merge="only", merge="unique"
    )

    return ad_sc, ad_sp


def _add_interactions(
    adata: ad.AnnData,
    n_effect: int = 10,
    spatial_key: str = "spatial",
    subset_types: List[str] | None = None,
    p_active_spots: float = 0.2,
    p_inter: float = 0.5,
    spot_data: bool = True,
    signal_name: str | None = None,
    return_names: bool = True,
    subset_features: List[str] | None = None,
):

    if isinstance(signal_name, str):
        tag = "_" + signal_name
    else:
        tag = ""

    # necessary conditions for interaction
    # - proximity
    # - expresses signal
    # - expresses detection mechanicsm

    # x ~ Poi(b)
    # log [b] = log [library_size] + intercept

    # get data dimensions
    n_obs, n_var = adata.shape
    # get X for fast access
    if subset_features is not None:
        X = adata[:, subset_features].X.copy()
    else:
        X = adata.X.copy()

    if isinstance(X, spmatrix):
        X = X.toarray()
    # get log mean count of transcripts in each cell
    log_mean_count = np.log(X.mean(axis=1)).reshape(-1, 1)
    # get coordinates
    crd = adata.obsm[spatial_key]

    # get gene mean counts
    mus = X.mean(axis=0).flatten()
    mus = mus[mus > 0]

    # get mean counts larger than zero

    # get top quantile of mean counts (gene)
    q_t = np.quantile(mus, 0.99)
    # get high quantile of mean counts (gene)
    q_h = np.quantile(mus, 0.95)
    # get low quantile of mean counts (gene)
    q_l = np.quantile(mus, 0.25)
    # get mean counts (gene)
    q_m = np.mean(mus)
    # get log mean counts (gene)
    log_q_m = np.log(q_m)

    # anon. function to sample the coefficient in the GLM for activated features
    sample_active_coef = lambda: np.log(np.random.uniform(5 * q_h, 5 * q_t)) - log_q_m

    # intercept for S's (base level expression)
    s_intercept = float(np.log(np.random.uniform(q_l, q_h)))
    # S active feature coefficient
    s_active_coef = sample_active_coef()
    # intercept for E's (base level expression)
    e_intercepts = np.log(np.random.uniform(q_l, q_h, size=n_effect)).reshape(1, -1)
    # E active feature coefficients
    e_active_coefs = np.array([sample_active_coef() for x in range(n_effect)])
    e_active_coefs = e_active_coefs.reshape(1, -1)

    # if we are working with spot-like data (multiple cells per spatial observation)
    if spot_data:
        # get tuple form of coordinates
        tuple_crd = [tuple(crd[i]) for i in range(n_obs)]
        # get unique coordinate positions
        uni_crd = list(set(tuple_crd))
        # shuffle unique coordinates
        shuffle(uni_crd)
        # select active spots
        sel_crd = uni_crd[0 : int(len(uni_crd) * p_active_spots)]
        # register cells as active
        s_active = np.array(
            [(x in sel_crd) and (random() < p_active_spots) for x in tuple_crd]
        ).reshape(-1, 1)
    else:
        # randomly sample active cells
        s_active = np.random.randint(0, 2, size=n_obs).reshape(-1, 1)

    # get boolean representation of active cells
    s_active_bool = s_active.astype(bool).flatten()
    # number of sender cells
    n_active = np.sum(s_active)
    # number of non-sender cells
    n_inactive = n_obs - n_active

    # get receptive signaling cells that are not signaling cells
    r_receptive_non_s = np.random.choice(
        [0, 1], p=[1 - p_inter, p_inter], replace=True, size=n_inactive
    )
    r_receptive = np.zeros(n_obs)
    r_receptive[~s_active_bool] = r_receptive_non_s
    # find average mean 2-NN distance
    kd = cKDTree(crd)
    dists, _ = kd.query(crd, k=3)
    del kd
    dists = dists[:, 2::]
    # delta is mean 2-NN distance
    delta = np.mean(dists)

    # vector for cells proximal to sender cells
    r_proximal = np.zeros(n_obs)
    # find cells that are proximal to sender cells
    kd = cKDTree(crd)
    idxs = kd.query_ball_point(crd[s_active_bool], r=delta, eps=1e-2)
    del kd
    idxs = np.unique(np.array(reduce(lambda x, y: x + y, idxs)))

    # register cells proximal to sender cells
    r_proximal[idxs] = 1
    # get active cells (must be proximal and receptive)
    r_active = r_receptive * r_proximal
    r_active = r_active.reshape(-1, 1)

    # log[lambda] for signal feature
    log_lambda_s = s_intercept + s_active_coef * s_active + log_mean_count
    # log[lambda] for effect features
    log_lambda_e = e_intercepts + e_active_coefs * r_active + log_mean_count

    # sample signal expression [n_obs] x [1]
    x_s = np.random.poisson(np.exp(log_lambda_s))
    # sample effect expression [n_obs] x [n_effect]
    x_e = np.random.poisson(np.exp(log_lambda_e))

    # add new expression
    x_new = np.hstack((x_s, x_e))

    # get names for signal and effects
    names = ["signal{}".format(tag)] + [
        "effect{}_{}".format(tag, x) for x in range(n_effect)
    ]

    out = dict(
        x=x_new,
        var_names=names,
        is_s=s_active.flatten().astype(bool),
        is_r=r_active.flatten().astype(bool),
    )

    return out


def _linear_grad(crd, mu_low, mu_high, rad_theta):
    # get line to project on,  based on angle
    line = np.array([np.cos(rad_theta), np.sin(rad_theta)])[:, None]
    # project coordinates on line
    proj = np.dot(crd, line)
    # get max/min projection value
    max_proj = proj.max()
    min_proj = proj.min()
    # normalize projections
    nproj = (proj - min_proj) / (max_proj - min_proj)
    # get mu value for projection
    mu_proj = mu_low + mu_high * nproj
    # sample feature from poisson
    vals = np.random.poisson(mu_proj).flatten()

    return vals


def _add_gradients(
    adata: ad.AnnData, n_features: int | float, return_names: bool = True
):
    # get anndata shape
    n_obs, n_var = adata.shape
    # fast access to X, copy to avoid view
    X = adata.X.copy()
    # make sure it's numpy array
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    crd = adata.obsm["spatial"]
    # get number of features to add
    if n_features < 1:
        # if fraction
        n_add = int(np.ceil(n_features * n_var))
    else:
        # if absolute number
        n_add = int(n_features)

    # get observed mean values
    mus = X.mean(axis=0)
    # remove zero valued means
    mus = mus[mus > 0].flatten()
    # get lower and upper quantiles
    q_h = np.quantile(mus, 0.9)
    q_l = np.quantile(mus, 0.1)
    # get higher and lower mean values
    mus_h = mus[mus > q_h]
    mus_l = mus[mus <= q_l]

    # get radians for the linear gradients
    rad_thetas = np.random.choice(
        np.linspace(0, 2 * np.pi, n_add + 1), replace=False, size=n_add
    )

    # prepare expression matrix for additional featurs
    x_add = np.zeros((n_obs, n_add))

    # sample feature values
    for ii in range(n_add):
        # get upper limit mu
        mu_i_h = np.random.choice(mus_h)
        # get lower limit mu
        mu_i_l = np.random.choice(mus_l)
        # get angle (of line to project on)
        rad_theta_i = rad_thetas[ii]
        # add values
        x_add[:, ii] = _linear_grad(crd, mu_i_l, mu_i_h, rad_theta_i)

    # if only return feature matrix
    if not return_names:
        return x_add

    # add names to feature matrix, return both
    names = ["grad_feature_{:0.2f}".format(x) for x in rad_thetas]

    return x_add, names


def _generate_coords(p_mat: np.ndarray, encode_spatial: bool = True):
    # get number of spots
    n_spots = p_mat.shape[0]
    # determine square grid size
    grid_size = int(np.ceil(np.sqrt(n_spots)))
    # generate grid
    xx = np.arange(grid_size)
    yy = np.arange(grid_size)
    xx, yy = np.meshgrid(xx, yy)
    # convert to [n_spots]x[2] array
    scrd = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
    # remove grid points to match number of spots
    scrd = scrd[0:n_spots].astype(float)

    # if spatial patterns should be encoded
    if encode_spatial:
        # import necessary packages
        from ot import dist as otdist
        from ot import emd
        from sklearn.decomposition import PCA

        # check p_mat
        assert p_mat is not None, "p_mat needed to encode spatial"
        # project to 2D space
        pcrd = PCA(n_components=2).fit_transform(p_mat)
        # compute distance between spots
        D = otdist(scrd, pcrd)
        # Use EMD to get bijective mapping
        G = emd(np.ones(n_spots), np.ones(n_spots), D)
        # get mapping
        idx = np.argmax(G, axis=0)
        # rearrange coordinates according to map
        scrd = scrd[idx]

    return scrd


def _cellmix_type_balanced(
    ad_sc: ad.AnnData,
    n_spots: int,
    n_cells_per_spot: int = 10,
    n_types_per_spot: int = 3,
    label_col: str | None = None,
    add_spatial_structure: bool = True,
    encode_spatial: bool = False,
    n_spatial_grad: None | int = None,
    n_interactions: int | None = None,
    effect_size: int = 10,
    p_signal_spots: float = 0.5,
):
    """creates 'mixed' data akin to spot-based data that will have a pre-specified
    average number of cell types per spot as well as pre-defined number of cells per spot

    """

    # get cell type labels
    labels = ad_sc.obs[label_col].values

    # get unique labels
    uni_labels = np.unique(labels)
    # get number of labels
    n_labels = len(uni_labels)

    # original variable names
    og_var_names = ad_sc.var_names.copy()

    # make sure datagen specs are compatible with data
    assert (
        n_types_per_spot < n_labels
    ), "fewer labels than specified number of expected cell types per spot"

    # get number of cells per spot
    spot_cell_count = np.random.poisson(n_cells_per_spot, size=n_spots)
    # get number of types per spot
    spot_type_count = np.clip(
        np.random.poisson(n_types_per_spot, size=n_spots), a_min=1, a_max=n_labels
    )
    print(spot_type_count.mean())

    # split cell inidices (row) by their label and shuffle cell order to break dependency
    type_dict = {
        k: np.random.permutation(np.where(ad_sc.obs[label_col].values == label)[0])
        for k, label in enumerate(uni_labels)
    }

    # prepare list to hold info on which cell that belongs to each spot
    idx_list = [[] for ii in range(n_spots)]

    # set up mixed expression matrix
    n_var = ad_sc.shape[1]
    # matrix for expression
    x_mat = np.zeros((n_spots, n_var))
    # matrix for cell type proportions
    p_mat = np.zeros((n_spots, n_labels))
    n_mat = np.zeros_like(p_mat)

    # get sc data expression for fast access, anndata issue
    X = ad_sc.X
    if isinstance(X, spmatrix):
        X = X.toarray()

    # iterate over each spot
    for ii in range(n_spots):
        # check how many cells are left in each type (to avoid downstream error)
        n_cells_in_type = np.array([len(x) for x in type_dict.values()])

        # get number cells for spot i
        n_cells = spot_cell_count[ii]
        # get number of types at spot i
        n_types = spot_type_count[ii]

        type_has_cells = n_cells_in_type > n_cells
        n_good = np.sum(type_has_cells)

        if n_good > n_types:
            mask_idx = np.random.choice(n_good, size=n_good - n_types, replace=False)
            og_idx = np.where(type_has_cells)[0]
            n_cells_in_type[og_idx[mask_idx]] = 0

        type_idx = mhg.rvs(n_cells_in_type, n=n_cells)
        n_mat[ii, :] = type_idx
        p_mat[ii, :] = type_idx / type_idx.sum()

        # assign cells to spot
        for k, n in enumerate(type_idx):
            if n > 0:
                # get cells of correct type
                spot_i_idxs = type_dict[k][0:n]
                # track which cells are assigned to spot i
                idx_list[ii] += spot_i_idxs.tolist()
                # remove cells at spot i from pool
                type_dict[k] = type_dict[k][n::]
                # add cells expression to spot expression
                x_mat[ii, :] += X[spot_i_idxs, :].sum(axis=0)

    # create anndata object for spot data
    ad_sp = ad.AnnData(
        x_mat,
        obs=pd.DataFrame([], index=["spot_{}".format(x) for x in range(n_spots)]),
        var=ad_sc.var,
    )

    row_target = [x for y in idx_list for x in y]
    row_self = [[k] * len(x) for k, x in enumerate(idx_list)]
    row_self = [x for y in row_self for x in y]

    ad_sc = ad_sc[row_target, :].copy()
    row_target = [x for x in range(len(row_target))]

    shape = np.array((len(ad_sp), len(ad_sc)))

    # Note: [(s,t) for s,t in zip(row_self,row_target)]
    # will tell you (spot,cell) pairing for ad_sp, ad_sc

    # add grond truth mapping to spatial anndata object
    ad_sp.uns["cellmix_cell_map_mp"] = dict(
        row_self=row_self,
        row_target=row_target,
        shape=shape,
    )

    # add random spatial coordinates for compatibility with some methods
    coords = _generate_coords(p_mat, encode_spatial=encode_spatial)
    # add spatial coordinates
    ad_sp.obsm["spatial"] = coords
    # add cell type proportions
    ad_sp.obsm["ct_proportions"] = pd.DataFrame(
        p_mat,
        index=ad_sp.obs.index,
        columns=uni_labels,
    )

    ad_sp.obsm["ct_counts"] = pd.DataFrame(
        n_mat,
        index=ad_sp.obs.index,
        columns=uni_labels,
    )

    # which spot does cell j map to
    ad_sc.obs["spot_id"] = row_self
    ad_sc.obsm["spatial"] = ad_sp.obsm["spatial"][row_self]
    ad_sc.obsm["spatial_hires"] = ad_sc.obsm["spatial"] + np.random.normal(
        0, 0.1, size=(len(ad_sc), 2)
    )

    if n_spatial_grad is not None:
        x_add, names_add = _add_gradients(
            ad_sc[:, og_var_names].copy(), n_spatial_grad, return_names=True
        )
        ad_sc, ad_sp = _update_adatas(ad_sc, ad_sp, x_add, names_add)

    if n_interactions is not None:

        xs_int = list()
        var_names_int = list()

        for ii in range(n_interactions):
            int_res = _add_interactions(
                ad_sc[:, og_var_names].copy(),
                n_effect=effect_size,
                return_names=True,
                p_active_spots=p_signal_spots,
                spot_data=True,
                signal_name=str(ii),
            )
            x_int_i = int_res["x"]
            var_names_int_i = int_res["var_names"]
            xs_int.append(x_int_i)
            var_names_int += var_names_int_i
            is_s, is_r = int_res["is_s"], int_res["is_r"]
            ad_sc.obs[f"signaler_{ii}"] = is_s
            ad_sc.obs[f"receiver_{ii}"] = is_r
            active_state = np.array(["none"] * len(ad_sc))
            active_state[is_s] = "S"
            active_state[is_r] = "R"
            ad_sc.obs[f"S_R_{ii}"] = active_state

        xs_int = np.concatenate(xs_int, axis=1)

        ad_sc, ad_sp = _update_adatas(ad_sc, ad_sp, xs_int, var_names_int)

    return ad_sp, ad_sc


def _downsample(adata: ad.AnnData, downsample: List[float] | float | None = None):
    """downsamples exprssion according to a given percentage.
    the new library size of each observation will be downsample % of the original one.

    """

    n_spots, n_var = adata.shape

    # fast access, anndata issue
    x_mat = adata.X
    if isinstance(x_mat, spmatrix):
        x_mat = x_mat.toarray()

    # if no downsampling
    if downsample is None:
        return adata

    # listify
    downsample = downsample if isinstance(downsample, list) else [downsample]

    # iterate over downsampling options
    for ds in downsample:
        x_mat_new = np.zeros_like(x_mat)
        # downsample each spot
        for ii in range(n_spots):
            # original library size i
            og_lib_size = x_mat[ii].sum()
            # new library size for spot i
            nw_lib_size = int(np.ceil(float(ds) * og_lib_size))
            # get downsampled expression vector
            new_x_i = mhg.rvs(m=x_mat[ii].astype(int), n=nw_lib_size)
            # add to new matrix
            x_mat_new[ii] = new_x_i

        # update anndata object
        layer_name = "downsample_{}".format(ds)
        adata.layers[layer_name] = x_mat_new

    return adata


def cellmix(
    ad_sc: ad.AnnData,
    n_spots: int,
    n_cells_per_spot: int = 10,
    n_types_per_spot: int = 3,
    label_col: str | None = None,
    downsample: List[float] | float | None = None,
    encode_spatial: bool = False,
    n_spatial_grad: None | int = None,
    n_interactions: bool | None = None,
    effect_size: int = 10,
    p_signal_spots: float = 0.5,
):
    """pretty wrapper for cellmix"""

    if label_col is None:
        raise NotImplementedError
    else:
        ad_sp, ad_sc = _cellmix_type_balanced(
            ad_sc,
            n_spots,
            n_cells_per_spot,
            n_types_per_spot,
            label_col,
            encode_spatial=encode_spatial,
            n_spatial_grad=n_spatial_grad,
            n_interactions=n_interactions,
            effect_size=effect_size,
            p_signal_spots=p_signal_spots,
        )

        ad_sp = _downsample(ad_sp, downsample)

    return ad_sp, ad_sc

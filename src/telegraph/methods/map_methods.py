import os.path as osp
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_matrix, spmatrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import softmax
from torch.cuda import is_available

import telegraph.methods.policies as pol
import telegraph.methods.transforms as tf
import telegraph.methods.utils as ut
from telegraph.methods._methods import MethodClass


class MapMethodClass(MethodClass):
    # Base Class for MapMethods
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    @abstractmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        use_emb: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:
        pass


class RandomMap(MapMethodClass):
    # class that randomly maps object in "from"
    # to locations in "to"

    ins = ["X_to", "X_from"]
    outs = ["T"]

    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        seed: int = 1,
        experiment_name: str | None = None,
        use_emb: bool = False,
        **kwargs,
    ):
        # set random seed for reproducibility
        rng = np.random.default_rng(seed)

        if use_emb:
            X_to = input_dict.get("Z_to")
            # anndata object that we map _from_
            X_from = input_dict.get("Z_from")
            X_to = ut.df2ad(X_to)
            X_from = ut.df2ad(X_from)
        else:
            # anndata object that we map _to_
            X_to = input_dict["X_to"]
            pol.check_values(X_to, "X_to")
            # anndata object that we map _from_
            X_from = input_dict["X_from"]
            pol.check_values(X_from, "X_from")

        # get names of observations
        to_names = X_to.obs_names
        from_names = X_from.obs_names

        # get dimensions
        n_rows = X_to.shape[0]
        n_cols = X_from.shape[0]

        # create random map, each observation in "from" is
        # assigned to a random object in "to"
        col_idx = np.arange(n_cols).astype(int)
        row_idx = rng.choice(n_rows, replace=True, size=n_cols).astype(int)

        # dictionary to return
        out = dict()
        # create sparse map and add to out dict
        out["T_hard"] = tf.sparsify_hard_map(
            row_idx,
            col_idx,
            n_rows,
            n_cols,
            row_names=to_names,
            col_names=from_names,
        )

        out["T"] = out["T_hard"]
        T = out["T"]

        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (n_rows, n_cols))

        return out


class ArgMaxCorrMap(MapMethodClass):
    # Method that assigns each observation in "from"
    # to the observation in "to" that it has the highest
    # correlation with, w.r.t. feature expression
    ins = ["X_from", "X_to"]
    outs = ["T"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    @classmethod
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        experiment_name: str | None = None,
        use_emb: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:

        if use_emb:
            # get embedding to
            X_to = input_dict.get("Z_to")
            # get embedding from
            X_from = input_dict.get("Z_from")
            X_to = ut.df2ad(X_to)
            X_from = ut.df2ad(X_from)

        else:
            # anndata object that we map _to_
            X_to = input_dict.get("X_to")
            pol.check_values(X_to, "X_to")
            # anndata object that we map _from_
            X_from = input_dict.get("X_from")
            pol.check_values(X_from, "X_from")

        # get names of observations
        to_names = X_to.obs_names
        from_names = X_from.obs_names

        # n_obs in from
        n_cols = X_from.shape[0]
        # n_obs in to
        n_rows = X_to.shape[0]

        col_idx = np.arange(n_cols).astype(int)

        overlap = list(set(X_to.var_names).intersection(set(X_from.var_names)))

        # get correlation between all observations in "to" and "from"
        sim = ut.matrix_correlation(X_to[:, overlap].X.T, X_from[:, overlap].X.T)
        # set nan to max anticorrelation
        sim[np.isnan(sim)] = -np.inf
        # make probabilistic
        T_soft = softmax(sim, axis=1)
        # for each observation in "from" get id of
        # observation in "to" that it correlates the most with
        row_idx = np.argmax(sim, axis=1).astype(int)

        # output
        out = dict()
        # make correlation matrix T_soft
        out["T"] = ut.array_to_sparse_df(T_soft.T, index=to_names, columns=from_names)

        # update output with sparse map
        # create sparse map and add to out dict
        out["T_hard"] = tf.sparsify_hard_map(
            row_idx,
            col_idx,
            n_rows,
            n_cols,
            row_names=to_names,
            col_names=from_names,
        )

        # add standard objects to out dict

        T = out["T"]
        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (n_rows, n_cols))

        return out


class TangramMap(MapMethodClass):
    # TangramMap Baseclass
    # tangram module to use
    tg = None
    # version number
    version = None

    ins = [("X_to", "Z_to"), ("X_from", "Z_from")]
    outs = ["T"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass

    @classmethod
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        to_spatial_key: str = "spatial",
        from_spatial_key: str = "spatial",
        num_epochs: int = 1000,
        genes: List[str] | str | None = None,
        experiment_name: str | None = None,
        use_emb: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:

        if cls.version == "1":
            import tangram as tg
        elif cls.version == "2":
            import tangram2 as tg
        else:
            raise NotImplementedError

        if use_emb:
            X_to = input_dict.get("Z_to")
            # anndata object that we map _from_
            X_from = input_dict.get("Z_from")
        else:
            # anndata object that we map _to_
            X_to = input_dict["X_to"]
            pol.check_values(X_to, "X_to")
            # anndata object that we map _from_
            X_from = input_dict["X_from"]
            pol.check_values(X_from, "X_from")

        ad_to = ut.df2ad(X_to)
        ad_from = ut.df2ad(X_from)

        # n_obs in "from"
        n_cols = ad_from.shape[0]
        # n_obs in "to"
        n_rows = ad_to.shape[0]

        # get marker genes from tangram
        if genes is not None:
            genes = ut.list_or_path_get(genes)

        # preprocess anndata for mapping
        tg.pp_adatas(ad_from, ad_to, genes=genes)
        mode = kwargs.pop("mode", "cells")
        wandb_config = kwargs.pop("wandb_config", {})
        wandb_config["step_prefix"] = experiment_name

        random_state = kwargs.get("random_state")
        if random_state is None:
            random_state = kwargs.get("seed")
            if random_state is None:
                random_state = 42

        method_params = dict(
            adata_sc=ad_from,
            adata_sp=ad_to,
            mode=mode,
            device=("cuda:0" if is_available() else "cpu"),
            num_epochs=num_epochs,
            cluster_label=kwargs.pop("cluster_label", None),
            random_state=random_state,
        )

        default_loss_params = {
            "learning_rate": 0.1,
            "lambda_d": 0,
            "density_prior": "rna_count_based",
            "lambda_g1": 1,
            "lambda_g2": 0,
            "lambda_r": 0,
            "scale": True,
            "lambda_count": 1,
            "lambda_f_reg": 1,
            "target_count": None,
        }

        for key, value in default_loss_params.items():
            if key in kwargs:
                method_params[key] = kwargs[key]
            else:
                method_params[key] = value

        if cls.version == "2":
            method_params
            method_params["wandb_log"] = (kwargs.pop("wandb_log", False),)
            method_params["wandb_config"] = wandb_config

        # map cells in "from" to "to"
        tg_out = tg.map_cells_to_space(
            **method_params,
        )

        # depending on mode and version, treat output differently
        if (cls.version == "2") and (mode == "hejin_workflow"):
            # hejin_workflow mode in tg2 returns a tuple
            # the map (T) and the re-scaled "from" data
            ad_map, X_from_scaled = tg_out
        elif (cls.version == "1") or (cls.version == "2"):
            # all other modes and versions return a single values
            # the map (T)
            ad_map = tg_out
            # set scaled to None for later
            X_from_scaled = None
        else:
            NotImplementedError

        # get the map T (here [n_from] x [n_to])
        T_soft = ad_map.X.astype(float)
        # predict coordinates of observations in "from" by weighted average

        # output dict
        out = dict()
        if to_spatial_key in ad_to.obsm:
            # spatial coordinates of "to"
            S_to = ad_to.obsm[to_spatial_key].astype(float)
            # spatial coordinates for "from" : [n_from] x [n_spatial_dims]
            S_from = T_soft @ S_to
            out["S_from"] = S_from
            out["S_to"] = S_to

        # observation names for "to"
        to_names = ad_to.obs.index.values.tolist()
        # observation named for "from"
        from_names = ad_from.obs.index.values.tolist()
        # transpose map (T) to be in expected format [n_to] x [n_from]
        out["T"] = ut.array_to_sparse_df(T_soft.T, index=to_names, columns=from_names)

        # anndata with rescaled (with coefficient) "from" data
        out["X_from_scaled"] = X_from_scaled
        out["to_names"] = to_names
        out["from_names"] = from_names

        T = out["T"]
        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (n_rows, n_cols))

        return out


class TangramV1Map(TangramMap):
    # Method class for TangramV1
    version = "1"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class TangramV2Map(TangramMap):
    # Method class for TangramV2
    version = "2"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass

    @classmethod
    @property
    def custom_save_funcs(cls) -> Dict[str, Callable]:
        _funcs = dict(w=cls._save_w)
        return _funcs

    @classmethod
    def _save_w(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        # update this function to add
        # save method for the 'w'
        pass


class CeLEryMap(MapMethodClass):
    import CeLEry as cel

    # Method class for CeLEry
    # github: https://github.com/QihuangZhang/CeLEry
    ins = ["X_to", "X_from"]
    outs = ["S_from"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        hidden_dims: List[int] = [30, 25, 15],
        num_epochs_max: int = 100,
        spatial_key: str = "spatial",
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        # anndata of "to" object
        X_to = input_dict["X_to"]
        # anndata of "from" object
        X_from = input_dict["X_from"]
        # add necessary columns for to .obs for method to work
        # CeLEry will look for "x_pixel" and "y_pixel" in .obs

        X_to.obs[["x_pixel", "y_pixel"]] = X_to.obsm[spatial_key]

        # CeLEry saves output that we don't care about
        # pipe this to a tempdir to avoid junk files
        with TemporaryDirectory() as tmpdir:
            # fit model using "to" data
            model_train = cel.Fit_cord(
                data_train=X_to,
                hidden_dims=hidden_dims,
                num_epochs_max=num_epochs_max,
                path=tmpdir,
                filename="celery_model",
            )

            # predict coordinates of "from"
            pred_cord = cel.Predict_cord(
                data_test=X_from, path=tmpdir, filename="celery_model"
            )

        # Note: this method does not give a map (T)
        # only predicted coordinates for "from" (S_from)

        # create out dict
        out = dict(model=model_train, T=None, S_from=pred_cord)

        return out


class SpaOTscMap(MapMethodClass):

    # SpaOTsc class
    # Expected Inputs and Outputs
    ins = ["X_to", "X_from"]
    outs = ["T"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass

    @classmethod
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        to_spatial_key: str = "spatial",
        experiment_name: str | None = None,
        seed: int | None = None,
        use_emb: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:

        from spaotsc import SpaOTsc

        if use_emb:
            X_to = input_dict.get("Z_to")
            # anndata object that we map _from_
            X_from = input_dict.get("Z_from")
        else:
            # anndata object that we map _to_
            X_to = input_dict["X_to"]
            pol.check_values(X_to, "X_to")
            # anndata object that we map _from_
            X_from = input_dict["X_from"]
            pol.check_values(X_from, "X_from")

        if isinstance(X_to, pd.DataFrame):
            ad_to = ut.df2ad(X_to)
        if isinstance(X_to, ad.AnnData):
            ad_to = X_to
        if isinstance(X_from, pd.DataFrame):
            ad_from = ut.df2ad(X_from)
        if isinstance(X_from, ad.AnnData):
            ad_from = X_from

        # spatial coordinates of "to"
        if to_spatial_key in ad_to.obsm:
            S_to = ad_to.obsm[to_spatial_key]
        elif "S_to" in input_dict:
            S_to = input_dict["S_to"]
        else:
            raise ValueError('spatial coordinates must be provided for "to" object')

        if seed is not None:
            import ot

            ot.backend.NumpyBackend.seed(seed)

        # Processing the SC data
        # Generate PCA40 from the X_from preprocessed data
        # Taking Alma's suggestion on exposing the HVG parameters
        default_hvg_dict = dict(min_mean=0.0125, max_mean=3, min_disp=0.5)
        hvg_dict = kwargs.get("hvg_dict", {})
        # Checking to fill in default values if not all provided
        for key, val in default_hvg_dict.items():
            if key not in hvg_dict:
                hvg_dict[key] = val
        sc.pp.highly_variable_genes(ad_from, **hvg_dict)
        default_pca_dict = dict(n_comps=40, svd_solver="arpack")
        pca_dict = kwargs.get("pca_dict", {})
        # Checking to fill in defaut values if not all provided
        for key, val in default_pca_dict.items():
            if key not in pca_dict:
                pca_dict[key] = val
        sc.pp.pca(
            ad_from,
            use_highly_variable=kwargs.get("use_highly_variable_genes", True),
            **pca_dict,
        )

        # Determining the SC data dissimilarity based on PCA40
        sc_corr = ut.matrix_correlation(
            ad_from.obsm["X_pca"].T, ad_from.obsm["X_pca"].T
        )
        sc_dmat = np.exp(sc_corr)
        # Generating the DataFrame SC input
        df_sc = ad_from.to_df()

        # Processing the SP data
        # Filtering the nan coordinates in the SP spatial coords data
        goodidx = ~np.isnan(S_to).any(axis=1)
        ad_to = ad_to[goodidx]
        # TODO : double check on how to proceed with modified inputs
        input_dict["X_to"] = ad_to
        # Determining the SP distance matrix based on spatial coordinates
        default_dist_metric = dict(metric="euclidean")
        dist_metric = kwargs.get("dist_metric", default_dist_metric)
        sp_dmat = cdist(
            ad_to.obsm[to_spatial_key], ad_to.obsm[to_spatial_key], **dist_metric
        )
        # Generating the DataFrame SP input
        df_sp = ad_to.to_df()

        # Determining the Cost matrix
        # Finding overlapping genes between SC and SP data
        overlap_genes = list(set(ad_to.var_names).intersection(set(ad_from.var_names)))
        sc_sp_corr = ut.matrix_correlation(
            df_sp[overlap_genes].T, df_sc[overlap_genes].T
        )
        Cost = (np.exp(1 - sc_sp_corr)) ** 2

        # Instantiate the SpaOTsc object
        spaotsc_obj = SpaOTsc.spatial_sc(
            sc_data=df_sc, is_data=df_sp, sc_dmat=sc_dmat, is_dmat=sp_dmat
        )
        # Run optimal transport optimization
        # Note: This step can take an upwards of several hours
        default_transport_plan_dict = dict(
            alpha=0.1, rho=100.0, epsilon=1.0, scaling=False
        )
        transport_plan_dict = kwargs.get("transport_plan_dict", {})
        for key, val in default_transport_plan_dict.items():
            if key not in transport_plan_dict:
                transport_plan_dict[key] = val
        spaotsc_obj.transport_plan(
            Cost,
            **transport_plan_dict,
        )

        # Retrieve optimal transport plan [cells x locations]
        T_soft = spaotsc_obj.gamma_mapping

        # output dict
        out = dict()

        # observation names for "to"
        to_names = ad_to.obs.index.values.tolist()
        # observation named for "from"
        from_names = ad_from.obs.index.values.tolist()

        # transpose map (T) to be in expected format [n_to] x [n_from]
        out["T"] = ut.array_to_sparse_df(T_soft.T, index=to_names, columns=from_names)

        T = out["T"]
        n_rows = ad_to.shape[0]
        n_cols = ad_from.shape[0]

        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (n_rows, n_cols))

        return out


class MoscotMap(MapMethodClass):

    # Method class for moscot
    # github: https://github.com/theislab/moscot
    ins = ["X_to", "X_from"]
    outs = ["T", "moscot_solution"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        genes: List[str] | str | None = None,
        experiment_name: str | None = None,
        return_T_norm: bool = True,
        seed: int | None = None,
        use_emb: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray]:

        from moscot.problems.space import MappingProblem

        # anndata object that we map _to_
        if seed is not None:
            np.random.seed(seed)

        if use_emb:
            X_to = input_dict.get("Z_to")
            # anndata object that we map _from_
            X_from = input_dict.get("Z_from")
            S_to = input_dict.get("S_to")
            X_to = ut.df2ad(X_to)
            X_from = ut.df2ad(X_from)
            X_to.obsm["spatial"] = S_to

        else:
            # anndata object that we map _to_
            X_to = input_dict["X_to"]
            pol.check_values(X_to, "X_to")
            # anndata object that we map _from_
            X_from = input_dict["X_from"]
            pol.check_values(X_from, "X_from")

        # get genes
        if genes is not None:
            genes = ut.list_or_path_get(genes)
            in_from = set(genes).intersection(set(X_from.var_names))
            in_to = set(genes).intersection(set(X_to.var_names))
            genes = list(in_from.intersection(in_to))

        # TODO: this does not run
        prep_kwargs = kwargs.get("prepare", {})

        if not prep_kwargs:
            if "X_pca" in X_from.obsm:
                prep_kwargs["sc_attr"] = dict(attr="obsm", key="X_pca")
            else:
                prep_kwargs["sc_attr"] = dict(attr="X")

        prep_kwargs["var_names"] = genes
        solve_kwargs = kwargs.get("solve", {})

        # set up the mapping problem
        mp = MappingProblem(adata_sc=X_from, adata_sp=X_to)
        # prepare for mapping
        mp = mp.prepare(**prep_kwargs)

        # solve mapping problem
        mp = mp.solve(
            **solve_kwargs,
        )
        transport_plan = mp["src", "tgt"].solution.transport_matrix
        T_soft = np.asarray(transport_plan)

        marginals = dict(a=mp.problems["src", "tgt"].a, b=mp.problems["src", "tgt"].b)

        # output dict
        out = dict()

        to_names = X_to.obs.index.values.tolist()
        from_names = X_from.obs.index.values.tolist()

        out["T"] = ut.array_to_sparse_df(
            T_soft,
            index=to_names,
            columns=from_names,
        )

        out["marginals"] = marginals
        out["converged"] = mp["src", "tgt"].solution.converged

        if return_T_norm:
            T_norm = T_soft / T_soft.sum(axis=0)  # .reshape(-1, 1)
            out["T_norm"] = T_norm

        T = out["T"]

        n_rows = X_to.shape[0]
        n_cols = X_from.shape[0]

        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (n_rows, n_cols))

        return out

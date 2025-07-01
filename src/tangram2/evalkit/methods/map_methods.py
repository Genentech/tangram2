import os.path as osp
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Literal, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch as t
from scipy.sparse import coo_matrix, spmatrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import softmax
from torch.cuda import is_available

import tangram2.evalkit.methods.policies as pol
import tangram2.evalkit.methods.utils as ut
import tangram2.evalkit.utils.transforms as tf
from tangram2.evalkit.methods._methods import MethodClass
from tangram2.evalkit.methods.models import vanilla as vn

from . import _map_utils as mut

__all__ = [
    "RandomMap",
    "ArgMaxCorrMap",
    "Tangram1Map",
    "Tangram2Map",
    "SpaOTscMap",
    "MoscotMap",
]


class MapMethodClass(MethodClass):
    """Mapping method base class"""

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
        """Run the mapping method.

        Parameters
        ----------
        input_dict: Dict[str, Any] - input dictionary
        use_emb: bool :
             (Default value = False)

        Returns
        -------
        Output Dictionary

        """

        pass


class RandomMap(MapMethodClass):
    """Random Mapping Class"""

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
        """Method that randomly maps observations in 'from' to observations in 'to'.

        Parameters
        ----------
        input_dict: Dict[str, Any] :
        seed: int :
             (Default value = 1)
        experiment_name: str | None :
             (Default value = None)
        use_emb: bool :
             (Default value = False)
        **kwargs :

        Returns
        -------
        Output Dictionary

        """
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
    """Pearson Argmax Mapping class"""

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
        """

        Parameters
        ----------
        input_dict: Dict[str, Any] :
        experiment_name: str | None :
             (Default value = None)
        use_emb: bool :
             (Default value = False)

        Returns
        -------
        Output Dictionary with T object

        """

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
    """Tangram Mapping General Class"""

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
        """

        Parameters
        ----------
        input_dict: Dict[str :

        Any] :

        to_spatial_key: str :
             (Default value = "spatial")
        from_spatial_key: str :
             (Default value = "spatial")
        num_epochs: int :
             (Default value = 1000)
        genes: List[str] | str | None :
             (Default value = None)
        experiment_name: str | None :
             (Default value = None)
        use_emb: bool :
             (Default value = False)
        **kwargs :


        Returns
        -------

        """

        if cls.version == "1":
            import tangram as tg
        elif cls.version == "2":
            import tangram2.mapping as tg
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
        mut.pp_adatas(
            ad_from, ad_to, genes=genes, use_filter=kwargs.get("use_filter", True)
        )

        default_mode = "hejin_workflow" if cls.version == "2" else "cells"
        mode = kwargs.pop("mode", default_mode)
        wandb_config = kwargs.pop("wandb_config", {})
        wandb_config["step_prefix"] = experiment_name

        random_state = kwargs.get("random_state")
        if random_state is None:
            random_state = kwargs.get("seed")
            if random_state is None:
                random_state = 42

        cluster_label = None
        if (cls.version == "2") and (mode == "hejin_workflow"):
            cluster_label = kwargs.pop("cluster_label", None)
            if (cluster_label is None) and ("D_from" in input_dict):
                D_from = input_dict["D_from"]
                subset_cols = kwargs.get("subset_labels", D_from.columns)
                if np.sum([x not in D_from.columns for x in subset_cols]) > 0:
                    raise ValueError("subset labels were not in design matrix")
                label_values = D_from.idxmax(axis=1)
                ad_from.obs["__label"] = label_values
                cluster_label = "__label"
            elif cluster_label is None:
                raise ValueError("Can't use workflow without cluster_label")

        method_params = dict(
            adata_sc=ad_from,
            adata_sp=ad_to,
            mode=mode,
            device=("cuda:0" if is_available() else "cpu"),
            num_epochs=num_epochs,
            cluster_label=cluster_label,
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
            method_params["wandb_log"] = kwargs.pop("wandb_log", False)
            method_params["wandb_config"] = wandb_config

        # map cells in "from" to "to"
        tg_out = tg.map_cells_to_space(
            **method_params,
        )

        if cluster_label == "__label":
            del ad_from.obs["__label"]

        # depending on mode and version, treat output differently
        if (cls.version == "2") and (mode == "hejin_workflow"):
            # hejin_workflow mode in tg2 returns a tuple
            # the map (T) and the re-scaled "from" data
            ad_map, X_from_scaled = tg_out
            w = ad_map.uns["coefficient"]
        elif (cls.version == "1") or (cls.version == "2"):
            # all other modes and versions return a single values
            # the map (T)
            ad_map = tg_out
            # set scaled to None for later
            X_from_scaled = None
            w = None
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

        out["w"] = w

        return out


class Tangram1Map(TangramMap):
    """Tangram1 Mapping"""

    # Method class for TangramV1
    version = "1"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class Tangram2Map(TangramMap):
    """Tangram2 Mapping Class"""

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
        """ """
        _funcs = dict(w=cls._save_w)
        return _funcs

    @classmethod
    def _save_w(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        """
        save method for w
        """
        pass


class SpaOTscMap(MapMethodClass):
    """SpaOTsc Mapping class"""

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
        """

        Parameters
        ----------
        input_dict: Dict[str,Any] :
        to_spatial_key: str :
             (Default value = "spatial")
        experiment_name: str | None :
             (Default value = None)
        seed: int | None :
             (Default value = None)
        use_emb: bool :
             (Default value = False)
        **kwargs :

        Returns
        -------
        Output Dictionary with T object

        """

        from tangram2.external.spaotsc import SpaOTsc

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
    """ """

    # Method class for moscot
    # wrapper around: https://github.com/theislab/moscot
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
        """

        Parameters
        ----------
        input_dict: Dict[str, Any] :
        genes: List[str] | str | None :
             (Default value = None)
        experiment_name: str | None :
             (Default value = None)
        return_T_norm: bool :
             (Default value = True)
        seed: int | None :
             (Default value = None)
        use_emb: bool :
             (Default value = False)

        Returns
        -------
        Output Dictionary with T object

        """

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
            out["T_norm"] = ut.array_to_sparse_df(T_norm)

        T = out["T"]

        n_rows = X_to.shape[0]
        n_cols = X_from.shape[0]

        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (n_rows, n_cols))

        return out

    @classmethod
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        seed: int = 1,
        experiment_name: str | None = None,
        use_emb: bool = False,
        num_epochs: int = 1000,
        verbose: bool = False,
        device: str = "cuda:0",
        learning_rate: float = 0.001,
        train_genes: List[str] | None = None,
        normalize: bool = False,
        temperature: float = 1e-2,
        **kwargs,
    ):
        """

        Parameters
        ----------
        input_dict: Dict[str, Any] :
        seed: int :
             (Default value = 1)
        experiment_name: str | None :
             (Default value = None)
        use_emb: bool :
             (Default value = False)
        num_epochs: int :
             (Default value = 1000)
        verbose: bool :
             (Default value = False)
        device: str :
             (Default value = "cuda:0")
        learning_rate: float :
             (Default value = 0.001)
        train_genes: List[str] | None :
             (Default value = None)
        normalize: bool :
             (Default value = False)
        temperature: float :
             (Default value = 1e-2)

        Returns
        -------
        Output Dictionary with T object

        """
        if verbose:
            import tqdm

            iterator = tqdm.tqdm(range(num_epochs))
        else:
            iterator = range(num_epochs)

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

        if isinstance(X_to, ad.AnnData):
            X_to = X_to.to_df()

        if isinstance(X_from, ad.AnnData):
            X_from = X_from.to_df()

        inter = X_from.columns.intersection(X_to.columns)

        if train_genes is not None:
            inter = pd.Index(train_genes).intersection(inter)

        from_names = X_from.index
        to_names = X_to.index

        X_from = X_from.loc[:, inter].values
        X_to = X_to.loc[:, inter].values

        if normalize:
            X_from = X_from / (X_from.sum(axis=0) + 1e-10)
            X_to = X_to / (X_to.sum(axis=0) + 1e-10)

        if "cuda" in device:
            device = device if t.cuda.is_available() else "cpu"

        if seed is not None:
            t.manual_seed(seed)

        model = cls.model(X_to, X_from, device=device, temperature=temperature)
        optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

        for epoch in iterator:

            model.train()  # Set the model to training mode

            loss = model.loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        out = cls.get_out(
            model, to_names=to_names, from_names=from_names, var_names=inter
        )

        return out

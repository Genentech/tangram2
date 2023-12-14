import os.path as osp
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Literal

import anndata as ad
import CeLEry as cel
import numpy as np
import pandas as pd
import scanpy as sc
import tangram as tg1
import tangram2 as tg2
from moscot.problems.space import MappingProblem
from scipy.sparse import coo_matrix, spmatrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from spaotsc import SpaOTsc
from torch.cuda import is_available

from eval._methods import MethodClass

from . import utils as ut


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
        X_to: Any,
        X_from: Any,
        *args,
        S_to: np.ndarray | None = None,
        S_from: np.ndarray | None = None,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:
        pass

    @staticmethod
    def hard_update_out_dict(
        out_dict: Dict[str, spmatrix | np.ndarray],
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        n_rows: int,
        n_cols: int,
        return_sparse: bool,
    ) -> None:
        # helper function to generate the output
        # for hard maps in a sparse format

        ordr = np.argsort(col_idx)
        row_idx = row_idx[ordr]
        col_idx = col_idx[ordr]

        T_sparse = coo_matrix(
            (np.ones(n_cols), (row_idx, col_idx)), shape=(n_rows, n_cols)
        )
        if return_sparse:
            out_dict["T"] = T_sparse
        else:
            out_dict["T"] = T_sparse.toarray()

        return out_dict

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        save_keys: Literal["T", "S_from", "S_to"] | str | None = None,
        compress: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        # shared save function for all Methods

        # get names of data being mapped (from)
        from_names = res_dict["from_names"]
        # get names of data we map onto (to)
        to_names = res_dict["to_names"]

        # to save with loop, less code
        # format is {object_name : (row_names,col_names)}
        # in the output
        save_items = dict(
            T=(to_names, from_names),
            S_from=(from_names, ["x", "y"]),
            S_to=(to_names, ["x", "y"]),
        )

        # filter to only include objects we
        # specified to save, if None then save all
        if save_keys is not None:
            if isinstance(save_keys, str):
                save_keys = [save_keys]
            save_items = {
                key: val for key, val in save_items.items() if key in save_keys
            }

        # save each object
        for key, (index, columns) in save_items.items():
            if verbose:
                print(f"Saving object: {key}")
            if key in res_dict:
                # create data frame
                matrix = res_dict[key]
                df = pd.DataFrame(
                    matrix.toarray() if isinstance(matrix, spmatrix) else matrix,
                    index=index,
                    columns=columns,
                )

                # assemble out path
                out_pth = osp.join(out_dir, key + ".csv")
                # if save as .gz (compressed)
                if compress:
                    out_pth = out_pth + ".gz"
                    ut.to_csv_gzip(df, out_pth)
                else:
                    df.to_csv(out_pth)


class RandomMap(MapMethodClass):
    # class that randomly maps object in "from"
    # to locations in "to"

    ins = ["X_to", "X_from"]
    outs = ["S_from"]

    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @ut.ad2np
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        seed: int = 1,
        return_sparse: bool = False,
        experiment_name: str | None = None,
        **kwargs,
    ):
        # set random seed for reproducibility
        rng = np.random.default_rng(seed)

        # anndata object that we map _to_
        X_to = input_dict["X_to"]
        # anndata object that we map _from_
        X_from = input_dict["X_from"]

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
        cls.hard_update_out_dict(
            out,
            row_idx,
            col_idx,
            n_rows,
            n_cols,
            return_sparse,
        )

        # add standard objects to out dict
        # S_from is spatial coordinates : [n_from] x [n_spatial_dimensions]
        out["S_from"] = input_dict["S_to"][row_idx]

        return out


class ArgMaxCorrMap(MapMethodClass):
    # Method that assigns each observation in "from"
    # to the observation in "to" that it has the highest
    # correlation with, w.r.t. feature expression
    ins = ["X_from", "X_to"]
    outs = ["S_from", "T"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    @classmethod
    @ut.ad2np
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        experiment_name: str | None = None,
        return_sparse: bool = False,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:
        # anndata of "to"
        X_to = input_dict["X_to"]
        # anndata of "from"
        X_from = input_dict["X_from"]

        # n_obs in from
        n_cols = X_from.shape[0]
        # n_obs in to
        n_rows = X_to.shape[0]

        col_idx = np.arange(n_cols).astype(int)

        # get correlation between all observations in "to" and "from"
        sim = ut.matrix_correlation(X_to.T, X_from.T)
        # set nan to max anticorrelation
        sim[np.isnan(sim)] = -np.inf
        # for each observation in "from" get id of
        # observation in "to" that it correlates the most with
        row_idx = np.argmax(sim, axis=1).astype(int)

        # output
        out = dict()

        # update output with sparse map
        cls.hard_update_out_dict(
            out,
            row_idx,
            col_idx,
            n_rows,
            n_cols,
            return_sparse,
        )

        # add standard objects to out dict
        # S_from is spatial coordinates : [n_from] x [n_spatial_dimensions]
        out["S_from"] = input_dict["S_to"][row_idx]

        return out


class TangramMap(MapMethodClass):
    # TangramMap Baseclass

    # tangram module to use
    tg = None
    # version number
    version = None

    ins = ["X_to", "X_from"]
    outs = ["T", "S_from"]

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
        return_sparse: bool = True,
        pos_by_argmax: bool = True,
        pos_by_weight: bool = False,
        num_epochs: int = 1000,
        hard_map: bool = False,
        genes: List[str] | str | None = None,
        experiment_name: str | None = None,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:

        # n_obs in "from"
        n_cols = input_dict["X_from"].shape[0]
        # n_obs in "to"
        n_rows = input_dict["X_to"].shape[0]

        # anndata of "from"
        ad_from = input_dict["X_from"]
        # anndata of "to"
        ad_to = input_dict["X_to"]
        # spatial coordinates of "to"
        S_to = ad_to.obsm[to_spatial_key]

        # get marker genes from tangram
        if genes is not None:
            genes = ut.list_or_path_get(genes)

        # preprocess anndata for mapping
        cls.tg.pp_adatas(ad_from, ad_to, genes=genes)
        mode = kwargs.pop("mode", "cells")
        wandb_config = kwargs.pop("wandb_config", {})
        wandb_config["step_prefix"] = experiment_name

        # map cells in "from" to "to"
        tg_out = cls.tg.map_cells_to_space(
            adata_sc=ad_from,
            adata_sp=ad_to,
            mode=mode,
            device=("cuda:0" if is_available() else "cpu"),
            num_epochs=num_epochs,
            cluster_label=kwargs.pop("cluster_label"),
            random_state=kwargs.pop("random_state", 42),
            wandb_log=kwargs.pop("wandb_log", False),
            wandb_config=wandb_config,
            **kwargs
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
        T_soft = ad_map.X
        # predict coordinates of observations in "from" by weighted average
        S_from = T_soft @ S_to

        # output dict
        out = dict()

        # transpose map (T) to be in expected format [n_to] x [n_from]
        out["T"] = T_soft.T
        # spatial coordinates for "from" : [n_from] x [n_spatial_dims]
        out["S_from"] = S_from
        # anndata with rescaled (with coefficient) "from" data
        out["X_from_scaled"] = X_from_scaled

        # observation names for "to"
        out["to_names"] = ad_to.obs.index.values.tolist()
        # observation named for "from"
        out["from_names"] = ad_from.obs.index.values.tolist()

        # convert soft map (T) to hard map if specified
        if hard_map and (pos_by_argmax or pos_by_weight):
            col_idx = np.arange(n_cols)

            # assign hard positions by argmax
            if pos_by_weight:
                # build kd tree of spatial coordinates in "to"
                kd = cKDTree(S_to)
                _, idxs = kd.query(S_from, k=2)

                row_idx = idxs[:, 1::].flatten()

            # assign hard positions by argmax
            if pos_by_argmax:
                row_idx = np.argmax(T_soft, axis=0).flatten()

            # save hard map as sparse matrix
            cls.hard_update_out_dict(
                out,
                row_idx,
                col_idx,
                n_rows,
                n_cols,
                return_sparse,
            )

        return out


class TangramV1Map(TangramMap):
    # Method class for TangramV1
    tg = tg1
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
    tg = tg2
    version = "2"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class CeLEryMap(MapMethodClass):
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
        return_sparse: bool = True,
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
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:

        # anndata of "from"
        ad_from = input_dict["X_from"]
        # anndata of "to"
        ad_to = input_dict["X_to"]
        # spatial coordinates of "to"
        S_to = ad_to.obsm[to_spatial_key]

        # Processing the SC data
        # Generate PCA40 from the X_from preprocessed data
        # Taking Alma's suggestion on exposing the HVG parameters
        default_hvg_dict = dict(min_mean=0.0125, max_mean=3, min_disp=0.5)
        hvg_dict = kwargs.get("hvg_dict", {})
        # Checking to fill in default values if not all provided
        for key,val in default_hvg_dict.items():
            if key not in hvg_dict:
                hvg_dict[key] = val
        sc.pp.highly_variable_genes(ad_from, **hvg_dict)
        default_pca_dict = dict(n_comps=40, svd_solver="arpack")
        pca_dict = kwargs.get("pca_dict", {})
        # Checking to fill in defaut values if not all provided
        for key,val in default_pca_dict.items():
            if key not in pca_dict:
                pca_dict[key] = val
        sc.tl.pca(ad_from, use_highly_variable=True, **pca_dict)

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
        dist_metric = kwargs.get("dist_metric", {default_dist_metric})
        sp_dmat = cdist(ad_to.obsm[to_spatial_key], ad_to.obsm[to_spatial_key], **dist_metric)
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
        transport_plan_dict = kwargs.get(
            "transport_plan_dict", {}
        )
        for key,val in default_transport_plan_dict.items():
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

        # transpose map (T) to be in expected format [n_to] x [n_from]
        out["T"] = T_soft.T

        # observation names for "to"
        out["to_names"] = ad_to.obs.index.values.tolist()
        # observation named for "from"
        out["from_names"] = ad_from.obs.index.values.tolist()

        return out


class MoscotMap(MapMethodClass):
    # Method class for moscot
    # github: https://github.com/theislab/moscot
    ins = ["X_to", "X_from"]
    outs = ["T"]

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
            # spatial_key: str = "spatial",
            **kwargs,
    ) -> Dict[str, np.ndarray]:
        # single cell anndata
        X_from = input_dict['X_from']
        # spatial anndata
        X_to = input_dict['X_to']

        # get genes
        if genes is not None:
            genes = ut.list_or_path_get(genes)

        prep_kwargs = kwargs.get("prepare", {})
        prep_kwargs["var_names"] = genes
        solve_kwargs = kwargs.get("solve", {})

        # set up the mapping problem
        mp = MappingProblem(adata_sc=X_from, adata_sp=X_to)
        # prepare for mapping
        mp = mp.prepare(
            **prep_kwargs)

        # solve mapping problem
        mp = mp.solve(
            **solve_kwargs,
        )
        transport_plan = mp['src', 'tgt'].solution.transport_matrix
        T_soft = transport_plan.T

        # output dict
        out = dict()

        out['T'] = T_soft
        out['solution'] = mp
        out['to_names'] = X_to.obs.index.values.tolist()
        out['from_names'] = X_from.obs.index.values.tolist()

        if return_T_norm:
            T_norm = T_soft / T_soft.sum(axis=1).reshape(-1,1)
            out['T_norm'] = T_norm

        return out

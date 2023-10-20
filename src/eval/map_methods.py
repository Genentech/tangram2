import os.path as osp
from abc import ABC, abstractmethod
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Literal

import anndata as ad
import CeLEry as cel
import numpy as np
import pandas as pd
import tangram as tg1
import tangram2 as tg2
from scipy.sparse import coo_matrix, spmatrix
from scipy.spatial import cKDTree
from torch.cuda import is_available

from eval._methods import MethodClass

from . import utils as ut


class MapMethodClass(MethodClass):
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
        ordr = np.argsort(col_idx)
        row_idx = row_idx[ordr]
        col_idx = col_idx[ordr]

        T_sparse = coo_matrix(
            (np.ones(n_cols), (row_idx, col_idx)), shape=(n_rows, n_cols)
        )
        if return_sparse:
            out_dict["pred"] = T_sparse
        else:
            out_dict["pred"] = T_sparse.toarray()

        return out_dict

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        save_keys: Literal["T", "S_from", "S_to"] | str | None = None,
        **kwargs,
    ) -> None:

        from_names = res_dict["from_names"]
        to_names = res_dict["to_names"]

        save_items = dict(
            T=(to_names, from_names),
            S_from=(from_names, ["x", "y"]),
            S_to=(to_names, ["x", "y"]),
        )

        if save_keys is not None:
            if isinstance(save_keys, str):
                save_keys = [save_keys]
            save_items = {
                key: val for key, val in save_items.items() if key in save_keys
            }

        for key, (index, columns) in save_items.items():
            if key in res_dict:
                df = pd.DataFrame(
                    res_dict[key],
                    index=index,
                    columns=columns,
                )

                out_pth = osp.join(out_dir, key + ".csv")
                df.to_csv(out_pth)


class RandomMap(MapMethodClass):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @ut.ad2np
    def run(
        cls,
        input_dict: Dict[str, Any],
        seed: int = 1,
        return_sparse: bool = True,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)

        X_to = input_dict["X_to"]
        X_from = input_dict["X_from"]

        n_rows = X_to.shape[0]
        n_cols = X_from.shape[0]

        col_idx = np.arange(n_cols).astype(int)
        row_idx = rng.choice(n_rows, replace=True, size=n_cols).astype(int)

        out = dict()

        cls.hard_update_out_dict(
            out,
            row_idx,
            col_idx,
            n_rows,
            n_cols,
            return_sparse,
        )

        out["T"] = out["pred"]
        out["S_from"] = input_dict["S_to"][row_idx]

        return out


class ArgMaxCorrMap(MapMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    @classmethod
    @ut.ad2np
    def run(
        cls,
        input_dict: Dict[str, Any],
        return_sparse: bool = True,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:
        X_to = input_dict["X_to"]
        X_from = input_dict["X_from"]

        n_cols = X_from.shape[0]
        n_rows = X_to.shape[0]

        col_idx = np.arange(n_cols).astype(int)

        sim = ut.matrix_correlation(X_to.T, X_from.T)
        sim[np.isnan(sim)] = -np.inf
        row_idx = np.argmax(sim, axis=1).astype(int)

        out = dict()

        cls.hard_update_out_dict(
            out,
            row_idx,
            col_idx,
            n_rows,
            n_cols,
            return_sparse,
        )

        out["T"] = out["pred"]
        out["S_from"] = input_dict["S_to"][row_idx]

        return out


class TangramMap(MapMethodClass):
    tg = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass

    @staticmethod
    def get_kwargs(*args, **kwargs):
        out_kwargs = dict()

        markers_path = kwargs.get("marker_path", None)

        if markers_path is not None:
            if markers_path.endswith(".csv"):
                markers = pd.read_csv(markers_path, index_col=0)
                markers = np.reshape(markers.values, -1)
            elif markers_path.endswith(".txt"):
                with open(markers_path, "r") as f:
                    markers = f.readlines()
                    markers = [x.rstrip("\n") for x in markers]
            else:
                raise NotImplementedError

        else:
            markers = None

        out_kwargs["genes"] = markers

        return out_kwargs

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        train_genes: List[str] | None = None,
        to_spatial_key: str = "spatial",
        from_spatial_key: str = "spatial",
        return_sparse: bool = True,
        pos_by_argmax: bool = True,
        pos_by_weight: bool = False,
        num_epochs: int = 1000,
        hard_map: bool = True,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:
        n_cols = input_dict["X_from"].shape[0]
        n_rows = input_dict["X_to"].shape[0]

        ad_from = input_dict["X_from"]
        ad_to = input_dict["X_to"]
        S_to = ad_to.obsm[to_spatial_key]

        cls.tg.pp_adatas(ad_from, ad_to, genes=kwargs.get("genes", None))

        ad_map = cls.tg.map_cells_to_space(
            adata_sc=ad_from,
            adata_sp=ad_to,
            mode=kwargs.get("mode", "cells"),
            device=("cuda:0" if is_available() else "cpu"),
            num_epochs=num_epochs,
        )

        T_soft = ad_map.X
        S_from = T_soft @ S_to

        if hard_map:
            col_idx = np.arange(n_cols)

            if pos_by_argmax:
                kd = cKDTree(S_to)
                _, idxs = kd.query(S_from, k=2)

                row_idx = idxs[:, 1::].flatten()

            if pos_by_weight:
                row_idx = np.argmax(T_soft, axis=0).flatten()

            out = dict()

            cls.hard_update_out_dict(
                out,
                row_idx,
                col_idx,
                n_rows,
                n_cols,
                return_sparse,
            )
        else:
            out = dict(pred=T_soft)

        out["T"] = T_soft.T
        out["S_from"] = S_from

        out["to_names"] = ad_to.obs.index.values.tolist()
        out["from_names"] = ad_from.obs.index.values.tolist()

        return out


class TangramV1Map(TangramMap):
    tg = tg1

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class TangramV2Map(TangramMap):
    tg = tg2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class CeLEryMap(MapMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @staticmethod
    def get_kwargs(*args, **kwargs):
        return dict()

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
        X_to = input_dict["X_to"]
        X_from = input_dict["X_from"]
        X_to.obs[["x_pixel", "y_pixel"]] = X_to.obsm[spatial_key]
        with TemporaryDirectory() as tmpdir:
            model_train = cel.Fit_cord(
                data_train=X_to,
                hidden_dims=hidden_dims,
                num_epochs_max=num_epochs_max,
                path=tmpdir,
                filename="celery_model",
            )
            pred_cord = cel.Predict_cord(
                data_test=X_from, path=tmpdir, filename="celery_model"
            )

        out = dict(pred=pred_cord, model=model_train, T=None, S_from=pred_cord)

        return out

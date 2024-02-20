import os.path as osp
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, spmatrix


class StandardSaveMethods:
    @classmethod
    @property
    def standard_save_funcs(
        cls,
    ):
        _funcs = dict(
            T_soft=cls._save_T_soft,
            T_hard=cls._save_T_hard,
            X_to_pred=cls._save_X_to_pred,
            X_from_pred=cls._save_X_from_pred,
            S_to=cls._save_S_to,
            S_from=cls._save_S_from,
            D_to=cls._save_D_to,
            D_from=cls._save_D_from,
            DEA=cls._save_DEA,
        )
        return _funcs

    @staticmethod
    def save_df(df: pd.DataFrame, out_pth: str, **kwargs):
        compress = kwargs.get("compress", False)
        if compress:
            out_pth = out_pth + ".gz"
            ut.to_csv_gzip(df, out_pth)
        else:
            df.to_csv(out_pth)

    @classmethod
    def _save_T(cls, res_dict: Dict[str, Any], obj: str, out_dir: str, **kwargs):
        obj_name = f"T_{obj}"
        T = res_dict[obj_name]
        # get names of data being mapped (from)
        out_pth = osp.join(out_dir, obj_name + ".csv")

        cls.save_df(T, out_pth, **kwargs)

    @classmethod
    def _save_T_soft(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        cls._save_T(res_dict, "soft", out_dir, **kwargs)

    @classmethod
    def _save_T_hard(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        cls._save_T(res_dict, "hard", out_dir, **kwargs)

    @classmethod
    def _save_S(cls, res_dict: Dict[str, Any], obj: str, out_dir: str, **kwargs):
        # get names of data being mapped (from)
        index = res_dict["{}_names".format(obj)]
        columns = ["x", "y"]

        matrix = res_dict["S_{}".format(obj)]

        df = pd.DataFrame(
            matrix.toarray() if isinstance(matrix, spmatrix) else matrix,
            index=index,
            columns=columns,
        )

        out_pth = osp.join(out_dir, f"S_{obj}" + ".csv")
        cls.save_df(df, out_pth, **kwargs)

    @classmethod
    def _save_S_to(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        cls._save_S(res_dict, "to", out_dir, **kwargs)

    @classmethod
    def _save_S_from(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        cls._save_S(res_dict, "from", out_dir, **kwargs)

    @classmethod
    def _save_X_pred(cls, res_dict: Dict[str, Any], obj: str, out_dir: str, **kwargs):
        # get object name
        obj_name = f"X_{obj}_pred"
        # grab object from results dict
        # if not available return None
        obj_X = res_dict[obj_name]
        # if object is present
        # create a data_frame using the object
        obj_df = pd.DataFrame(
            obj_X,
            index=res_dict[f"{obj}_pred_names"],
            columns=res_dict[f"{obj}_pred_var"],
        )

        out_pth = osp.join(out_dir, obj_name + ".csv")
        cls.save_df(obj_df, out_pth, **kwargs)

    @classmethod
    def _save_X_to_pred(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        **kwargs,
    ):
        cls._save_X_pred(res_dict, "to", out_dir, **kwargs)

    @classmethod
    def _save_X_from_pred(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        cls._save_X_pred(res_dict, "from", out_dir, **kwargs)

    @classmethod
    def _save_D(cls, res_dict: Dict[str, Any], obj: str, out_dir: str, **kwargs):
        df = res_dict[f"D_{obj}"]
        out_pth = osp.join(out_dir, f"D_{obj}" + ".csv", **kwargs)
        cls.save_df(df, out_pth, **kwargs)

    @classmethod
    def _save_D_to(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        **kwargs,
    ):
        cls._save_D(res_dict, "to", out_dir, **kwargs)

    @classmethod
    def _save_D_from(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        **kwargs,
    ):
        cls._save_D(res_dict, "from", out_dir, **kwargs)

    @classmethod
    def _save_DEA(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        dea = res_dict["DEA"]
        for key, df in dea.items():
            out_pth = osp.join(out_dir, f"{key}_vs_rest_dea.csv")
            cls.save_df(df, out_pth, **kwargs)

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from eval._methods import MethodClass

from . import utils as ut


class GroupMethodClass(MethodClass):
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
            input_dict: Dict[str,Any],
        *args,
        **kwargs,
    ) -> Dict[str,pd.DataFrame]:
        pass


class ThresholdGroup(GroupMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
            input_dict: Dict[str,Any],
        feature_name: List[str] | str,
        thres_t: float | Tuple[float, float] = 0.5,
        thres_x: float | Tuple[float, float] = 0.5,
        **kwargs,
    ) -> pd.DataFrame:

        X_to = input_dict['X_to']
        X_from = input_dict['X_from']
        X_to_pred = input_dict['X_to_pred']
        T = input_dict['T']


        if isinstance(feature_name, str):
            feature_name = [feature_name]

        if isinstance(thres_x, (list, tuple)):
            thres_x_low, thres_x_high = thres_x
        else:
            thres_x_low, thres_x_high = thres_x,thres_x

        if isinstance(thres_x, (list, tuple)):
            thres_t_low, thres_t_high = thres_t
        else:
            thres_t_low, thres_t_high = thres_t,thres_t


        Ds_from, Ds_to = [], []

        X_to_pred.columns = X_to_pred.columns.str.lower()

        for feature in feature_name:
            val = X_to_pred[feature.lower()].values

            x_high = val > thres_x_high
            x_low = val < thres_x_low


            D_to = np.zeros((X_to.shape[0], 2))
            D_to[x_high, 1] = 1
            D_to[x_low, 0] = 1

            D_to = pd.DataFrame(
                D_to,
                columns=[f"low_{feature}", f"high_{feature}"],
                index=X_to.obs.index,
            )

            D_from = np.zeros(( X_from.shape[0], 2 ))

            if np.sum(x_high) > 0:
                t_high = T[x_high, :].sum(axis=0) > thres_t_high
            else:
                t_high = np.zeros(T.shape[1]).astype(bool)

            if np.sum(x_low) > 0:
                t_low = T[x_low, :].sum(axis=0) < thres_t_low
            else:
                t_low = np.zeros(T.shape[1]).astype(bool)

            D_from[t_high, 1] = 1
            D_from[t_low, 0] = 1

            D_from = pd.DataFrame(
                D_from,
                columns=[f"low_{feature}", f"high_{feature}"],
                index=X_from.obs.index,
            )

            Ds_from.append(D_from)
            Ds_to.append(D_to)

        Ds_from = pd.concat(Ds_from, axis=1)
        Ds_to = pd.concat(Ds_to, axis=1)

        return dict(D_to=Ds_to, D_from=Ds_from)

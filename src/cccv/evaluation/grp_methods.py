import os.path as osp
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from cccv.evaluation._methods import MethodClass

from . import _grp_utils as gut
from . import utils as ut


class GroupMethodClass(MethodClass):
    # Group Method Baseclass
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
        *args,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        pass


class ThresholdGroup(GroupMethodClass):
    # Threshold based grouping
    # we will base our grouping of the "to"
    # observations the expression of a given feature
    # the "from" observations are grouped based on their
    # how much of their mass is mapped to the two different
    # groups in the "to" data

    ins = ["X_from", "X_to_pred"]
    outs = ["D_to", "D_from"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    @ut.check_in_out
    @gut.add_covariates
    def run(
        cls,
        input_dict: Dict[str, Any],
        feature_name: List[str] | str,
        thres_t: float | Tuple[float, float] = 0.5,
        thres_x: float | Tuple[float, float] = 0.5,
        **kwargs,
    ) -> pd.DataFrame:

        # get anndata of "to"
        # X_to = input_dict["X_to"]
        # get anndata of "from"
        X_from = input_dict["X_from"]
        # get dataframe of "X_to_pred"
        X_to_pred = input_dict["X_to_pred"]
        if isinstance(X_to_pred, ad.AnnData):
            X_to_pred = X_to_pred.to_df()

        # get map (T) : [n_to] x [n_from]
        T = input_dict["T"]

        # make sure feature_name is in list format
        feature_name = ut.listify(feature_name)

        # get and high/low feature (x) thresholds
        if isinstance(thres_x, (list, tuple)):
            thres_x_low, thres_x_high = thres_x
        else:
            thres_x_low, thres_x_high = thres_x, thres_x

        # get high/low map (t) thresholds
        if isinstance(thres_t, (list, tuple)):
            thres_t_low, thres_t_high = thres_t
        else:
            thres_t_low, thres_t_high = thres_t, thres_t

        # create lists to store indicators of each feature
        Ds_from, Ds_to = [], []

        # set feature names in X_to_pred to lowercase
        # this is to match names with specified features
        X_to_pred.columns = X_to_pred.columns.str.lower()

        # iterate over all features
        for feature in feature_name:

            # get feature expression, numpy format
            # feature to lowercase to avoid
            # case-mismatch
            val = X_to_pred[feature.lower()].values

            # boolean vector of observations in "to_pred"
            # that are higher than the "high" threshold
            x_high = val > thres_x_high
            # boolean vector of observations in "to_pred"
            # that are lower than the "low" threshold
            x_low = val < thres_x_low

            # create blank "to" design matrix
            D_to = np.zeros((X_to_pred.shape[0], 2))
            # update "to" design matrix according to
            # high/low assignments
            D_to[x_high, 1] = 1
            D_to[x_low, 0] = 1

            # converts design matrix to data frame
            D_to = pd.DataFrame(
                D_to.astype(int),
                columns=[f"low_{feature}", f"high_{feature}"],
                index=X_to_pred.index,
            )

            # instantiate "from" design matrix
            D_from = np.zeros((X_from.shape[0], 2))

            # get observations in "from" with a mass higher
            # that thres_t_high assigned to the "high" observations
            # in "to"
            if np.sum(x_high) > 0:
                t_high = T[x_high, :].sum(axis=0) >= thres_t_high
            else:
                t_high = np.zeros(T.shape[1]).astype(bool)

            # update "from" design matrix
            D_from[t_high, 1] = 1
            D_from[~t_high, 0] = 1

            # convert "from" design matrix to data frame
            D_from = pd.DataFrame(
                D_from.astype(int),
                columns=[f"nadj_{feature}", f"adj_{feature}"],
                index=X_from.obs.index,
            )

            # save feature specific design matrix
            Ds_from.append(D_from)
            Ds_to.append(D_to)

        # join indicators from all features to create single design matrix
        Ds_from = pd.concat(Ds_from, axis=1)
        Ds_to = pd.concat(Ds_to, axis=1)

        # Ds_from is [n_from] x [2 x n_features]
        # Ds_to is in [n_to] x [2 x n_features]

        # Note: if we specify add covariates
        # additional covariates will be appended to the design matrix

        return dict(D_to=Ds_to, D_from=Ds_from)

import os.path as osp
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd

import telegraph.evaluation.policies as pol
from telegraph.evaluation._methods import MethodClass

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

    ins = ["X_from", "X_to_pred", "T"]
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

        # anndata for predict to data
        X_to_pred = input_dict["X_to_pred"]
        pol.check_values(X_to_pred, "X_to_pred")
        pol.check_type(X_to_pred, "X_to_pred")

        # anndata object that we map _from_
        X_from = input_dict["X_from"]
        pol.check_values(X_from, "X_from")
        pol.check_type(X_from, "X_from")

        if isinstance(X_to_pred, ad.AnnData):
            X_to_pred = X_to_pred.to_df()

        # get map (T) : [n_to] x [n_from]
        T = input_dict["T"]

        n_to = X_to_pred.shape[0]
        n_from = X_from.shape[0]

        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (n_to, n_from))

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

        base_groups = []

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
                t_high = T.values[x_high, :].sum(axis=0) >= thres_t_high
            else:
                t_high = np.zeros(T.shape[1]).astype(bool)

            # update "from" design matrix
            D_from[t_high, 1] = 1
            D_from[~t_high, 0] = 1

            cols = [f"nadj_{feature}", f"adj_{feature}"]
            base_groups.append(cols)
            # convert "from" design matrix to data frame
            D_from = pd.DataFrame(
                D_from.astype(int),
                columns=cols,
                index=X_from.obs.index,
            )

            # save feature specific design matrix
            Ds_from.append(D_from)
            Ds_to.append(D_to)

        # join indicators from all features to create single design matrix
        Ds_from = pd.concat(Ds_from, axis=1)
        Ds_to = pd.concat(Ds_to, axis=1)

        # Ds_to is in [n_to] x [2 x n_features]
        pol.check_values(D_to, "D_to")
        pol.check_type(D_to, "D_to")
        pol.check_dimensions(D_to, "D_to", (n_to, None))

        # Ds_from is [n_from] x [2 x n_features]
        pol.check_values(D_from, "D_from")
        pol.check_type(D_from, "D_from")
        pol.check_dimensions(D_from, "D_from", (n_from, None))

        # Note: if we specify add covariates
        # additional covariates will be appended to the design matrix

        return dict(D_to=Ds_to, D_from=Ds_from, base_groups=base_groups)


class AssociationScore(GroupMethodClass):
    """Association Score between

    how strongly is observation i in 'from' associated
    with feature f in 'from'


    Calculates `Q` where, Q_{if} indicates how strongly associated
    observation i in 'from' is with feature f in 'to_pred'.


    We use the formula: $Q = T^t \cdot X_to_pred = T^t \cdot (T \cdot X_{to\_pred})$

    The dimensions for the objects are:

    X_from : [n_from] x [n_var] (e.g., spatial gene expression)
    X_to_pred : [n_to] x [n_var] (e.g., single cell gene expression)
    T : [n_to] x [n_from] (e.g., map of single cells to visium spots)
    Q : [n_from] x [n_var]

    Q should be a pd.DataFrame object

    """

    ins = ["X_from", "X_to_pred", "T"]
    outs = ["D_to"]

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
        feature_name: List[str] | str | None = None,
        **kwargs,
    ) -> pd.DataFrame:

        X_from = input_dict.get("X_from")
        assert X_from is not None, "X_from needs to be an object"
        # get dataframe of "X_to_pred"
        X_to_pred = input_dict.get("X_to_pred")
        assert X_to_pred is not None, "X_to_pred needs to be an object"

        if isinstance(X_to_pred, ad.AnnData):
            X_to_pred = X_to_pred.to_df()

        # get map (T) : [n_to] x [n_from]
        T = input_dict["T"]

        # compute association between obs i in 'from' with feature f in 'to_pred'
        Q = T.values.T @ X_to_pred.values

        # convert to dataframe
        Q = pd.DataFrame(
            Q,
            index=X_from.obs.index,
            columns=X_to_pred.columns,
        )

        # make sure feature_name is in list format
        feature_name = ut.listify(feature_name)

        # only keep features specified
        if feature_name[0] is not None:
            feature_not_in_vars = [x for x in feature_name if x not in Q.columns]
            assert (
                len(feature_not_in_vars) == 0
            ), "Features: {} were not found in the data".format(
                ",".join(feature_not_in_vars)
            )

            Q = Q.loc[:, feature_name]

        return dict(D_from=Q)

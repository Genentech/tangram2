import os.path as osp
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd

import telegraph.methods.policies as pol
from telegraph.methods._methods import MethodClass

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

    ins = ["X_from", ("X_to_pred", "X_to"), "T"]
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
        add_complement: bool = True,
        **kwargs,
    ) -> pd.DataFrame:

        # anndata for predict to data
        X_to_pred = input_dict.get("X_to_pred")
        if X_to_pred is not None:
            pol.check_values(X_to_pred, "X_to_pred")
            pol.check_type(X_to_pred, "X_to_pred")
        else:
            X_to_pred = input_dict.get("X_to")
            if X_to_pred is None:
                raise ValueError('Must provide one of "X_to" or "X_to_pred"')

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
            D_to = np.zeros((X_to_pred.shape[0], 1))
            # update "to" design matrix according to
            # high/low assignments
            D_to[x_high, 0] = 1

            # converts design matrix to data frame
            to_cols = [f"high_{feature}"]

            D_to = pd.DataFrame(
                D_to.astype(int),
                columns=to_cols,
                index=X_to_pred.index,
            )
            if add_complement:
                D_to[f"low_{feature}"] = 1 - D_to[f"high_{feature}"].values

            # instantiate "from" design matrix
            D_from = np.zeros((X_from.shape[0], 1))

            # get observations in "from" with a mass higher
            # that thres_t_high assigned to the "high" observations
            # in "to"
            if np.sum(x_high) > 0:
                t_high = T.values[x_high, :].sum(axis=0) >= thres_t_high
            else:
                t_high = np.zeros(T.shape[1]).astype(bool)

            # update "from" design matrix
            D_from[t_high, 0] = 1

            from_cols = [f"adj_{feature}"]

            # convert "from" design matrix to data frame
            from_index = (
                X_from.obs.index if isinstance(X_from, ad.AnnData) else X_from.index
            )
            D_from = pd.DataFrame(
                D_from.astype(int),
                columns=from_cols,
                index=from_index,
            )
            if add_complement:
                D_from[f"nadj_{feature}"] = 1 - D_from[f"adj_{feature}"].values

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

        D_to = gut.add_groups_to_old_D(Ds_to, input_dict, target="to")
        D_from = gut.add_groups_to_old_D(Ds_from, input_dict, target="from")

        # Note: if we specify add covariates
        # additional covariates will be appended to the design matrix

        return dict(
            D_to=D_to,
            D_from=D_from,
        )


class AssociationScore(GroupMethodClass):
    """Association Score between

    how strongly is observation i in 'from' associated
    with feature f in 'from'


    Calculates `Q` where, Q_{if} indicates how strongly associated
    observation i in 'from' is with feature f in 'to_pred'.


    We use the formula: :math:`Q = T^t \cdot X_to_pred = T^t \cdot (T \cdot X_{to\_pred})`

    The dimensions for the objects are:

    X_from : [n_from] x [n_var] (e.g., spatial gene expression)
    X_to_pred : [n_to] x [n_var] (e.g., single cell gene expression)
    T : [n_to] x [n_from] (e.g., map of single cells to visium spots)
    Q : [n_from] x [n_var]

    Q should be a pd.DataFrame object

    """

    ins = ["X_from", "T", ("X_to", "X_to_pred")]
    outs = ["D_from"]

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
        if isinstance(X_from, ad.AnnData):
            X_from = X_from.to_df()
        # get dataframe of "X_to_pred"
        X_to_pred = input_dict.get("X_to_pred")
        if X_to_pred is None:
            X_to_pred = input_dict.get("X_to")
            if X_to_pred is None:
                raise ValueError("Must provide 'X_to' or 'X_to_pred'")
            print("Operating with X_to")

        if isinstance(X_to_pred, ad.AnnData):
            X_to_pred = X_to_pred.to_df()

        # get map (T) : [n_to] x [n_from]
        T = input_dict["T"]

        # compute association between obs i in 'from' with feature f in 'to_pred'
        Q = T.values.T @ X_to_pred.values

        # convert to dataframe
        Q = pd.DataFrame(
            Q,
            index=X_from.index,
            columns=X_to_pred.columns,
        )

        Q.columns = Q.columns.str.lower()

        # make sure feature_name is in list format
        feature_name = ut.listify(feature_name)
        # make input features lowercase to make sure it matches with Q columns which are made lower case above
        feature_name = [f.lower() for f in feature_name]

        # only keep features specified
        if feature_name[0] is not None:
            feature_not_in_vars = [x for x in feature_name if x not in Q.columns]
            assert (
                len(feature_not_in_vars) == 0
            ), "Features: {} were not found in the data".format(
                ",".join(feature_not_in_vars)
            )

            Q = Q.loc[:, feature_name]

        D_from = gut.add_groups_to_old_D(Q, input_dict, target="from")

        return dict(D_from=D_from)


class QuantileGroup(GroupMethodClass):

    ins = ["X_from", "X_to", "T"]
    outs = ["D_to", "D_from"]

    @classmethod
    @ut.check_in_out
    @gut.add_covariates
    def run(
        cls,
        input_dict: Dict[str, Any],
        feature_name: List[str] | str,
        q_t: float | Tuple[float, float] = 0.25,
        q_x: float | Tuple[float, float] = 0.25,
        subset_covs: Dict[str, List[str]] | None = None,
        **kwargs,
    ) -> pd.DataFrame:

        # anndata for predict to data
        X_to_use = input_dict.get("X_to_pred")
        if X_to_use is None:
            X_to_use = input_dict.get("X_to")
            X_to_use_name = "X_to"
        else:
            X_to_use_name = "X_to_pred"

        pol.check_values(X_to_use, X_to_use_name)
        pol.check_type(X_to_use, X_to_use_name)

        # anndata object that we map _from_
        X_from = input_dict["X_from"]
        pol.check_values(X_from, "X_from")
        pol.check_type(X_from, "X_from")

        if isinstance(X_to_use, ad.AnnData):
            X_to_use = X_to_use.to_df()

        # get map (T) : [n_to] x [n_from]
        T = input_dict["T"]

        n_to = X_to_use.shape[0]
        n_from = X_from.shape[0]

        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (n_to, n_from))

        to_prefix = ""
        from_prefix = ""
        subset_idxs = {
            "to": np.ones(X_to_use.shape[0]).astype(bool),
            "from": np.ones(X_from.shape[0]).astype(bool),
        }

        from_index = (
            X_from.obs.index if isinstance(X_from, ad.AnnData) else X_from.index
        )

        to_index = (
            X_to_use.obs.index if isinstance(X_to_use, ad.AnnData) else X_to_use.index
        )

        if subset_covs is not None:
            subset_names = dict()
            for target, covs in subset_covs.items():
                D_old = input_dict.get("D_{}".format(target))
                subset_names[target] = []
                if D_old is not None:
                    subset_idx = subset_idxs.pop(target)
                    for cov in covs:
                        if cov in D_old.columns:
                            subset_idx *= D_old[cov].values.astype(bool)
                            subset_names[target].append(cov)
                    subset_idxs[target] = subset_idx
            if "to" in subset_names:
                to_prefix += "_".join(subset_names["to"])
                if len(to_prefix) > 0:
                    to_predix += "_"
                X_to_use = X_to_use[subset_idxs["to"]]

            if "from" in subset_names:
                from_prefix += "_".join(subset_names["from"])
                if len(from_prefix) > 0:
                    from_prefix += "_"
                X_from = X_from[subset_idxs["from"]]

        # make sure feature_name is in list format
        feature_name = ut.listify(feature_name)

        Ds_from, Ds_to = [], []

        # set feature names in X_to_use to lowercase
        # this is to match names with specified features
        X_to_use.columns = X_to_use.columns.str.lower()

        tix = np.where(subset_idxs["to"])[0]
        fix = np.where(subset_idxs["from"])[0]

        T = T.values[tix, :][:, fix]

        # iterate over all features
        for feature in feature_name:

            # get feature expression, numpy format
            # feature to lowercase to avoid
            # case-mismatch
            val = X_to_use[feature.lower()].values
            q_x_high = np.quantile(val, 1 - q_x)
            q_x_low = np.quantile(val, q_x)

            # boolean vector of observations in "to_pred"
            # that are higher than the "high" threshold
            x_high = val >= q_x_high
            # boolean vector of observations in "to_pred"
            # that are lower than the "low" threshold
            x_low = val <= q_x_low

            # create blank "to" design matrix
            D_to = np.zeros((n_to, 2))
            # update "to" design matrix according to
            # high/low assignments
            D_to[tix, :][x_low, 0] = 1
            D_to[tix, :][x_high, 1] = 1

            # converts design matrix to data frame
            to_cols = [f"{to_prefix}low_{feature}", f"{to_prefix}high_{feature}"]

            D_to = pd.DataFrame(
                D_to.astype(int),
                columns=to_cols,
                index=to_index,
            )

            # instantiate "from" design matrix
            D_from = np.zeros((n_from, 2))

            # get observations in "from" with a mass higher
            # that thres_t_high assigned to the "high" observations
            # in "to"

            if np.sum(x_high) > 0:
                T_high_sum = T[x_high, :].sum(axis=0)
                q_t_high = np.quantile(T_high_sum, 1 - q_t)
                t_high = T[x_high, :].sum(axis=0) >= q_t_high
            else:
                t_high = np.zeros(T.shape[1]).astype(bool)

            if np.sum(x_low) > 0:
                T_low_sum = T[x_low, :].sum(axis=0)
                q_t_low = np.quantile(T_low_sum, 1 - q_t)
                t_low = T[x_low, :].sum(axis=0) >= q_t_low
            else:
                t_low = np.zeros(T.shape[1]).astype(bool)

            is_both = np.where(t_low & t_high)[0]

            # update "from" design matrix
            t_low = np.where(t_low)[0]
            t_high = np.where(t_high)[0]

            D_from[fix[t_low], 0] = 1
            D_from[fix[t_high], 1] = 1

            D_from[fix[is_both], 0] = 0
            D_from[fix[is_both], 1] = 0

            from_cols = [f"{from_prefix}nadj_{feature}", f"{from_prefix}adj_{feature}"]
            # convert "from" design matrix to data frame

            D_from = pd.DataFrame(
                D_from.astype(int),
                columns=from_cols,
                index=from_index,
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

        D_to = gut.add_groups_to_old_D(Ds_to, input_dict, target="to")
        D_from = gut.add_groups_to_old_D(Ds_from, input_dict, target="from")

        return dict(
            D_to=D_to,
            D_from=D_from,
        )


class DistanceBasedGroup(GroupMethodClass):
    """Grouping Method based on distance between observations

    This methods is primarily recommended when using single cell resolution spatial data (e.g.,
    Xenium, CosMX, and Vizgen). It allows you to specify a reference feature and a quantile, all
    cells in X_to expressing that feature at a level above the quantile will be considered as a
    "high" group, cells below the quantile threshold are "low". The K nearest neighbors of each
    high cell that are not in the high group will be considered as adjacent cells to that feature.

    """

    ins = [("X_to_pred", "X_to"), "S_to"]
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
        q_high: float | Tuple[float, float] = 0.95,
        add_complement: bool = True,
        k=5,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """

        Args:
            input_dict: standard input dictionary
            feature_name: name of feature to base high/low groups on
            q_high: quantile to separate feature expression w.r.t.
            add_complement: include both high/low and adj/nadj groups in output design matrix
            k: number of spatial neighbors
        Returns:
            Dictionary with design matrix for to (D_to)

        """
        from scipy.spatial import cKDTree

        # anndata for predict to data
        X_to_obj = input_dict.get("X_to_pred")
        if X_to_obj is not None:
            pol.check_values(X_to_obj, "X_to_pred")
            pol.check_type(X_to_obj, "X_to_pred")
        else:
            X_to_obj = input_dict.get("X_to")
            if X_to_obj is None:
                raise ValueError('Must provide one of "X_to" or "X_to_pred"')

        n_obs = X_to_obj.shape[0]

        S_to = input_dict.get("S_to")
        if S_to is None:
            raise ValueError("'S_to' must be given")

        feature_name = ut.listify(feature_name)
        D_to = pd.DataFrame(
            [],
            index=(
                X_to_obj.obs_names
                if isinstance(X_to_obj, ad.AnnData)
                else X_to_obj.index
            ),
        )

        for feature_i in feature_name:

            if isinstance(X_to_obj, ad.AnnData):
                fv_i = X_to_obj.obs_vector(feature_i)
            else:
                fv_i = X_to_obj[feature_i].values

            assert len(S_to) == n_obs, "S_to and X_to_{pred} is not of the same size"

            q_val_i = np.quantile(fv_i, q_high)
            is_high_i = fv_i >= q_val_i
            kd = cKDTree(S_to[~is_high_i])
            is_low_i = np.where(~is_high_i)[0]
            _, idxs_i = kd.query(S_to[is_high_i], k=k)
            idxs_i = np.unique(idxs_i.flatten())
            idxs_i = is_low_i[idxs_i]

            indicator_i = np.zeros(n_obs)
            indicator_i[idxs_i] = 1
            D_to[f"adj_{feature_i}"] = indicator_i
            D_to[f"high_{feature_i}"] = is_high_i.astype(float)
            if add_complement:
                D_to[f"nadj_{feature_i}"] = 1 - indicator_i

        return dict(D_to=D_to)

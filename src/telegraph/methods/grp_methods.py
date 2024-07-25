import os.path as osp
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import telegraph.methods.policies as pol
from telegraph.methods._methods import MethodClass

from . import _grp_utils as gut
from . import transforms as tf
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

        # if subset_covs is not None:
        # subset_names = dict()
        # for target, covs in subset_covs.items():
        #     D_old = input_dict.get("D_{}".format(target))
        #     subset_names[target] = []
        #     if D_old is not None:
        #         subset_idx = subset_idxs.pop(target)
        #         for cov in covs:
        #             if cov in D_old.columns:
        #                 subset_idx *= D_old[cov].values.astype(bool)
        #                 subset_names[target].append(cov)
        #         subset_idxs[target] = subset_idx
        # if "to" in subset_names:
        #     to_prefix += "_".join(subset_names["to"])
        #     if len(to_prefix) > 0:
        #         to_predix += "_"
        #     X_to_use = X_to_use[subset_idxs["to"]]

        # if "from" in subset_names:
        #     from_prefix += "_".join(subset_names["from"])
        #     if len(from_prefix) > 0:
        #         from_prefix += "_"
        #     X_from = X_from[subset_idxs["from"]]

        # make sure feature_name is in list format
        feature_name = ut.listify(feature_name)

        Ds_from, Ds_to = [], []

        # set feature names in X_to_use to lowercase
        # this is to match names with specified features
        X_to_use.columns = X_to_use.columns.str.lower()

        tix = np.where(subset_idxs["to"])[0]
        fix = np.where(subset_idxs["from"])[0]

        T = T.values[tix, :][:, fix]

        criteria = [
            feature if isinstance(feature, (list, tuple)) else (feature, None)
            for feature in feature_name
        ]

        # iterate over all features
        for criterion in criteria:

            feature, subset = criterion[0], criterion[1::]

            subset_name = ""
            if subset[0] is not None:
                D_from_old = input_dict.get("D_from")
                if D_from_old is not None:
                    is_criteria = np.all(D_from_old.loc[:, subset].values == 1, axis=1)
                    use_from_idx = np.where(is_criteria)[0]
                    subset_name = "_" + "_".join(subset)
                else:
                    raise ValueError("Subset not in D_from")
            else:
                use_from_idx = np.arange(T.shape[1])

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
                T_high_sum = T[x_high, :][:, use_from_idx].sum(axis=0).flatten()
                q_t_high = np.quantile(T_high_sum, 1 - q_t)
                t_high = T_high_sum >= q_t_high
            else:
                t_high = np.zeros(T.shape[1]).astype(bool)

            if np.sum(x_low) > 0:
                T_low_sum = T[x_low, :][:, use_from_idx].sum(axis=0).flatten()
                q_t_low = np.quantile(T_low_sum, 1 - q_t)
                t_low = T_low_sum >= q_t_low
            else:
                t_low = np.zeros(T.shape[1]).astype(bool)

            is_both = np.where(t_low & t_high)[0].astype(int)
            is_both = use_from_idx[is_both]

            # update "from" design matrix
            t_low = np.where(t_low)[0]
            t_low = use_from_idx[t_low]
            t_high = np.where(t_high)[0]
            t_high = use_from_idx[t_high]

            D_from[fix[t_low], 0] = 1
            D_from[fix[t_high], 1] = 1

            D_from[fix[is_both], 0] = 0
            D_from[fix[is_both], 1] = 0

            from_cols = [
                f"{from_prefix}nadj_{feature}{subset_name}",
                f"{from_prefix}adj_{feature}{subset_name}",
            ]
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


class SpotBasedGroup(GroupMethodClass):
    ins = ["X_from", "T"]
    outs = ["D_to", "D_from"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        feature_name: List[str] | str,
        p_thres: float = 0.8,
        add_complement: bool = True,
    ) -> Dict[str, pd.DataFrame]:

        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture as gmm

        X_from = input_dict.get("X_from")
        if X_from is None:
            raise ValueError("Must provide X_from for SpotBaseGroup to work")

        pol.check_values(X_from, "X_from")
        if isinstance(X_from, ad.AnnData):
            X_from = X_from.to_df()

        T = input_dict.get("T")
        pol.check_values(T, "T")
        cidx = np.argmax(T.values, axis=0)
        n_obs = T.shape[1]

        D_from = pd.DataFrame(
            [],
            index=(
                X_from.obs_names if isinstance(X_from, ad.AnnData) else X_from.index
            ),
        )

        def _cluster_helper(_vals, p_thres, pca=False):
            vals = _vals.copy()
            cluster_model = gmm(n_components=2)

            if len(vals.shape) == 1:
                vals = vals[:, None]

            if pca:
                vals = PCA(n_components=25).fit_transform(vals)

            cluster_model.fit(vals)
            clu_prob = cluster_model.predict_proba(vals)

            if pca:
                clu_weights = cluster_model.weights_.flatten()
                clu_ordr = np.argsort(clu_weights)[0]
            else:
                clu_means = cluster_model.means_.flatten()
                clu_ordr = np.argsort(clu_means)[-1]

            clu_prob = clu_prob[:, clu_ordr].flatten()
            return clu_prob > p_thres, clu_prob < 1 - p_thres

        for feature_i in ut.listify(feature_name):

            if isinstance(X_from, ad.AnnData):
                fv_i = X_from.obs_vector(feature_i)
            else:
                fv_i = X_from[feature_i].values

            is_high_i, is_low_i = _cluster_helper(fv_i, p_thres)
            idx_high_i = np.unique(cidx[is_high_i])
            in_high_spot = np.isin(cidx, idx_high_i)
            idx_rec_elegible_i = np.where(in_high_spot & (~is_high_i))[0]
            indicator_i = np.zeros(n_obs)

            indicator_i[idx_rec_elegible_i] = 1
            D_from[f"adj_{feature_i}"] = indicator_i
            D_from[f"nadj_{feature_i}"] = (~in_high_spot).astype(float)
            D_from[f"high_{feature_i}"] = is_high_i.astype(float)
            D_from[f"low_{feature_i}"] = is_low_i.astype(float)

        D_from = gut.add_groups_to_old_D(D_from, input_dict, target="from")

        return dict(D_from=D_from)


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
    # @gut.add_covariates
    def run(
        cls,
        input_dict: Dict[str, Any],
        feature_name: List[str] | str,
        q_high: float | Tuple[float, float] = 0.95,
        add_complement: bool = True,
        subset_receiver: List[str] | str | None = None,
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

        D_to_old = input_dict.get("D_to")

        D_to = pd.DataFrame(
            [],
            index=(
                X_to_obj.obs_names
                if isinstance(X_to_obj, ad.AnnData)
                else X_to_obj.index
            ),
        )

        if D_to_old is not None and subset_receiver is not None:
            is_receiver = np.zeros(len(X_to_obj))
            subset_receiver = ut.listify(subset_receiver)
            for sub in subset_receiver:
                is_receiver += D_to_old[sub].values
            is_receiver = is_receiver > 0
        else:
            is_receiver = np.ones(len(X_to_obj)).astype(bool)

        for feature_i in feature_name:

            if isinstance(X_to_obj, ad.AnnData):
                fv_i = X_to_obj.obs_vector(feature_i)
            else:
                fv_i = X_to_obj[feature_i].values

            assert len(S_to) == n_obs, "S_to and X_to_{pred} is not of the same size"

            km = KMeans(n_clusters=2)
            clu_idx = km.fit_predict(fv_i[:, None])
            clu_cnt = km.cluster_centers_.flatten()
            ordr = np.argsort(clu_cnt)
            low_clu = ordr[0]
            high_clu = ordr[-1]
            is_high_i = clu_idx == high_clu

            is_target = np.where(~is_high_i)[0]

            kd = cKDTree(S_to.values[is_target])

            _, idxs_i = kd.query(S_to.values[is_high_i], k=k)
            idxs_i = np.unique(idxs_i.flatten())
            idxs_i = is_target[idxs_i]
            idxs_i = np.intersect1d(idxs_i, np.where(is_receiver)[0])

            indicator_i = np.zeros(n_obs)
            indicator_i[idxs_i] = 1
            D_to[f"adj_{feature_i}"] = indicator_i
            D_to[f"high_{feature_i}"] = is_high_i.astype(int)
            if add_complement:
                complement = 1 - indicator_i
                complement *= is_receiver.astype(float)
                complement[is_high_i] = 0
                D_to[f"nadj_{feature_i}"] = complement

        if D_to_old is not None:
            overlap = [x for x in D_to.columns if x in D_to_old.columns]
            if len(overlap) > 1:
                D_to_old.drop(columns=overlap, inplace=True)
            D_to_new = pd.concat((D_to_old, D_to), axis=1)

        return dict(D_to=D_to_new)


class RandomGroup(GroupMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    def _get_random_groups(
        cls,
        groups: List[str],
        size: int,
        probs: np.ndarray | None = None,
    ):

        if probs is not None:
            if len(probs) != len(groups):
                raise ValueError("length of probs must be same as length of groups")
            else:
                probs = np.ones(len(groups)) / len(groups)

        return np.random.choice(groups, p=probs, size=size, replace=True)

    @classmethod
    def run_with_adata(
        cls,
        adata: ad.AnnData,
        groups: List[str],
        subset_col: str | None = None,
        label_col: str = "random_groups",
        subset_labels: str | List[str] = None,
        background_label="background",
        probs: np.ndarray | None = None,
    ):

        is_label = ut.get_adata_subset_idx(adata, subset_col, subset_labels)

        n_obs = len(adata)
        n_sub_obs = len(is_label)

        new_label_array = np.array([background_label] * n_obs, dtype=object)
        new_labels = cls._get_random_groups(groups, size=n_sub_obs, probs=probs)
        new_label_array[is_label] = new_labels

        adata.obs[label_col] = new_label_array

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplemented(
            "The RanomGroup Method has not yet been implemented for the telegraph workflow",
        )


class ClusterGroup(GroupMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    def _get_cluster_groups(
        cls,
        X: np.ndarray,
        n_clusters: int = 2,
        groups: str | List[str] = None,
        method: Literal["kmeans"] = "kmeans",
    ):

        from sklearn import cluster as clu
        from sklearn.decomposition import PCA

        if groups is not None:
            n_clusters = len(groups)
            clu_map = {k: g for k, g in enumerate(groups)}
        else:
            clu_map = {k: k for k in range(n_clusters)}

        if X.shape[1] > 50:
            E = PCA(n_components=50).fit_transform(X)
        else:
            E = X

        clu_algs = {
            "kmeans": clu.KMeans,
            "mean_shift": clu.MeanShift,
            "spectral": clu.SpectralClustering,
        }

        clu_alg = clu_algs.get(method)
        if clu_algs is None:
            raise ValueError(
                "We do not support the clustering method {}".format(method)
            )

        clu_idx = clu_alg(n_clusters=n_clusters).fit_predict(E)
        clu_idx = np.array([clu_map[k] for k in clu_idx])

        return clu_idx

    @classmethod
    def run_with_adata(
        cls,
        adata: ad.AnnData,
        groups: List[str],
        subset_col: str | None = None,
        label_col: str = "random_groups",
        subset_labels: str | List[str] = None,
        background_label="background",
        layer: str | None = None,
        cluster_method: str = "kmeans",
        n_clusters: int = 2,
    ):

        is_label = ut.get_adata_subset_idx(adata, subset_col, subset_labels)

        n_obs = len(adata)
        n_sub_obs = len(is_label)

        new_label_array = np.array([background_label] * n_obs, dtype=object)

        new_labels = cls._get_cluster_groups(
            adata.to_df(layer=layer).values[is_label],
            groups=groups,
            method=cluster_method,
            n_clusters=n_clusters,
        )

        new_label_array[is_label] = new_labels

        adata.obs[label_col] = new_label_array

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplemented(
            "The ClusterGroup Method has not yet been implemented for the telegraph workflow",
        )

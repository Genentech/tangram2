import os.path as osp
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from . import _dea_utils as dut
from . import policies as pol
from . import utils as ut
from ._dea_utils import DEA
from ._methods import MethodClass

# import telegraph.methods.policies as pol
# from telegraph.methods._methods import MethodClass


class DEAMethodClass(MethodClass):
    # DEA Method Baseclass
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
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        pass


class ScanpyDEA(DEAMethodClass):
    # Scanpy DEA Method class
    # allows you to execute scanpy.tl.rank_genes
    # using a design matrix and input data
    ins = ["D_from", "D_to", "X_from", "X_to_pred"]
    outs = ["DEA"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    # @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        groups: List[str] | str = "all",
        method: str = "wilcoxon",
        sort_by: str = "pvals_adj",
        pval_cutoff: float | None = None,
        mode: Literal["pos", "neg", "both"] = "both",
        method_kwargs: Dict[str, Any] = {},
        normalize: bool = False,
        subset_features: Dict[str, List[str]] | Dict[str, str] | None = None,
        min_group_obs: int = 2,
        target: List[str] | str = "both",
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:

        # groups = [ ('group_1,group_2)]

        if target == "both":
            _target = ["to", "from"]
        else:
            _target = ut.listify(target)

        # data frame of predicted "to" data : [n_to] x [n_from_features]
        X_to_pred = input_dict.get("X_to_pred")
        if X_to_pred is None:
            X_to_pred = input_dict.get("X_to")
        # anndata of "from" : [n_from] x [n_from_features]
        X_from = input_dict.get("X_from")
        # design matrix for "to" : [n_to] x [n_to_covariates]
        D_to = input_dict.get("D_to")
        # design matrix for "from" : [n_from] x [n_from_covariates]
        D_from = input_dict.get("D_from")

        X_from = input_dict.get("X_from")

        objects = dict()

        if ("to" in _target) and (D_to is not None) and (X_to_pred is not None):
            pol.check_values(X_to_pred, "X_to_pred")
            pol.check_type(X_to_pred, "X_to_pred")

            n_to = X_to_pred.shape[0]

            pol.check_values(D_to, "D_to")
            pol.check_type(D_to, "D_to")
            pol.check_dimensions(D_to, "D_to", (n_to, None))

            # update objects
            objects["to"] = dict(D=D_to, X=X_to_pred)

        if ("from" in _target) and (D_from is not None) and (X_from is not None):
            pol.check_values(X_from, "X_from")
            pol.check_type(X_from, "X_from")

            n_from = X_from.shape[0]

            pol.check_values(D_from, "D_from")
            pol.check_type(D_from, "D_from")
            pol.check_dimensions(D_from, "D_from", (n_from, None))

            objects["from"] = dict(D=D_from, X=X_from)

        out = dict()

        for obj_name in objects.keys():
            D = objects[obj_name]["D"].copy()
            X = objects[obj_name]["X"].copy()

            match groups:
                case str():
                    _groups = [(groups)]
                case (str(), str()):
                    _groups = [groups]
                case _:
                    _groups = groups

            for group_pair in _groups:

                D_new, grp_1, grp_2 = dut.scanpy_dea_labels_from_D(D, group_pair)

                adata = dut.anndata_from_X_and_D(X, D_new)
                labels = adata.obs["label"].values

                if (grp_1 in labels) and ((grp_2 in labels) or (grp_2 == "rest")):
                    # execute DE test
                    sc.tl.rank_genes_groups(
                        adata,
                        groupby="label",
                        groups=[
                            grp_1
                        ],  # groups _must_ be a list or str, not np.ndarray
                        reference=grp_2,
                        method=method,
                        **method_kwargs,
                    )

                    dedf = sc.get.rank_genes_groups_df(
                        adata,
                        group=grp_1,
                        pval_cutoff=pval_cutoff,
                    )

                    dedf.rename(
                        columns={
                            "pvals": DEA.p_value.value,
                            "pvals_adj": DEA.adj_p_value,
                            "name": DEA.feature.value,
                            "score": DEA.score.value,
                        }
                    )

                else:
                    dedf = dut.get_empty_dea_df()

                name = f"{obj_name}_{grp_1}_vs_{grp_2}"
                out[name] = dedf

        for key in out.keys():
            dedf = out[key]
            if mode == "both":
                pass
            elif mode == "pos":
                out[key] = dedf[dedf["scores"].values > 0]
            elif mode == "neg":
                out[key] = dedf[dedf["scores"].values < 0]
            else:
                raise NotImplementedError

        return dict(DEA=out)


class GLMDEA(DEAMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    ins = ["D_from", "D_to", "X_from", "X_to"]
    outs = ["DEA"]

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        family: str = "negative.binomial",
        target: Literal["to", "from", "both"] = "both",
        use_covariates: List[str] | str | None = None,
        drop_covariates: List[str] | str | None = None,
        merge_covariates: List[List[str]] | None = None,
        use_pred: bool | Dict[str, bool] = True,
        subset_features: List[str] | str | None = None,
        use_obs_with_covariates: bool = False,
        fit_intercept: bool = False,
        mht_method: str = "fdr_bh",
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        # make method specific imports
        import glum as gl
        from statsmodels.stats.multitest import multipletests as mht

        # get what object to target
        if target == "both":
            target_l = ["to", "from"]
        else:
            target_l = ut.listify(target)

        # instantiate output dictionary
        out = dict()

        # iterate over target objects
        for tgt in target_l:
            # get design matrix
            D = input_dict.get("D_{}".format(tgt))
            # assert design matrix is defined
            assert D is not None, f"D_{tgt} is not defined"

            # copy object to prevent permanent changes
            D_inp = D.copy()

            pol.check_values(D_inp, f"D_{tgt}")
            pol.check_type(D_inp, f"D_{tgt}")

            # get name of feature object
            X_name = "X_{}".format(tgt)
            # if pred should be used modify name
            if isinstance(use_pred, dict):
                if use_pred.get(tgt, False):
                    X_name += "_pred"
            elif isinstance(use_pred, bool):
                if use_pred and (input_dict.get(f"X_{tgt}_pred", None) is not None):
                    X_name += "_pred"

            # get X object
            X = input_dict.get(X_name)
            # make sure X object is defined
            assert X is not None, f"{X_name} is not defined"
            # copy object to prevent permanent changes
            X_inp = X.copy()

            pol.check_values(X_inp, X_name)
            pol.check_type(X_inp, X_name)

            # convert X to pandas data frame
            if isinstance(X_inp, ad.AnnData):
                X_inp = X_inp.to_df()

            if merge_covariates is not None:
                if isinstance(merge_covariates[0], str):
                    merge_covariates = [merge_covariates]

                drop_cols = []
                for merge_cols in merge_covariates:
                    is_new = np.all(D_inp[merge_cols].values, axis=1)
                    new_label = np.zeros(len(D_inp))
                    new_label[is_new] = 1
                    new_label_name = "_".join(merge_cols)
                    D_inp[new_label_name] = new_label
                    drop_cols += drop_cols

                    if use_covariates is not None:
                        use_covariates = [
                            x for x in use_covariates if x not in merge_cols
                        ]
                        use_covariates += [new_label_name]

                    D_inp.drop(columns=drop_cols, inplace=True)

            # subset to specified covariates, if None then use all
            if use_covariates is not None:
                adj_use_covariates = ut.listify(use_covariates)
                if isinstance(adj_use_covariates[0], list):
                    adj_use_covariates = [
                        x[0:-1] if isinstance(x, list) else [x]
                        for x in adj_use_covariates
                    ]
                    adj_use_covariates = [x for y in adj_use_covariates for x in y]
                    left_out_covariates = {
                        x[0]: x[1] for x in use_covariates if len(x) == 2
                    }
                else:
                    adj_use_covariates = use_covariates
                    left_out_covariates = {}

                D_inp = D_inp[adj_use_covariates]

            # drop covariates if specified
            if drop_covariates is not None:
                keep_cols = [
                    x for x in D.columns if x not in ut.listify(drop_covariates)
                ]
                D_inp = D_inp[keep_cols]

            if use_obs_with_covariates:
                has_covs = np.any(D_inp.values != 0, axis=1)
                X_inp = X_inp.iloc[has_covs, :]
                D_inp = D_inp.iloc[has_covs, :]

            # check what features to test
            _features = X_inp.columns if subset_features is None else subset_features

            # default glm parameters
            # from: https://glum.readthedocs.io/en/latest/glm.html#glum.GeneralizedLinearRegressor (2024-01-25)
            glm_default_params = dict(
                alpha=0,
                l1_ratio=0,
                P1="identity",
                P2="identity",
                link="auto",
                solver="auto",
                max_iter=100,
                gradient_tol=None,
                step_size_tol=None,
                hessian_approx=0,
                warm_start=False,
                alpha_search=False,
                alphas=None,
                n_alphas=100,
                min_alpha_ratio=None,
                min_alpha=None,
                start_params=None,
                selection="cyclic",
                random_state=None,
                copy_X=None,
                check_input=True,
                verbose=0,
                scale_predictors=False,
                lower_bounds=None,
                upper_bounds=None,
                A_ineq=None,
                b_ineq=None,
                force_all_finite=True,
                drop_first=False,
                robust=True,
                expected_information=False,
            )

            # update parameters from kwargs
            glm_params = ut.merge_default_dict_with_kwargs(glm_default_params, kwargs)
            # add intercept
            glm_params["fit_intercept"] = fit_intercept
            # glm_params['fit_intercept'] = True
            # D_inp['intercept'] = 1

            # add distribution family
            glm_params["family"] = family

            # prepare results dictionary
            res = {
                "covariate": [],
                DEA.coeff.value: [],
                DEA.p_value.value: [],
                DEA.feature.value: [],
            }

            # iterate over features
            for feature in _features:
                # prepare dependent variable
                y = X_inp[feature].values.flatten().astype(np.float64)
                if np.sum(y) == 0:
                    continue
                # create glm object
                glm = gl.GeneralizedLinearRegressor(**glm_params)
                # fit glm object
                glm.fit(X=D_inp, y=y)
                # get coefficient table
                coef_table = glm.coef_table(X=D_inp.astype(np.float64), y=y)
                # transfer values to results dictionary
                res["covariate"] += coef_table.index.tolist()
                res[DEA.p_value.value] += coef_table["p_value"].values.tolist()
                res[DEA.coeff.value] += coef_table["coef"].values.tolist()
                res[DEA.feature.value] += [feature] * coef_table.shape[0]

            # because we need all p-values when doing the correction
            _, p_val_adj, _, _ = mht(
                res[DEA.p_value.value],
                method=mht_method,
            )

            # add adjusted p-values
            res[DEA.adj_p_value.value] = p_val_adj

            # convert to dataframe
            res = pd.DataFrame(res)
            # split by covariate
            tgt_out = {
                f"{tgt}_{cov}": res.iloc[res["covariate"].values == cov].copy()
                for cov in D_inp.columns
            }

            # update output dictionary
            out.update(tgt_out)

        if "DEA" in input_dict:
            input_dict["DEA"].update(out)
            out = input_dict["DEA"]

        return dict(DEA=out)


class LRDEA(DEAMethodClass):
    ins = [("D_from", "D_to"), ("X_from", "X_to", "X_to_pred")]
    outs = ["DEA"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    # @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        covariates: Dict[str, str] | str | List[str],
        sort_by: str = "pvals_adj",
        pval_cutoff: float | None = None,
        # subset_features: Dict[str, List[str]] | Dict[str, str] | None = None,
        random_state: int = 0,
        C: float = 1,
        max_iter: int = 100,
        target: Literal["both", "to", "from"] = "from",
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:

        from sklearn.linear_model import LogisticRegression as LR

        if target == "both":
            _target = ["to", "from"]
            if not isinstance(covariates, dict):
                raise ValueError(
                    "If using both to and from as targets, provide covariates in a dict"
                )
        else:
            _target = target
            if isinstance(covariates, str):
                covariates = {_target: [covariates]}
            elif isinstance(covariates, list):
                covariates = {_target: covariates}

        for tgt, val in covariates.items():
            if isinstance(val, str):
                covariates[key] = [val]

        out = dict()

        for tgt in _target:
            D_tgt = input_dict.get("D_{}".format(tgt))

            if D_tgt is None:
                raise ValueError("Could not find D_{} in the input_dict".format(tgt))

            X_tgt = input_dict.get("X_{}")
            if X_tgt is None:
                X_tgt = input_dict.get("X_{}_pred")
            if X_tgt is None:
                raise ValueError(
                    "Could not find X_{} or X_{}_pred in the input_dict".format(
                        tgt, tgt
                    )
                )

            if isinstance(X_tgt, ad.AnnData):
                X_tgt = X_tgt.to_df()

            for cov in covariates[tgt]:
                lr = LR(
                    random_state=random_state,
                    penalty="l1",
                    solver="saga",
                    C=C,
                    max_iter=max_iter,
                )

                lr = lr.fit(X_tgt.values, D_tgt[cov].values.astype(float))

                coef = lr.coef_.flatten()

                dea = pd.DataFrame(
                    {
                        DEA.score.value: coef,
                        DEA.feature.value: X_tgt.columns.tolist(),
                    }
                )

                out[f"{tgt}_{cov}"] = dea

            return dict(DEA=out)


class RandomFeatureDEA(DEAMethodClass):
    ins = [("X_from", "X_to", "X_to_pred")]
    outs = ["DEA"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    def get_random_dea(cls, names, lfc_min_max: Tuple[float, float] = (-2, 2)):

        n_obs = len(names)
        pvals = np.random.uniform(0, 1, size=n_obs)
        lfc_min, lfc_max = lfc_min_max
        lfc = np.random.uniform(lfc_min, lfc_max, size=n_obs)

        dedf = pd.DataFrame(
            {
                DEA.feature.value: names,
                DEA.logfold.value: lfc,
                DEA.adj_p_value.value: pvals,
            },
        )

        return dedf

    @classmethod
    def run_with_adata(
        cls,
        adata: ad.AnnData,
    ):

        names = adata.var_names.tolist()
        dedf = cls.get_random_dea(
            names,
        )
        return dedf

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        *args,
        **kwargs,
    ):
        raise NotImplemented(
            "The RandomFeatureDEA  method has not been implemented for telegraph workflow use yet"
        )


class HVGFeatureDEA(DEAMethodClass):
    ins = [("X_from", "X_to", "X_to_pred")]
    outs = ["DEA"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    def get_hvg_dea(cls, X_df, n_bins: int = 20):
        # Inspired by scanpy.pp.highly_variable_genes

        names = X_df.columns
        mean = X_df.values.mean(axis=0)
        var = X_df.values.var(axis=0, ddof=1)

        mean[mean == 0] = 1e-12
        dispersion = var / mean
        mean = np.log1p(mean)
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)

        df = pd.DataFrame(
            {"means": mean, "vars": var, "dispersions": dispersion},
            index=names,
        )
        df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
        df_stats = df.groupby("mean_bin", observed=True)["dispersions"].agg(
            avg="mean", dev="std"
        )

        one_gene_per_bin = df_stats["dev"].isnull()
        gen_indices = np.flatnonzero(one_gene_per_bin.loc[df["mean_bin"]])

        if len(gen_indices) > 0:
            df_stats.loc[one_gene_per_bin, "dev"] = df_stats.loc[
                one_gene_per_bin, "avg"
            ]
            df_stats.loc[one_gene_per_bin, "avg"] = 0

        df_stats = df_stats.loc[df["mean_bin"]].set_index(df.index)

        norm_dispersion = (
            df["dispersions"].values - df_stats["avg"].values
        ) / df_stats["dev"].values

        dedf = pd.DataFrame(
            {
                DEA.feature.value: names,
                DEA.score.value: norm_dispersion,
                "means": df["means"],
            },
        )
        dedf[DEA.adj_p_value.value] = np.nan

        return dedf

    @classmethod
    def run_with_adata(cls, adata: ad.AnnData, layer: str | None = None):

        dedf = cls.get_hvg_dea(adata.to_df(layer=layer))
        return dedf

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        *args,
        **kwargs,
    ):
        raise NotImplemented(
            "The RandomFeatureDEA  method has not been implemented for telegraph workflow use yet"
        )

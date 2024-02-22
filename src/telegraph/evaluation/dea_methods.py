import os.path as osp
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from . import policies as pol
from . import utils as ut
from ._methods import MethodClass

# import telegraph.evaluation.policies as pol
# from telegraph.evaluation._methods import MethodClass


class DEA(Enum):
    adj_p_value = "pvals_adj"
    logfold = "log2fc"
    p_value = "pvals"
    feature = "feature"
    score = "score"
    coeff = "score"


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
    @ut.check_in_out
    def run(
        cls,
        input_dict: Dict[str, Any],
        method: str = "wilcoxon",
        sort_by: str = "pvals_adj",
        pval_cutoff: float | None = None,
        mode: Literal["pos", "neg", "both"] = "both",
        groups: List[str] | str = "all",
        method_kwargs: Dict[str, Any] = {},
        normalize: bool = False,
        subset_features: Dict[str, List[str]] | Dict[str, str] | None = None,
        min_group_obs: int = 2,
        split_by_base: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:

        # data frame of predicted "to" data : [n_to] x [n_from_features]
        X_to_pred = input_dict.get("X_to_pred")
        # anndata of "from" : [n_from] x [n_from_features]
        X_from = input_dict.get("X_from")
        # design matrix for "to" : [n_to] x [n_to_covariates]
        D_to = input_dict.get("D_to")
        # design matrix for "from" : [n_from] x [n_from_covariates]
        D_from = input_dict.get("D_from")

        X_from = input_dict.get("X_from")

        objects = dict()

        if (D_to is not None) and (X_to_pred is not None):
            pol.check_values(X_to_pred, "X_to_pred")
            pol.check_type(X_to_pred, "X_to_pred")

            n_to = X_to_pred.shape[0]

            pol.check_values(D_to, "D_to")
            pol.check_type(D_to, "D_to")
            pol.check_dimensions(D_to, "D_to", (n_to, None))

            # data frame of predicted "to" data : [n_to] x [n_from_features]
            if isinstance(X_to_pred, pd.DataFrame):
                to_pred_names = input_dict["to_pred_names"]
                to_pred_var = input_dict["to_pred_var"]
                adata_to = ad.AnnData(
                    X_to_pred,
                    obs=pd.DataFrame(
                        [],
                        index=to_pred_names,
                    ),
                    var=pd.DataFrame([], index=to_pred_var),
                )
            elif isinstance(X_to_pred, ad.AnnData):
                adata_to = X_to_pred
            else:
                raise NotImplementedError

            # update objects
            objects["to"] = dict(D=D_to, adata=adata_to)

        if (D_from is not None) and (X_from is not None):
            pol.check_values(X_from, "X_from")
            pol.check_type(X_from, "X_from")

            n_from = X_from.shape[0]

            pol.check_values(D_from, "D_from")
            pol.check_type(D_from, "D_from")
            pol.check_dimensions(D_from, "D_from", (n_from, None))

            objects["from"] = dict(D=D_from, adata=X_from)

        out = dict()

        for obj_name in objects.keys():
            D = objects[obj_name]["D"]
            adata = objects[obj_name]["adata"]

            base_groups_og = input_dict.get("base_groups", None)

            if split_by_base:
                if base_groups_og is not None:
                    from itertools import chain

                    flat_base_groups = list(chain(*base_groups_og))
                    base_cols = np.array([x in flat_base_groups for x in D.columns])
                    add_cols = D.columns[~base_cols].tolist()
                    base_groups = base_groups_og
                else:
                    raise NotImplementedError
            else:
                base_groups = [D.columns.tolist()]
                add_cols = []
                groups = "all"

            # subset design matrices if specified
            if subset_features is not None:
                if not isinstance(subset_features, dict):
                    NotImplementedError
                if obj_name in subset_features:
                    base_groups = [
                        x
                        for x in base_groups
                        if any(y in subset_features[obj_name] for y in x)
                    ]
                    add_cols = [x for x in add_cols if x in subset_features[obj_name]]

            # normalize data if specified
            if normalize:
                X_old = adata.X.copy()
                sc.pp.normalize_total(adata, 1e4)
                sc.pp.log1p(adata)

            for idx in base_groups:
                sel_cols = list(idx) + list(add_cols)
                # get labels from design matrix
                labels = ut.design_matrix_to_labels(D.loc[:, sel_cols])
                # set unlabeled observations to "background"
                labels[labels == ""] = "background"
                uni_labels = np.unique(labels)

                # add labels as a column in adata ("from")
                # this is to be able to use the scanpy function
                adata.obs["_label"] = labels

                # make group specification align with expected scanpy input
                if groups == "all":
                    # if no groups specified set to "all"
                    main_group = "all"
                    ref_group = "rest"

                    # execute DE test
                    sc.tl.rank_genes_groups(
                        adata,
                        groupby="_label",
                        groups=main_group,  # groups _must_ be a list or str, not np.ndarray
                        reference=ref_group,
                        method=method,
                        **method_kwargs,
                    )

                    for lab in uni_labels:
                        dedf = sc.get.rank_genes_groups_df(
                            adata,
                            group=lab,
                            pval_cutoff=pval_cutoff,
                        )
                        name = f"{obj_name}_{lab}_vs_rest"
                        out[name] = dedf

                else:
                    # iterate over groups
                    # groups are [(grp_1_a,grp_1_b),(grp_2_a,grp_2_b)]
                    if groups is None:
                        if base_groups_og is not None:
                            _groups = ut.update_default_groups(base_groups, uni_labels)
                        else:
                            NotImplementedError

                    for group in _groups:
                        grp_1, grp_2 = group

                        # execute DE test
                        sc.tl.rank_genes_groups(
                            adata,
                            groupby="_label",
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

                        name = f"{obj_name}_{grp_1}_vs_{grp_2}"
                        out[name] = dedf

            # undo the normalization
            if normalize:
                adata.X = X_old

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
        use_pred: bool | Dict[str, bool] = True,
        subset_features: List[str] | str | None = None,
        fit_intercept: bool = True,
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
                if use_pred and f"X_{tgt}_pred" in input_dict:
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

            # subset to specified covariates, if None then use all
            if use_covariates is not None:
                D_inp = D_inp[ut.listify(use_covariates)]
            # drop covariates if specified
            if drop_covariates is not None:
                keep_cols = [
                    x for x in D.columns if x not in ut.listify(drop_covariates)
                ]
                D_inp = D_inp[keep_cols]

            # check what features to test
            _features = X_inp.columns if subset_features is None else subset_features

            # default glm parameters
            # from: https://glum.readthedocs.io/en/latest/glm.html#glum.GeneralizedLinearRegressor (2024-01-25)
            glm_default_params = dict(
                alpha=None,
                l1_ratio=0,
                P1="identity",
                P2="identity",
                link="auto",
                solver="auto",
                max_iter=100,
                gradient_tol=None,
                step_size_tol=None,
                hessian_approx=0.0,
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
                y = X_inp[feature].values.flatten()
                # create glm object
                glm = gl.GeneralizedLinearRegressor(**glm_params)
                # fit glm object
                glm.fit(X=D_inp, y=y)
                # get coefficient table
                coef_table = glm.coef_table(X=D_inp, y=y)
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

        return dict(DEA=out)

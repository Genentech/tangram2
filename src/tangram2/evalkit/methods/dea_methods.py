import os.path as osp
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, List, Literal, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from tangram2.evalkit.methods._methods import MethodClass

from . import _dea_utils as dut
from . import policies as pol
from . import utils as ut
from ._dea_utils import DEA


class DEAMethodClass(MethodClass):
    """DEA Base Class"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    @abstractmethod
    def run(
        cls, input_dict: Dict[str, Any], **kwargs
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Executes a process with the given input.

        Args:
            cls: The class instance invoking the method.
            input_dict (Dict[str, Any]): Input data for the process.
            **kwargs: Additional keyword arguments for the process.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: A dictionary containing processed
                dataframes organized by keys.

        Raises:
            NotImplementedError: This function is not implemented.
        """
        pass


class ScanpyDEA(DEAMethodClass):
    """Scanpy DEA class"""

    ins = ["D_from", "D_to", "X_from", "X_to_pred"]
    outs = ["DEA"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
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
        """Performs differential expression analysis.

        Args:
            input_dict: Dictionary containing input data. Keys may include "X_to_pred"
                or "X_to" (predicted "to" data), "X_from" ("from" data), "D_to"
                ("to" design matrix), and "D_from" ("from" design matrix).
            groups: Group or list of groups to compare. Defaults to "all".
            method: Differential expression test method. Defaults to "wilcoxon".
            sort_by: Column to sort results by. Defaults to "pvals_adj".
            pval_cutoff: Adjusted p-value cutoff for filtering results.
            mode: Whether to return "pos" (positive scores), "neg" (negative scores),
                or "both". Defaults to "both".
            method_kwargs: Additional keyword arguments for the DE test method.
            normalize: Whether to normalize data. Defaults to False.
            subset_features: Features to subset. Defaults to None.
            min_group_obs: Minimum number of observations per group. Defaults to 2.
            target: Target data to analyze ("to", "from", or "both").
                Defaults to "both".
            **kwargs: Additional keyword arguments.

        Returns:
            Dictionary containing differential expression results.

        Raises:
            NotImplementedError: If `mode` is not "pos", "neg", or "both".
        """
        if target == "both":
            _target = ["to", "from"]
        else:
            _target = ut.listify(target)
        X_to_pred = input_dict.get("X_to_pred")
        if X_to_pred is None:
            X_to_pred = input_dict.get("X_to")
        X_from = input_dict.get("X_from")
        D_to = input_dict.get("D_to")
        D_from = input_dict.get("D_from")
        X_from = input_dict.get("X_from")
        objects = dict()
        if "to" in _target and D_to is not None and (X_to_pred is not None):
            pol.check_values(X_to_pred, "X_to_pred")
            pol.check_type(X_to_pred, "X_to_pred")
            n_to = X_to_pred.shape[0]
            pol.check_values(D_to, "D_to")
            pol.check_type(D_to, "D_to")
            pol.check_dimensions(D_to, "D_to", (n_to, None))
            objects["to"] = dict(D=D_to, X=X_to_pred)
        if "from" in _target and D_from is not None and (X_from is not None):
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
                    _groups = [groups]
                case [str(), str()]:
                    _groups = [groups]
                case _:
                    _groups = groups
            for group_pair in _groups:
                D_new, grp_1, grp_2 = dut.scanpy_dea_labels_from_D(D, group_pair)
                if grp_1 is None and grp_2 is None:
                    continue
                adata = dut.anndata_from_X_and_D(X, D_new)
                labels = adata.obs["label"].values
                n_grp_1 = np.sum(labels == grp_1)
                if grp_2 == "rest":
                    n_grp_2 = min_group_obs + 1
                else:
                    n_grp_2 = np.sum(labels == grp_2)
                if n_grp_1 >= min_group_obs and n_grp_2 >= min_group_obs:
                    sc.tl.rank_genes_groups(
                        adata,
                        groupby="label",
                        groups=[grp_1],
                        reference=grp_2,
                        method=method,
                        **method_kwargs,
                    )
                    dedf = sc.get.rank_genes_groups_df(
                        adata, group=grp_1, pval_cutoff=pval_cutoff
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
                    print(
                        f"[WARNING] : Both groups must have more than {min_group_obs}  members; returning empty dataframe.\n Current count {grp_1} : {n_grp_1} and {grp_2} : {n_grp_1}"
                    )
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
    """Generalized Linear Model Differential Expression Analysis class."""

    def __init__(self, *args, **kwargs):
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
        """Performs differential expression analysis.

        Args:
            input_dict (Dict[str, Any]): A dictionary containing the input data.
                Must contain keys "D_to" and/or "D_from" representing design matrices,
                and "X_to" and/or "X_from" (and optionally "X_to_pred", "X_from_pred")
                representing feature matrices.  These can be pandas DataFrames or AnnData objects.
            family (str, optional): The distribution family to use in the GLM.
                Defaults to "negative.binomial".
            target (Literal["to", "from", "both"], optional): Which interaction type to analyze.
                Defaults to "both".
            use_covariates (List[str] | str | None, optional):  A list of covariates to use.
                If None, all covariates are used. Defaults to None.
            drop_covariates (List[str] | str | None, optional): A list of covariates to drop.
                Defaults to None.
            merge_covariates (List[List[str]] | None, optional): Covariates to merge. Defaults to None.
            use_pred (bool | Dict[str, bool], optional): Whether to use predicted values.
                Defaults to True.
            subset_features (List[str] | str | None, optional): A list of features to analyze.
                If None, all features are analyzed. Defaults to None.
            use_obs_with_covariates (bool, optional): Whether to use only observations
                with covariates. Defaults to False.
            fit_intercept (bool, optional): Whether to fit an intercept in the GLM.
                Defaults to False.
            mht_method (str, optional): The multiple hypothesis testing correction method.
                Defaults to "fdr_bh".
            **kwargs: Additional keyword arguments to pass to the GLM.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: A dictionary where the key is "DEA" and the
                value is another dictionary containing DataFrames of differential expression
                results, keyed by target and covariate.

        Raises:
            AssertionError: If required keys are missing in `input_dict`.
            ValueError: If input data contains invalid values or types.

        """
        import glum as gl
        from statsmodels.stats.multitest import multipletests as mht

        if target == "both":
            target_l = ["to", "from"]
        else:
            target_l = ut.listify(target)
        out = dict()
        for tgt in target_l:
            D = input_dict.get("D_{}".format(tgt))
            assert D is not None, f"D_{tgt} is not defined"
            D_inp = D.copy()
            pol.check_values(D_inp, f"D_{tgt}")
            pol.check_type(D_inp, f"D_{tgt}")
            X_name = "X_{}".format(tgt)
            if isinstance(use_pred, dict):
                if use_pred.get(tgt, False):
                    X_name += "_pred"
            elif isinstance(use_pred, bool):
                if use_pred and input_dict.get(f"X_{tgt}_pred", None) is not None:
                    X_name += "_pred"
            X = input_dict.get(X_name)
            assert X is not None, f"{X_name} is not defined"
            X_inp = X.copy()
            pol.check_values(X_inp, X_name)
            pol.check_type(X_inp, X_name)
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
            if drop_covariates is not None:
                keep_cols = [
                    x for x in D.columns if x not in ut.listify(drop_covariates)
                ]
                D_inp = D_inp[keep_cols]
            if use_obs_with_covariates:
                has_covs = np.any(D_inp.values != 0, axis=1)
                X_inp = X_inp.iloc[has_covs, :]
                D_inp = D_inp.iloc[has_covs, :]
            _features = X_inp.columns if subset_features is None else subset_features
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
            glm_params = ut.merge_default_dict_with_kwargs(glm_default_params, kwargs)
            glm_params["fit_intercept"] = fit_intercept
            glm_params["family"] = family
            res = {
                "covariate": [],
                DEA.coeff.value: [],
                DEA.p_value.value: [],
                DEA.feature.value: [],
            }
            for feature in _features:
                y = X_inp[feature].values.flatten().astype(np.float64)
                if np.sum(y) == 0:
                    continue
                glm = gl.GeneralizedLinearRegressor(**glm_params)
                glm.fit(X=D_inp, y=y)
                coef_table = glm.coef_table(X=D_inp.astype(np.float64), y=y)
                res["covariate"] += coef_table.index.tolist()
                res[DEA.p_value.value] += coef_table["p_value"].values.tolist()
                res[DEA.coeff.value] += coef_table["coef"].values.tolist()
                res[DEA.feature.value] += [feature] * coef_table.shape[0]
            _, p_val_adj, _, _ = mht(res[DEA.p_value.value], method=mht_method)
            res[DEA.adj_p_value.value] = p_val_adj
            res = pd.DataFrame(res)
            tgt_out = {
                f"{tgt}_{cov}": res.iloc[res["covariate"].values == cov].copy()
                for cov in D_inp.columns
            }
            out.update(tgt_out)
        if "DEA" in input_dict:
            input_dict["DEA"].update(out)
            out = input_dict["DEA"]
        return dict(DEA=out)


class LRDEA(DEAMethodClass):
    """Logistic Regression Differential Expression Analysis class."""

    ins = [("D_from", "D_to"), ("X_from", "X_to", "X_to_pred")]
    outs = ["DEA"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        covariates: Dict[str, str] | str | List[str],
        sort_by: str = "pvals_adj",
        pval_cutoff: float | None = None,
        random_state: int = 0,
        C: float = 1,
        max_iter: int = 100,
        target: Literal["both", "to", "from"] = "from",
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Performs differential expression analysis.

        Args:
            input_dict (Dict[str, Any]): A dictionary containing input data. Requires
                keys "D_{to/from}" for the dependent variable and "X_{to/from}" or
                "X_{to/from}_pred" for the independent variable.
            covariates (Dict[str, str] | str | List[str]): Covariates to use in the
                analysis. If target is "both", must be a dictionary. If a string or
                list, assumed to apply to the specified target.
            sort_by (str): Column to sort results by. Defaults to "pvals_adj".
            pval_cutoff (float | None): P-value cutoff for filtering results. Defaults
                to None.
            random_state (int): Random seed. Defaults to 0.
            C (float): Regularization strength. Defaults to 1.
            max_iter (int): Maximum number of iterations for the solver. Defaults to 100.
            target (Literal["both", "to", "from"]): Target type. Defaults to "from".
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: A dictionary containing the
                differential expression analysis results.

        Raises:
            ValueError: If input data is missing or covariates are incorrectly specified.
        """
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
                    {DEA.score.value: coef, DEA.feature.value: X_tgt.columns.tolist()}
                )
                out[f"{tgt}_{cov}"] = dea
            return dict(DEA=out)


class RandomFeatureDEA(DEAMethodClass):
    """Random Feature Differential Expression Analysis class."""

    ins = [("X_from", "X_to", "X_to_pred")]
    outs = ["DEA"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def get_random_dea(cls, names, lfc_min_max: Tuple[float, float] = (-2, 2)):
        """Generates a DataFrame with random differential expression analysis (DEA) data.

        Args:
            names: A sequence of feature names.
            lfc_min_max: A tuple specifying the minimum and maximum log fold change.
                Defaults to (-2, 2).

        Returns:
            A pandas DataFrame containing the simulated DEA data with columns for
            feature names, log fold change, and adjusted p-values.

        Raises:
            TypeError: If input types are not as expected.
            ValueError: if lfc_min_max is not a two element tuple or if elements cannot be converted to float
        """
        n_obs = len(names)
        pvals = np.random.uniform(0, 1, size=n_obs)
        lfc_min, lfc_max = lfc_min_max
        lfc = np.random.uniform(lfc_min, lfc_max, size=n_obs)
        dedf = pd.DataFrame(
            {
                DEA.feature.value: names,
                DEA.logfold.value: lfc,
                DEA.adj_p_value.value: pvals,
            }
        )
        return dedf

    @classmethod
    def run_with_adata(cls, adata: ad.AnnData):
        """Computes a differential expression analysis result.

        Args:
            cls: The class instance used for the analysis.
            adata: Annotated data matrix containing gene expression data.

        Returns:
            pandas.DataFrame: A DataFrame containing the differential expression
                analysis results.

        Raises:
            TypeError: If input data is not in the expected format.

        """
        names = adata.var_names.tolist()
        dedf = cls.get_random_dea(names)
        return dedf

    @classmethod
    def run(cls, input_dict: Dict[str, Any], *args, **kwargs):
        """Executes the RandomFeatureDEA method.

        Args:
            cls: The class instance.
            input_dict (Dict[str, Any]): The input dictionary.
            *args: Variable length positional arguments.
            **kwargs: Variable length keyword arguments.

        Returns:
            This method is not yet implemented and always raises an exception.

        Raises:
            NotImplementedError:  Always raised as this method is not yet implemented.
        """
        raise NotImplemented(
            "The RandomFeatureDEA  method has not been implemented for telegraph workflow use yet"
        )


class HVGFeatureDEA(DEAMethodClass):
    """Highly Variable Gene Differential Expression Analysis class."""

    ins = [("X_from", "X_to", "X_to_pred")]
    outs = ["DEA"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def get_hvg_dea(cls, X_df, n_bins: int = 20):
        """Identifies highly variable genes.

        Args:
            X_df: A pandas DataFrame where rows represent observations and columns
                represent genes.  Values should be non-negative.
            n_bins (int): Number of bins for mean expression. Default is 20.

        Returns:
            pd.DataFrame: A DataFrame with highly variable gene information, including:
                - 'feature': Gene names.
                - 'score': Normalized dispersion score.
                - 'means': Log-transformed mean expression.
                - 'adj_p_value': Placeholder NaN values for adjusted p-values.

        Raises:
            ValueError: If `X_df` contains negative values.
        """
        names = X_df.columns
        mean = X_df.values.mean(axis=0)
        var = X_df.values.var(axis=0, ddof=1)
        mean[mean == 0] = 1e-12
        dispersion = var / mean
        mean = np.log1p(mean)
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        df = pd.DataFrame(
            {"means": mean, "vars": var, "dispersions": dispersion}, index=names
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
            }
        )
        dedf[DEA.adj_p_value.value] = np.nan
        return dedf

    @classmethod
    def run_with_adata(
        cls,
        adata: ad.AnnData,
        subset_col: str,
        subset_labels: str | List[str] | None = None,
        layer: str | None = None,
    ):
        """Calculates differentially expressed genes for a subset of cells in an AnnData object.

        Args:
            adata: AnnData object containing gene expression data.
            subset_col: Name of the column in `adata.obs` used for subsetting.
            subset_labels: Label(s) in `subset_col` to select. If None, all cells are used.
            layer: Name of the layer in `adata` to use. If None, uses the default layer.

        Returns:
            pandas.DataFrame: DataFrame containing differentially expressed genes and their scores.

        Raises:
            KeyError: If `subset_col` or `layer` are not found in `adata`.
            ValueError: If `subset_labels` are not found in `adata.obs[subset_col]`.
        """
        is_label = ut.get_adata_subset_idx(adata, subset_col, subset_labels)
        dedf = cls.get_hvg_dea(adata[is_label].to_df(layer=layer))
        dedf = dedf.iloc[~dedf["score"].isna().values]
        return dedf

    @classmethod
    def run(cls, input_dict: Dict[str, Any], *args, **kwargs):
        """Executes the DEA method on input_dict.

        Args:
            cls: The class instance.
            input_dict (Dict[str, Any]): The input dictionary.
            *args: Variable length positional arguments.
            **kwargs: Variable length keyword arguments.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplemented(
            "The HVGFeatureDEA  method has not been implemented for telegraph workflow use yet"
        )


class CorrelationDEA(DEAMethodClass):
    """Correlation Differential Expression Analysis class."""

    ins = ["D_from", "D_to", "X_from", "X_to_pred"]
    outs = ["DEA"]

    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def compute_correlations_pval(cls, X, Y, method="pearson"):
        """Computes correlations and corrected p-values between a dataset X and multiple target variables.

        Args:
            X (numpy.ndarray): 1D array of input data.
            Y (numpy.ndarray): 2D array where each column represents a target variable.
            method (str): Correlation method ('pearson' or 'spearmanr'). Defaults to 'pearson'.

        Returns:
            tuple: A tuple containing two numpy arrays:
                - correlations (numpy.ndarray): Array of correlation coefficients.
                - corrected_p_values (numpy.ndarray): Array of Bonferroni corrected p-values.

        Raises:
            ValueError: If an invalid correlation `method` is provided.
        """
        if method == "pearson":
            corr_fun = pearsonr
        elif method == "spearmanr":
            corr_fun = spearmanr
        n_features = Y.shape[1]
        correlations = np.zeros(n_features)
        p_values = np.zeros(n_features)
        for i in range(n_features):
            correlations[i], p_values[i] = corr_fun(X, Y[:, i])
        corrected_p_values = multipletests(p_values, method="bonferroni")[1]
        return (correlations, corrected_p_values)

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        covariates: List[str] | str,
        target: List[str] | str = "both",
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Computes correlations and adjusted p-values.

        Args:
            cls: Class instance used for computation.
            input_dict (Dict[str, Any]): Dictionary containing input data. Keys "D_to", "X_to",
                "D_from", and "X_from" are expected to contain dataframes or AnnData objects.
            covariates (List[str] | str):  Covariate names. If string, it's treated as a single covariate.
                Can be a list of covariates if the same covariates should be used for both "to" and "from"
                or a dictionary with keys "from" and "to" containing separate lists for each.
            target (List[str] | str, optional):  Target keys, either "to", "from", or "both" (default).
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Dict[str, pd.DataFrame]]: A dictionary containing correlation results.
                The outer key is "DEA". The inner dictionary keys are formatted as "{obj}_{cov}" where
                "obj" is "to" or "from" and "cov" is a covariate name.  Values are dataframes with
                "corr", "names", and "pvals_adj" columns.

        Raises:
            ValueError: If `input_dict` is missing required keys or contains invalid data.
        """
        if target == "both":
            _target = ["to", "from"]
        else:
            _target = ut.listify(target)
        if isinstance(covariates, str):
            covariates = [covariates]
        if isinstance(covariates, (tuple, list)):
            covariates = {"from": covariates, "to": covariates}
        out = dict()
        for obj in _target:
            D = input_dict.get(f"D_{obj}")
            X = input_dict.get(f"X_{obj}")
            if isinstance(X, ad.AnnData):
                X = X.to_df()
            if X is None or D is None:
                raise ValueError(f"D_{obj} and X_{obj} must be provided")
            for cov in covariates[obj]:
                print(cov)
                y = D[cov].values
                skip_obs = np.isnan(y)
                x = X.values[~skip_obs, :].copy()
                y = y[~skip_obs]
                x_var_sum = x.sum(axis=0).flatten()
                keep_var = x_var_sum > 0
                x = x[:, keep_var]
                corrs, pvals = cls.compute_correlations_pval(y, x)
                corrs = pd.DataFrame(
                    corrs.T, index=X.columns[keep_var], columns=["corr"]
                )
                corrs["names"] = corrs.index
                corrs["pvals_adj"] = pvals
                out[f"{obj}_{cov}"] = corrs
        return dict(DEA=out)

import os.path as osp
from abc import abstractmethod
from typing import Any, Dict, List, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from eval._methods import MethodClass

from . import utils as ut


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

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        **kwargs,
    ) -> None:
        dea = res_dict["DEA"]
        for key, df in dea.items():
            out_pth = osp.join(out_dir, f"{key}_vs_rest_dea.csv")
            df.to_csv(out_pth)


class ScanpyDEA(DEAMethodClass):
    # Scanpy DEA Method class
    # allows you to execute scanpy.tl.rank_genes
    # using a design matrix and input data
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
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
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:

        # data frame of predicted "to" data : [n_to] x [n_from_features]
        X_to_pred = input_dict["X_to_pred"]
        # anndata of "from" : [n_from] x [n_from_features]
        adata = input_dict["X_from"]
        # design matrix for "to" : [n_to] x [n_to_covariates]
        D_to = input_dict["D_to"]
        # design matrix for "from" : [n_from] x [n_from_covariates]
        D_from = input_dict["D_from"]

        # subset design matrices if specified
        if subset_features is not None:
            if not isinstance(subset_features, dict):
                NotImplementedError

            if "to" in subset_features:
                D_to = D_to.loc[:, subset_features["to"]]
            if "from" in subset_features:
                D_from = D_from.loc[:, subset_features["from"]]

        # get labels from design matrix
        labels = ut.design_matrix_to_labels(D_from)

        # set unlabeled observations to "background"
        labels[labels == ""] = "background"

        # add labels as a column in adata ("from")
        # this is to be able to use the scanpy function
        adata.obs["label"] = labels

        # make group specification align with expected scanpy input
        if groups is not None:
            # if groups are "all" remove listified
            if groups == "all":
                uni_labels = groups
            else:
                # subset identified labels (based on design matrix)
                # w.r.t. the specified groups
                groups = ut.listify(groups)
                uni_labels = np.unique(adata.obs["label"].values)
                uni_groups = np.unique(groups)
                uni_labels = [lab for lab in uni_labels if lab in uni_groups]
                # check that the subsetted labels are at least two
                if len(uni_labels) < 2:
                    return dict(pred=pd.DataFrame([]), DEA=pd.DataFrame([]))

        # if no groups specified set to "all"
        else:
            uni_labels = "all"

        # normalize data if specified
        if normalize:
            X_old = adata.X.copy()
            sc.pp.normalize_total(adata, 1e4)
            sc.pp.log1p(adata)

        # execute DE test
        sc.tl.rank_genes_groups(
            adata,
            groupby="label",
            groups=[uni_labels[0]],  # groups _must_ be a list or str, not np.ndarray
            reference=uni_labels[1],
            method=method,
            **method_kwargs,
        )

        # get data frame of test
        dedf = sc.get.rank_genes_groups_df(
            adata,
            group=uni_labels[0],
            pval_cutoff=pval_cutoff,
        )

        # instantiate out dict
        out = dict()

        # modify output based on mode

        lab = uni_labels[0]

        if mode == "both":
            out[lab] = dedf
        elif mode == "pos":
            out[lab] = dedf[dedf["scores"].values > 0]
        elif mode == "neg":
            out[lab] = dedf[dedf["scores"].values < 0]
        else:
            raise NotImplementedError

        # undo the normalization
        if normalize:
            adata.X = X_old

        return dict(pred=out, DEA=out)

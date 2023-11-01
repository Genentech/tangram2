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
        pval_cutoff: float = 0.01,
        mode: Literal["pos", "neg", "both"] = "both",
        groups: List[str] | str = "all",
        method_kwargs: Dict[str, Any] = {},
        normalize: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:

        X_to_pred = input_dict["X_to_pred"]
        adata = input_dict["X_from"]
        D_to = input_dict["D_to"]
        D_from = input_dict["D_from"]

        labels = ut.design_matrix_to_labels(D_from)

        labels[labels == ""] = "background"

        adata.obs["label"] = labels
        groups = ut.listify(groups)

        if groups is not None:
            if groups[0] == "all":
                uni_labels = groups
            else:
                uni_labels = np.unique(adata.obs["label"].values)
                uni_groups = np.unique(groups)
                uni_labels = [lab for lab in uni_labels if lab in uni_groups]
                if len(uni_labels) < 2:
                    return dict(pred=pd.DataFrame([]), DEA=pd.DataFrame([]))

        else:
            uni_labels = "all"

        if normalize:
            X_old = adata.X.copy()
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            use_raw = False
        else:
            use_raw = None

        sc.tl.rank_genes_groups(
            adata,
            groupby="label",
            groups=uni_labels,
            method=method,
            use_raw=use_raw,
            **method_kwargs,
        )

        out = dict()

        for lab in uni_labels:
            dedf = sc.get.rank_genes_groups_df(
                adata,
                group=lab,
                pval_cutoff=pval_cutoff,
            )

            if mode == "both":
                out[lab] = dedf
            elif mode == "pos":
                out[lab] = dedf[dedf["scores"].values > 0]
            elif mode == "neg":
                out[lab] = dedf[dedf["scores"].values < 0]
            else:
                raise NotImplementedError

        if normalize:
            adata.X = X_old

        return dict(pred=out, DEA=out)

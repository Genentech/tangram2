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
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        X_to_pred = input_dict["X_to_pred"]
        adata = input_dict["X_from"]
        D_to = input_dict["D_to"]
        D_from = input_dict["D_from"]

        labels = np.array(
            list(
                map(
                    lambda x: "_".join(D_from.columns.values[x].tolist()),
                    D_from.values == 1,
                )
            )
        )

        labels[labels == ""] = "background"

        adata.obs["label"] = labels

        if groups is None:
            uni_labels = np.unique(adata.obs["label"].values)
        else:
            uni_labels = np.unique(groups)

        sc.tl.rank_genes_groups(
            adata,
            groupby="label",
            groups=groups,
            method=method,
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

        return dict(pred=out, DEA=out)

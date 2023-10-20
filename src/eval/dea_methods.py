import os.path as osp
from abc import abstractmethod
from typing import Any, Dict, Literal

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
        **kwargs,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:

        X_to_pred = input_dict["X_to_pred"]
        D_to = input_dict["D_to"]
        D_from = input_dict["D_from"]

        labels = np.apply_along_axis(
            lambda x: "_".join(D_to.columns.values[x].tolist()),
            arr=D_to.values == 1,
            axis=1,
        )

        labels = np.array(labels)
        labels[labels == ""] = "background"

        adata = ad.AnnData(
            X_to_pred.values,
            obs=pd.DataFrame([], index=X_to_pred.index),
            var=pd.DataFrame([], index=X_to_pred.columns),
        )

        adata.obs["label"] = labels

        uni_labels = np.unique(adata.obs["label"].values)

        sc.tl.rank_genes_groups(adata, groupby="label", method=method)

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

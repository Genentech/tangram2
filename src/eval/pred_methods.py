import os.path as osp
from abc import abstractmethod
from typing import Any, Dict

import anndata as ad
import numpy as np
import pandas as pd
import tangram as tg1
import tangram2 as tg2
from scipy.sparse import spmatrix

import eval.utils as ut
from eval._methods import MethodClass


class PredMethodClass(MethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def run(
        cls,
        X_to: Any,
        X_from: Any,
        S_from: np.ndarray | None,
        T: np.ndarray | spmatrix | None = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        pass

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        compress: bool = False,
        **kwargs,
    ) -> None:
        for obj_ind in ["to_pred", "from_pred"]:
            obj_name = f"X_{obj_ind}"

            obj = res_dict.get(obj_name, None)

            if obj is not None:
                obj_df = pd.DataFrame(
                    obj,
                    index=res_dict[f"{obj_ind}_names"],
                    columns=res_dict[f"{obj_ind}_var"],
                )

                out_pth = osp.join(out_dir, f"{obj_name}.csv")

                if compress:
                    out_pth = out_pth + ".gz"
                    ut.to_csv_gzip(obj_df, out_pth)
                else:
                    obj_df.to_csv(out_pth)


class TangramPred(PredMethodClass):
    tg = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        *args,
        spatial_key_to: str = "spatial",
        spatial_key_from: str = "spatial",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        X_to = input_dict["X_to"]
        X_from = input_dict["X_from"]
        T = input_dict["T"]

        if isinstance(T, spmatrix):
            T_soft = T.todense()
        else:
            T_soft = T.copy()

        ad_map = ad.AnnData(
            T_soft.T,
            var=X_to.obs,
            obs=X_from.obs,
        )

        if "training_genes" in X_to.uns:
            training_genes = X_to.uns["training_genes"]
        elif "training_genes" in X_from.uns:
            training_genes = X_from.uns["training_genes"]
        elif "training_genes" in kwargs:
            training_genes = kwargs["training_genes"]
        elif "markers" in kwargs:
            training_genes = kwargs["markers"]
        else:
            training_genes = X_to.var.index.intersection(X_from.var.index).tolist()

        ad_map.uns["train_genes_df"] = pd.DataFrame([], index=training_genes)

        ad_ge = cls.tg.project_genes(adata_map=ad_map, adata_sc=X_from)

        X_to_pred = ad_ge.to_df()

        to_pred_names = X_to_pred.index.values.tolist()
        to_pred_var = X_to_pred.columns.values.tolist()

        return dict(
            pred=X_to_pred,
            X_to_pred=X_to_pred,
            X_from_pred=None,
            to_pred_names=to_pred_names,
            to_pred_var=to_pred_var,
        )


class TangramV1Pred(TangramPred):
    tg = tg1

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class TangramV2Pred(TangramPred):
    tg = tg2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass

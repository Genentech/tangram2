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
    # Prediction Method Base class
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
        # common save method for pred methods
        # tries to save X_to_pred and X_from_pred
        # if they exist in the res_dict

        # loop over object prefixes
        for obj_ind in ["to_pred", "from_pred"]:

            # get object name
            obj_name = f"X_{obj_ind}"

            # grab object from results dict
            # if not available return None
            obj = res_dict.get(obj_name, None)

            # if object is present
            if obj is not None:

                # create a data_frame using the object
                obj_df = pd.DataFrame(
                    obj,
                    index=res_dict[f"{obj_ind}_names"],
                    columns=res_dict[f"{obj_ind}_var"],
                )

                # define output path
                out_pth = osp.join(out_dir, f"{obj_name}.csv")

                # write to .gz if compressed output is specified
                if compress:
                    out_pth = out_pth + ".gz"
                    ut.to_csv_gzip(obj_df, out_pth)
                # write to .csv if no compression specified
                else:
                    obj_df.to_csv(out_pth)


class TangramPred(PredMethodClass):
    # specify which tangram module to use
    tg = None
    # specify the tangram version
    version = None

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
        # Tangram Prediction Baseclass

        # get "to" anndata : [n_to] x [n_to_features]
        X_to = input_dict["X_to"]
        # get "from" anndata : [n_from] x [n_from_features]
        X_from = input_dict["X_from"]
        # get "scaled from" anndata, this is necessary
        # due to the adjustment of the "from" data
        # in tg2-hejin_workflow. Is None for all other tg outputs.
        # If not None [n_from] x [n_from_features]
        X_from_scaled = input_dict.get("X_from_scaled")
        # get map : [n_to] x [n_from]
        T = input_dict["T"]

        # convert map to dense matrix
        if isinstance(T, spmatrix):
            T_soft = T.todense()
        else:
            T_soft = T.copy()

        # create anndata object of map
        # tangram functions expects map to be [n_from] x [n_to]
        # hence the transpose
        ad_map = ad.AnnData(
            T_soft.T,
            var=X_to.obs,
            obs=X_from.obs,
        )

        # get training genes
        if "training_genes" in X_to.uns:
            training_genes = X_to.uns["training_genes"]
        elif "training_genes" in X_from.uns:
            training_genes = X_from.uns["training_genes"]
        elif "training_genes" in kwargs:
            training_genes = ut.list_or_path_get(kwargs["training_genes"])
        elif "markers" in kwargs:
            training_genes = ut.list_or_path_get(kwargs["markers"])
        else:
            training_genes = X_to.var.index.intersection(X_from.var.index).tolist()

        # update map anndata object to have training genes
        # necessary
        ad_map.uns["train_genes_df"] = pd.DataFrame([], index=training_genes)

        # chose which "from" features to project
        # depending on version and mode in tangram
        if (cls.version == "2") and (X_from_scaled is not None):
            ad_sc = X_from_scaled
        elif (cls.version == "1") or (cls.version == "2"):
            ad_sc = X_from
        else:
            NotImplementedError

        # project genes in the "to" data
        ad_ge = cls.tg.project_genes(adata_map=ad_map, adata_sc=ad_sc)

        # get data frame of projected genes
        X_to_pred = ad_ge.to_df()

        # get names for "to_pred" objects
        to_pred_names = X_to_pred.index.values.tolist()
        # get names for "to_pred" features
        to_pred_var = X_to_pred.columns.values.tolist()

        return dict(
            pred=X_to_pred,
            X_to_pred=X_to_pred,
            X_from_pred=None,
            to_pred_names=to_pred_names,
            to_pred_var=to_pred_var,
        )


class TangramV1Pred(TangramPred):
    # Tangram v1 Prediction Method class
    tg = tg1
    version = "1"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class TangramV2Pred(TangramPred):
    # Tangram v2 Prediction Method class
    tg = tg2
    version = "2"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass

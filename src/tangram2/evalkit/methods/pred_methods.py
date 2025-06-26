import os.path as osp
from abc import abstractmethod
from typing import Any, Dict, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

import tangram2.evalkit.methods.policies as pol
import tangram2.evalkit.methods.utils as ut
from tangram2.evalkit.methods._methods import MethodClass


class PredMethodClass(MethodClass):
    # Prediction Method Base class
    ins = ["T", "X_from"]
    outs = ["X_to_pred"]

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
        input_dict: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        pass


class TangramPred(PredMethodClass):
    # specify the tangram version
    version = None

    ins = ["T", "X_from"]
    outs = ["X_to_pred"]

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
        *args,
        spatial_key_to: str = "spatial",
        spatial_key_from: str = "spatial",
        rescale: bool = False,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        # Tangram Prediction Baseclass
        if cls.version == "1":
            import tangram as tg
        elif cls.version == "2":
            import tangram2.mapping as tg
        else:
            raise NotImplementedError

        # get "to" anndata : [n_to] x [n_to_features]
        X_to = input_dict["X_to"]
        # get "from" anndata : [n_from] x [n_from_features]
        X_from = input_dict["X_from"]
        # get "scaled from" anndata, this is necessary
        # due to the adjustment of the "from" data
        # If not None [n_from] x [n_from_features]
        X_from_scaled = input_dict.get("X_from_scaled")
        # get map : [n_to] x [n_from]
        T = input_dict["T"]

        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (X_to.shape[0], X_from.shape[0]))

        T_soft = T.values

        # create anndata object of map
        # tangram functions expects map to be [n_from] x [n_to]
        # hence the transpose
        # TODO : potentially change this to var,obs based on T
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
        ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=ad_sc)

        # get data frame of projected genes
        X_to_pred = ad_ge.to_df()

        if rescale:
            pass

        # get names for "to_pred" objects
        to_pred_names = X_to_pred.index.values.tolist()
        # get names for "to_pred" features
        to_pred_var = X_to_pred.columns.values.tolist()

        out = dict(
            X_to_pred=X_to_pred,
            X_from_pred=None,
            to_pred_names=to_pred_names,
            to_pred_var=to_pred_var,
        )

        pol.check_values(X_to_pred, "X_to_pred")
        pol.check_type(X_to_pred, "X_to_pred")
        pol.check_dimensions(X_to_pred, "X_to_pred", (X_to.shape[0], X_from.shape[1]))

        return out


class Tangram1Pred(TangramPred):
    # Tangram v1 Prediction Method class
    version = "1"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class Tangram2Pred(TangramPred):
    # Tangram v2 Prediction Method class
    version = "2"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class MoscotPred(PredMethodClass):
    # MOSCOT Prediction Method Class

    ins = ["T", "X_from"]
    outs = ["X_to_pred"]

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
        experiment_name: str | None = None,
        eps=1e-12,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:

        T = input_dict.get("T")
        assert T is not None, "T is not found in input"
        X_from = input_dict.get("X_from")
        assert X_from is not None, "X_from is not found in input"
        n_from = X_from.shape[0]

        to_names = T.index
        from_names = T.columns

        # var_names = kwargs.get("prediction_genes", None)
        marginals = kwargs.get("marginals", None)
        marginals = ut.ifnonereturn(marginals, input_dict.get("marginals"))
        marginals = ut.ifnonereturn(marginals, {})
        b = marginals.get("b")
        if b is None:
            b = np.ones(n_from) / n_from

        prediction_genes = kwargs.get("prediction_genes")

        if prediction_genes is None:
            prediction_genes = X_from.var_names

        X_to_pred = T.values @ ((X_from[:, prediction_genes].X) / (b[:, None] + eps))

        X_to_pred = pd.DataFrame(
            X_to_pred,
            index=to_names,
            columns=prediction_genes,
        )

        pol.check_values(X_to_pred, "X_to_pred")
        pol.check_type(X_to_pred, "X_to_pred")
        pol.check_dimensions(
            X_to_pred, "X_to_pred", (T.shape[0], len(prediction_genes))
        )

        out = dict(
            X_to_pred=X_to_pred,
            X_from_pred=None,
            to_pred_names=to_names,
            to_pred_var=prediction_genes,
        )

        return out


class MeanPred(PredMethodClass):
    ins = [("X_from", "X_to")]
    outs = ["X_to_pred"]

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
        experiment_name: str | None = None,
        target: Literal["to", "from"] = "to",
        train_genes: str | None = None,
        test_genes: str | None = None,
        casing: Literal["upper", "lower"] | None = None,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:

        if target == "from":
            X_use = input_dict.get("X_from")
        else:
            X_use = input_dict.get("X_to")

        if X_use is None:
            raise ValueError(
                "X_{} is None, must be adata or pandas dataframe".format(target)
            )

        if isinstance(X_use, ad.AnnData):
            X_use = X_use.to_df()

        match casing:
            case "upper":
                X_use.columns = [x.upper() for x in X_use.columns]
            case "lower":
                X_use.columns = [x.lower() for x in X_use.columns]

        if (train_genes is None) and (test_genes is None):
            raise ValueError("both train and test genes cannot be None")

        all_genes = X_use.columns.tolist()

        if test_genes is None:
            train_genes = set(all_genes).intersection(train_genes)
            test_genes = set(all_genes).difference(train_genes)
        elif train_genes is None:
            test_genes = set(all_genes).intersection(test_genes)
            train_genes = set(all_genes).difference(test_genes)

        X_mean = X_use[train_genes].values.mean(axis=1, keepdims=True)
        X_pred = pd.DataFrame(
            np.repeat(X_mean, len(test_genes), axis=1),
            index=X_use.index,
            columns=test_genes,
        )

        X_pred = pd.concat((X_use[train_genes], X_pred), axis=1)

        return {f"X_{target}_pred": X_pred}

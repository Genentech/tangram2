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
    """Base class for prediction methods."""

    ins = ["T", "X_from"]
    outs = ["X_to_pred"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def run(cls, input_dict: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """Runs expression prediction with provided input.

        Args:
            cls: The class to which the method belongs.
            input_dict: A dictionary containing input data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Dict[str, Any]: A dictionary containing the results.

        Raises:
            None
        """
        pass


class TangramPred(PredMethodClass):
    """Base class for Tangram prediction methods."""

    version = None
    ins = ["T", "X_from"]
    outs = ["X_to_pred"]

    def __init__(self, *args, **kwargs):
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
        """Runs expression prediction using a Tangram model.

        Args:
            cls: The Tangram class instance.
            input_dict: A dictionary containing input data.  Must include "X_to" (pd.DataFrame), "X_from" (pd.DataFrame), and "T" (pd.DataFrame).  May optionally include "X_from_scaled" (pd.DataFrame).
            *args: Variable length argument list.
            spatial_key_to: The key in the input data for the "to" spatial data. Default is "spatial".
            spatial_key_from: The key in the input data for the "from" spatial data. Default is "spatial".
            rescale: Whether to rescale the output. Default is False.
            **kwargs: Arbitrary keyword arguments.  May include "training_genes" or "markers".

        Returns:
            A dictionary containing the prediction results, including a DataFrame ("X_to_pred").

        Raises:
            NotImplementedError: If the Tangram version is not supported.
        """
        if cls.version == "1":
            import tangram as tg
        elif cls.version == "2":
            import tangram2.mapping as tg
        else:
            raise NotImplementedError
        X_to = input_dict["X_to"]
        X_from = input_dict["X_from"]
        X_from_scaled = input_dict.get("X_from_scaled")
        T = input_dict["T"]
        pol.check_type(T, "T")
        pol.check_values(T, "T")
        pol.check_dimensions(T, "T", (X_to.shape[0], X_from.shape[0]))
        T_soft = T.values
        ad_map = ad.AnnData(T_soft.T, var=X_to.obs, obs=X_from.obs)
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
        ad_map.uns["train_genes_df"] = pd.DataFrame([], index=training_genes)
        if cls.version == "2" and X_from_scaled is not None:
            ad_sc = X_from_scaled
        elif cls.version == "1" or cls.version == "2":
            ad_sc = X_from
        else:
            NotImplementedError
        ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=ad_sc)
        X_to_pred = ad_ge.to_df()
        if rescale:
            pass
        to_pred_names = X_to_pred.index.values.tolist()
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
    """Tangram1 prediction class"""

    version = "1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class Tangram2Pred(TangramPred):
    """Tangram2 prediction class"""

    version = "2"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass


class MoscotPred(PredMethodClass):
    """Moscot prediction class"""

    ins = ["T", "X_from"]
    outs = ["X_to_pred"]

    def __init__(self, *args, **kwargs):
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
        """Runs a prediction experiment.

        Args:
          cls: The class this method is bound to (unused).
          input_dict: A dictionary containing input data. Must include "T" and "X_from" keys.  "T" and "X_from" should be pandas DataFrames.  May optionally include "marginals" key (dictionary).
          experiment_name: An optional string specifying the experiment name.
          eps: A small float value used for numerical stability (default: 1e-12).
          **kwargs: Additional keyword arguments.  May include "prediction_genes" (list of strings) and "marginals" (dictionary).

        Returns:
          A dictionary containing the prediction results ("X_to_pred"), and metadata.  "X_to_pred" is a pandas DataFrame.

        Raises:
          AssertionError: If "T" or "X_from" keys are not found in input_dict.
        """
        T = input_dict.get("T")
        assert T is not None, "T is not found in input"
        X_from = input_dict.get("X_from")
        assert X_from is not None, "X_from is not found in input"
        n_from = X_from.shape[0]
        to_names = T.index
        from_names = T.columns
        marginals = kwargs.get("marginals", None)
        marginals = ut.ifnonereturn(marginals, input_dict.get("marginals"))
        marginals = ut.ifnonereturn(marginals, {})
        b = marginals.get("b")
        if b is None:
            b = np.ones(n_from) / n_from
        prediction_genes = kwargs.get("prediction_genes")
        if prediction_genes is None:
            prediction_genes = X_from.var_names
        X_to_pred = T.values @ (X_from[:, prediction_genes].X / (b[:, None] + eps))
        X_to_pred = pd.DataFrame(X_to_pred, index=to_names, columns=prediction_genes)
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
    """Mean vale prediction"""

    ins = [("X_from", "X_to")]
    outs = ["X_to_pred"]

    def __init__(self, *args, **kwargs):
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
        """Runs a prediction process.

        Args:
            input_dict (Dict[str, Any]): A dictionary containing input data.  Must contain either "X_to" or "X_from".
            experiment_name (str | None):  Name of the experiment (optional).
            target (Literal["to", "from"]): Specifies the target data ("to" or "from"). Defaults to "to".
            train_genes (str | None):  Genes to use for training. If None, it will be calculated based on test_genes.
            test_genes (str | None): Genes to use for testing. If None, it will be calculated based on train_genes.
            casing (Literal["upper", "lower"] | None): Case conversion for column names (optional).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the prediction results as a Pandas DataFrame.
                The key will be "X_{target}_pred".

        Raises:
            ValueError: If "X_to" or "X_from" is missing from input_dict, or if both train_genes and test_genes are None.
        """
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
        if train_genes is None and test_genes is None:
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

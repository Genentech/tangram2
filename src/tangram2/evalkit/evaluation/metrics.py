import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, spmatrix
from scipy.spatial import cKDTree
from scipy.stats import hypergeom
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

import tangram2.evalkit.utils.transforms as tf

from . import utils as ut


class MetricClass(ABC):
    # Metrics Base Class

    # indicate what kind of result this is a metric for: {map,pred,grp,dea}
    metric_type: str = ""
    # indicate name of metric
    metric_name: str = ""

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def make_standard_out(cls, value: float | Dict[str, float]) -> Dict[str, float]:
        # method to generate standard output
        # the output of each metric class should adhere
        # to the same format, that easily can be saved using
        # the save method. Returns a dictionary with the names of
        # the metric and the value of the metric.

        # check if value is dictionary, used for multivalue metric classes
        if isinstance(value, dict):
            # expand name with value indicator
            name = [
                cls.metric_type + "_" + cls.metric_name + "_" + k for k in value.keys()
            ]
            return dict(zip(name, value.values()))
        # if not multivalue, then just use standard name
        else:
            name = cls.metric_type + "_" + cls.metric_name
            return {name: value}

    @classmethod
    @abstractmethod
    def score(
        cls, res_dict: Dict[str, Any], ref_dict: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, float]:
        # method to _calculate_ the associated metric(s)
        pass

    @classmethod
    def save(cls, values: Dict[str, float], out_dir: str) -> None:
        # standard save function
        # will save each metric in a text file named
        # metric_name.txt in the indicated output directory
        # `out_dir`

        assert isinstance(
            values, dict
        ), "input to save must be a Dictionary on the format metric_name:metric_value"

        # iterate over all metrics
        for metric_name, value in values.items():
            # define file path to write metric value to
            metric_out_path = osp.join(out_dir, metric_name + ".txt")
            # save metric value
            with open(metric_out_path, "w+") as f:
                f.writelines(str(value))


class PrintMetric(MetricClass):
    """Print Metric

    Simply prints the results,
    no metric is calculated.
    Intended for development
    purposes

    """

    metric_type = "dev"
    metric_name = "print"

    @classmethod
    def get_gt(cls, *args, **kwargs):
        # return empty dictionary for compatibility
        return {}

    @classmethod
    def score(cls, res, *args, **kwargs):
        # print results
        print(res)

    @classmethod
    def save(cls, values, out_dir):
        # do not save anything
        pass


class MapMetricClass(MetricClass):
    metric_type = "map"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PredMetricClass(MetricClass):
    metric_type = "pred"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DEAMetricClass(MetricClass):
    """DEA Metric Baseclass"""

    # remember DEA methods return a dict
    # containing the DE analysis of each one group vs bg|alt_group

    # set type to be dea
    metric_type = "dea"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_gt(cls, input_dict: Dict[Any, str], to_lower: bool = True, **kwargs):
        # ground truth should be a data frame in csv
        # two columns are required signal and effect
        # the signal column holds the feature we condition on
        # the effect column hold the list of associated genes
        # to the given signal

        # get path to ground truth
        gt_path = kwargs["gt_path"]
        # which feature should we look at
        signal_name = kwargs["signal"]
        # group of interest
        group = kwargs["group"]
        # data frame listing signal name and effects
        signal_effect_df = pd.read_csv(gt_path, index_col=0)
        # get effect features for the associated
        gt_genes = signal_effect_df[
            signal_effect_df["signal"].str.upper() == signal_name.upper()
        ]["effect"].values[0]
        # from "[e_1,...,e_k]" to [e_1,...,e_k]
        gt_genes = gt_genes.split(",")
        # convert effect features to lowercase
        if to_lower:
            gt_genes = [gtg.lower() for gtg in gt_genes]

        return dict(true=gt_genes, group_name=group)


class HardMapMetricClass(MapMetricClass):
    """Metric class for hard maps (T)"""

    # hard maps is when each cell is assigned
    # completely to one spot, no spread across all spots

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def _pp(cls, obj: Any):
        if isinstance(obj, dict) and "row_self" in obj:
            new_obj = coo_matrix(
                (
                    np.ones(obj["row_self"].shape[0]),
                    (obj["row_self"], obj["row_target"]),
                ),
                shape=obj["shape"],
            )
            names_self = obj.get("names_self")
            names_target = obj.get("names_target")
            new_obj = pd.DataFrame.sparse.from_spmatrix(
                new_obj,
                index=names_self,
                columns=names_target,
            )

        elif isinstance(obj, np.ndarray):
            new_obj = coo_matrix(obj)
            new_obj = pd.DataFrame.sparse.from_spmatrix(new_obj)
        elif isinstance(obj, pd.DataFrame):
            new_obj = obj.astype(pd.SparseDtype("float", 0))
        else:
            raise NotImplementedError

        return new_obj

    @classmethod
    def _make_T_hard(
        cls, input_dict: Dict[str, Any], make_hard=True, hard_method="argmax"
    ):
        if make_hard:
            T_use = "T_hard"
            if "T_hard" not in input_dict.keys():
                if hard_method == "argmax":
                    params = {
                        "pos_by_argmax": True,
                        "pos_by_weight": False,
                        "S_to": None,
                        "S_from": None,
                    }
                elif hard_method == "weight":
                    params = {
                        "pos_by_argmax": False,
                        "pos_by_weight": True,
                        "S_to": input_dict.get("S_to", None),
                        "S_from": input_dict.get("S_from", None),
                    }
                else:
                    raise NotImplementedError
                input_dict["T_hard"] = tf.soft_T_to_hard(input_dict["T"], **params)
        else:
            T_use = "T"
        return T_use

class MapJaccardDist(HardMapMetricClass):
    metric_name = "jaccard"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(
        cls, res_dict: Dict[str, Any], ref_dict: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, float]:

        T_use = cls._make_T_hard(
            input_dict=res_dict,
            make_hard=kwargs.get("make_hard", True),
            hard_method=kwargs.get("hard_method", "argmax"),
        )
        # get true map
        T_true = cls._pp(ref_dict["T"])
        # get pred map
        T_pred = cls._pp(res_dict[T_use])

        T_true = T_true.sparse.to_coo()
        T_pred = T_pred.sparse.to_coo()

        inter = np.sum(T_pred.multiply(T_true))

        union = np.sum((T_pred + T_true) > 0)

        jc = inter / union

        # create output object
        out = cls.make_standard_out(jc)

        return out


class MapAccuracy(HardMapMetricClass):
    metric_name = "accuracy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(
        cls, res_dict: Dict[str, Any], ref_dict: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, float]:
        # add capability to use hard map for metrics only
        T_use = cls._make_T_hard(
            input_dict=res_dict,
            make_hard=kwargs.get("make_hard", True),
            hard_method=kwargs.get("hard_method", "argmax"),
        )
        # get predicted map
        T_true = cls._pp(ref_dict["T"])

        # get true map
        T_pred = cls._pp(res_dict[T_use])

        inter = np.sum(T_pred.values == T_true.values)
        full = T_true.shape[0] * T_true.shape[1]

        acc = inter / full

        out = cls.make_standard_out(acc)

        return out


class MapF1(HardMapMetricClass):
    metric_name = "f1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(
        cls, res_dict: Dict[str, Any], ref_dict: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, float]:
        from sklearn.metrics import f1_score

        # add capability to use hard map for metrics only
        T_use = cls._make_T_hard(
            input_dict=res_dict,
            make_hard=kwargs.get("make_hard", True),
            hard_method=kwargs.get("hard_method", "argmax"),
        )
        # get predicted map
        T_true = cls._pp(ref_dict["T"])

        # get true map
        T_pred = cls._pp(res_dict[T_use])

        y_pred = T_pred.values.flatten()
        y_true = T_true.values.flatten()

        score = f1_score(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0.0
        )

        out = cls.make_standard_out(score)

        return out


class MapRMSE(MapMetricClass):
    """RMSE for spatial coordinates

    calculates the RMSE between the true
    and the predicted spatial
    coordinates of 'from'.

    """

    metric_name = "rmse"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(
        cls, res_dict: Dict[str, np.ndarray], ref_dict: Dict[str, Any], *args, **kwargs
    ) -> float:
        # get true spatial coordinates for "from"
        S_from_true = ref_dict["S_from"]
        # get predicted spatial coordinates for "from"
        S_from_pred = res_dict["S_from"]

        # rmse : https://en.wikipedia.org/wiki/Root-mean-square_deviation
        rmse = np.sqrt(np.sum((S_from_true - S_from_pred) ** 2, axis=1).mean())

        # create standard output
        out = cls.make_standard_out(rmse)

        return out


class PredLeaveOutScore(PredMetricClass):
    """Leave out(Held out) set score for Pred result"""

    # define name
    metric_name = "loov"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(cls, res_dict: Dict[str, Any], ref_dict: None = None, **kwargs) -> float:
        X_to = res_dict.get("X_to").copy()
        assert X_to is not None, "X_to needs to be in input dictionary"
        if isinstance(X_to, ad.AnnData):
            X_to = X_to.to_df()

        X_to_pred = res_dict.get("X_to_pred").copy()
        assert X_to_pred is not None, "X_to_pred needs to be in input dictionary"
        if isinstance(X_to_pred, ad.AnnData):
            X_to_pred = X_to_pred.to_df()

        if isinstance(X_to_pred, ad.AnnData):
            X_to_pred = X_to_pred.to_df()
        elif not isinstance(X_to_pred, pd.DataFrame):
            raise NotImplementedError

        test_genes = kwargs.get("test_genes", None)
        train_genes = kwargs.get("train_genes", None)
        use_lowercase = kwargs.get("use_lowercase", False)

        if use_lowercase:
            X_to_pred.columns = [x.lower() for x in X_to_pred.columns]
            X_to.columns = [x.lower() for x in X_to.columns]

        if (test_genes is None) and (train_genes is None):
            raise AssertionError("Input either train/test gene set")
        elif test_genes is not None:
            test_genes = ut.list_or_path_get(test_genes)
            if use_lowercase:
                test_genes = [g.lower() for g in test_genes]
            eval_genes = list(
                set(test_genes).intersection(X_to.columns.tolist(), X_to_pred.columns)
            )
        else:
            train_genes = ut.list_or_path_get(train_genes)
            if use_lowercase:
                train_genes = [g.lower() for g in train_genes]
            test_genes = [g for g in X_to.columns.tolist() if g not in train_genes]
            eval_genes = list(set(test_genes).intersection(X_to_pred.columns))

        gex_true = X_to.loc[:, eval_genes].values
        gex_pred = X_to_pred.loc[:, eval_genes].values

        norm_sq = (
            np.linalg.norm(gex_true, axis=1) * np.linalg.norm(gex_pred, axis=1) + 1e-8
        )
        cos_sim = (gex_true * gex_pred).sum(axis=1) / norm_sq
        mean_cos_sim = np.nanmean(cos_sim)
        out = cls.make_standard_out(mean_cos_sim)

        return out



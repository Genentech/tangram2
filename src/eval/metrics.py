import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.stats import hypergeom
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

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
            ).T
        else:
            new_obj = obj

        return new_obj

    @classmethod
    def get_gt(cls, input_dict: Dict[Any, str], key: str | None = None, **kwargs):
        # we get the ground truth from the original
        # data (X_from)
        X_from = input_dict["X_from"]
        obj_map = ut.get_ad_value(X_from, key)
        row_idx = obj_map["row_self"]
        col_idx = obj_map["row_target"]
        n_rows, n_cols = obj_map["shape"]

        T = coo_matrix((np.ones(n_rows), (row_idx, col_idx)), shape=(n_rows, n_cols))

        return dict(true=T.T)


class MapJaccardDist(HardMapMetricClass):
    metric_name = "jaccard"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(
        cls, res_dict: Dict[str, Any], ref_dict: Dict[str, Any], *args, **kwargs
    ) -> Dict[str, float]:

        # get predicted map
        T_true = cls._pp(ref_dict["T"])

        # get true map
        T_pred = cls._pp(res_dict["T"])

        # number of rows
        n_rows = T_pred.shape[0]

        # jaccard helper function
        def _jaccard(u, v):
            inter = np.sum(u * v)
            union = np.sum((u + v) > 0)
            if union < 1:
                return 1
            return inter / union

        jc = 0
        # we do it like this to keep down memory usage
        for ii in range(n_rows):
            # get value for row ("to") i in predicted map
            u_a = T_pred.getrow(ii).toarray().flatten()
            # get value for row i ("to") in true map
            v_a = T_true.getrow(ii).toarray().flatten()
            # compute jaccard distance between true and pred
            jc += _jaccard(u_a, v_a)

        # mean
        jc /= n_rows

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
        T_true = cls._pp(ref_dict["T"])
        T_pred = cls._pp(res_dict["T"])

        # sparse matrices do not work with A * B
        inter = T_pred.multiply(T_true)
        inter = np.sum(inter)
        full = np.sum(T_true)
        acc = inter / full

        out = cls.make_standard_out(acc)

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
    def score(cls, res: Dict[str, np.ndarray], *args, **kwargs) -> float:
        # get true spatial coordinates for "from"
        S_from_true = ref_dict["S_from"]
        # get predicted spatial coordinates for "from"
        S_from_pred = res_dict["S_from"]

        # rmse : https://en.wikipedia.org/wiki/Root-mean-square_deviation
        rmse = np.sqrt(np.sum((S_from - S_from_pred) ** 2, axis=1).mean())

        # create standard output
        out = cls.make_standard_out(rmse)

        return out


class DEAHyperGeom(DEAMetricClass):
    """Hypergeometric Test for DEA result"""

    # define name
    metric_name = "hypergeom"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(
        cls, res_dict: Dict[str, Any], ref_dict: Dict[str, Any], *args, **kwargs
    ) -> float:

        # TODO: I don't like the dependency on X_from
        population = set(res["X_from"].var.index.values)
        pop_size = len(population)

        def _hg_test(true_set, pred_set):
            # number of genes in ground truth
            tot_success = len(set(population).intersection(true_set))
            # number of genes predicted in DEA
            sample_size = len(pred_set)
            # number of drawn successes
            drawn_success = len(set(true_set).intersection(pred_set))

            # cdf = hypergeom.cdf(drawn_success, pop_size, tot_success, sample_size)
            sf = hypergeom.sf(drawn_success - 1, pop_size, tot_success, sample_size)
            return sf

        DEA_pred = res_dict["DEA"]
        DEA_true = ref_dict["DEA"]

        group_names = list(DEA_pred.keys())
        sfs = dict()
        for group_name in group_names:
            true_set = DEA_true.get(group_name, {})
            pred_set = DEA_pred.get(group_name, {})
            if (len(true_set) < 0) or (len(pred_set) < 0):
                continue
            true_set = set(true_set[group_name]["names"].values)
            pred_set = set(pred_set[group_name]["names"].values)

            sf = _hg_test(true_set, pred_set)
            sfs[group_name] = sf

        out = cls.make_standard_out(sfs)

        return out


class DEAAuc(DEAMetricClass):
    """AUC for DEA result"""

    metric_name = "AU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(
        cls, res_dict: Dict[str, Any], ref_dict: Dict[str, Any], *args, **kwargs
    ) -> float:

        # TODO: again, don't like dependence on X_from
        genes = res["X_from"].var.index.values

        def _aupr():
            true_lbl = np.isin(true_set, genes).astype(int)
            pred_lbl = np.isin(pred_set, genes).astype(int)
            # AUC
            auroc = roc_auc_score(true_lbl, pred_lbl)
            # AUPR
            precision, recall, _ = precision_recall_curve(true_lbl, pred_lbl)
            aupr = auc(precision, recall)
            return aupr, auroc

        DEA_pred = res_dict["DEA"]
        DEA_true = ref_dict["DEA"]

        group_names = list(DEA_pred.keys())
        out_res = dict()
        for group_name in group_names:
            true_set = DEA_true.get(group_name, {})
            pred_set = DEA_pred.get(group_name, {})
            if (len(true_set) < 0) or (len(pred_set) < 0):
                continue

            true_set = true_set[group_name]["names"].values
            pred_set = pred_set[group_name]["names"].values

            aupr, auroc = _aupr(true_set, pred_set)
            out_res[group_name + "_PR"] = aupr
            out_res[group_name + "_ROC"] = auroc

        out = cls.make_standard_out(out_res)

        return out

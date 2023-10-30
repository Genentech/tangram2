import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


from . import utils as ut


class MetricClass(ABC):
    metric_type: str = ""
    metric_name: str = ""

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def get_gt(cls, *args, **kwargs):
        pass

    @classmethod
    def make_standard_out(cls, value: float | Dict[str, float]) -> Dict[str, float]:
        if isinstance(value, dict):
            name = [cls.metric_type + "_" + cls.metric_name + "_" + k for k in value.keys()]
            return dict(zip(name, value.values()))
        else:
            name = cls.metric_type + "_" + cls.metric_name
            return {name: value}

    @classmethod
    @abstractmethod
    def score(cls, res: Dict[str, Any], *args, **kwargs) -> Dict[str, float]:
        pass

    @classmethod
    def save(cls, values: Dict[str, float], out_dir: str) -> None:
        for metric_name, value in values.items():
            metric_out_path = osp.join(out_dir, metric_name + ".txt")
            with open(metric_out_path, "w+") as f:
                f.writelines(str(value))


class PrintMetric(MetricClass):

    metric_type = "dev"
    metric_name = "print"

    @classmethod
    def get_gt(cls, *args, **kwargs):
        return {}

    @classmethod
    def score(cls, res, *args, **kwargs):
        print(res)

    @classmethod
    def save(cls, values, out_dir):
        pass


class MapMetricClass(MetricClass):
    metric_type = "map"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DEAMetricClass(MetricClass):
    # remember DEA methods return a dict
    # containing the DE analysis of each group vs bg
    metric_type = "dea"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_gt(cls, input_dict: Dict[Any, str], **kwargs):
        gt_path = kwargs["gt_path"]
        feature_name = kwargs["feature_name"]
        group = kwargs["group"]
        signal_effect_df = pd.read_csv(gt_path, index_col=0)
        gt_genes = signal_effect_df[signal_effect_df['signal'] == feature_name.upper()]['effect'].values[0]
        gt_genes = gt_genes.split(",")
        gt_genes = [gtg.lower() for gtg in gt_genes]
        return dict(true=gt_genes, group_name=group)


class HardMapMetricClass(MapMetricClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_gt(cls, input_dict: Dict[Any, str], key: str | None = None, **kwargs):
        X_from = input_dict["X_from"]
        obj_map = ut.get_ad_value(X_from, key)
        row_idx = obj_map["row_self"]
        col_idx = obj_map["row_target"]
        n_rows, n_cols = obj_map["shape"]

        T = coo_matrix((np.ones(n_rows), (row_idx, col_idx)), shape=(n_rows, n_cols))

        return dict(true=T.T)


class SoftMapMetricClass(MapMetricClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_gt(cls, input_dict: Dict[Any, str], key: str | None = None, **kwargs):
        X_from = input_dict["X_from"]
        S = ut.get_ad_value(X_from, key)

        return dict(true=S)


class MapJaccardDist(HardMapMetricClass):
    metric_name = "jaccard"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(cls, res: Dict[str, coo_matrix], *args, **kwargs) -> float:
        T_true = res["true"]
        T_pred = res["pred"]

        n_rows = T_pred.shape[0]

        def _jaccard(u, v):
            inter = np.sum(u * v)
            union = np.sum((u + v) > 0)
            if union < 1:
                return 1
            return inter / union

        jc = 0
        # we do it like this to keep down memory usage
        for ii in range(n_rows):
            u_a = T_pred.getrow(ii).toarray().flatten()
            v_a = T_true.getrow(ii).toarray().flatten()
            jc += _jaccard(u_a, v_a)

        jc /= n_rows

        out = cls.make_standard_out(jc)

        return out


class MapAccuracy(HardMapMetricClass):
    metric_name = "accuracy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(cls, res: Dict[str, coo_matrix], *args, **kwargs) -> float:
        T_true = res["true"]
        T_pred = res["pred"]

        # sparse matrices do not work with A * B
        inter = T_pred.multiply(T_true)
        inter = np.sum(inter)
        full = np.sum(T_true)
        acc = inter / full

        out = cls.make_standard_out(acc)

        return out


class MapRMSE(SoftMapMetricClass):
    metric_name = "rmse"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(cls, res: Dict[str, np.ndarray], *args, **kwargs) -> float:
        S_true = res["true"]
        S_pred = res["pred"]

        # rmse - S_true is a nx2 matrix
        rmse = np.sqrt(np.sum((S_true - S_pred) ** 2, axis=1).mean())

        out = cls.make_standard_out(rmse)

        return out


class DEAHyperGeom(DEAMetricClass):
    metric_name = "hypergeom"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(cls, res: Dict[str, np.ndarray], *args, **kwargs) -> float:

        group_name = res["group_name"]
        # total number of genes in the dataset
        population = res["X_from"].var.index.values
        pop_size = len(population)
        # number of genes in ground truth
        tot_success = len(set(population).intersection(res["true"]))
        # number of genes predicted in DEA
        sample_size = len(res["pred"][group_name]['names'].values)
        # number of drawn successes
        drawn_success = len(set(res["true"]).intersection(res["pred"][group_name]['names'].values))
        # cdf = hypergeom.cdf(drawn_success, pop_size, tot_success, sample_size)
        sf = hypergeom.sf(drawn_success-1, pop_size, tot_success, sample_size)

        out = cls.make_standard_out(sf)

        return out


class DEAAuc(DEAMetricClass):
    metric_name = "AU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(cls, res: Dict[str, np.ndarray], *args, **kwargs) -> float:
        genes = res["X_from"].var.index.values
        true = [1 if g in res["true"] else 0 for g in genes]
        pred = [1 if g in res["pred"] else 0 for g in genes]
        # AUC
        auroc = roc_auc_score(true, pred)
        # AUPR
        precision, recall, _ = precision_recall_curve(true, pred)
        aupr = auc(precision, recall)

        out = cls.make_standard_out(dict(ROC=auroc, PR=aupr))

        return out

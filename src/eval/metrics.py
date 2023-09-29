from abc import ABC, abstractmethod
from typing import Any, Dict

import anndata as ad
import numpy as np
from scipy.sparse import coo_matrix
import os.path as osp

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
    def make_standard_out(cls, value: float) -> Dict[str, float]:
        name = cls.metric_type + "_" + cls.metric_name
        return {name: value}

    @classmethod
    @abstractmethod
    def score(cls, res: Dict[str, Any], *args, **kwargs) -> Dict[str, float]:
        pass

    @classmethod
    def save(cls, values: Dict[str, float], out_dir: str) -> None:
        for metric_name, value in values.items():
            metric_out_path = osp.join(out_dir, metric_name + '.txt')
            with open(metric_out_path, "w+") as f:
                f.writelines(str(value))


class MapMetricClass(MetricClass):
    metric_type = "map"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HardMapMetricClass(MapMetricClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_gt(cls, ad_to: ad.AnnData, ad_from: ad.AnnData, gt_key: str | None = None, **kwargs):
        obj_map = ut.get_ad_value(ad_from, gt_key)
        row_idx = obj_map["row_self"]
        col_idx = obj_map["row_target"]
        n_rows, n_cols = obj_map["shape"]

        T = coo_matrix((np.ones(n_rows), (row_idx, col_idx)), shape=(n_rows, n_cols))

        return dict(true=T.T)


class SoftMapMetricClass(MapMetricClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_gt(cls, ad_to: ad.AnnData, ad_from: ad.AnnData, gt_key: str | None = None, **kwargs):
        S = ut.get_ad_value(ad_from, gt_key)

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

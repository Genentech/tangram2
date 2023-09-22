import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import coo_matrix
import anndata as ad
from typing import Dict, Any

from . import utils as ut


class MapMetricClass(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def get_gt(cls,ad_to: ad.AnnData, ad_from: ad.AnnData,gt_key: str,*args,**kwargs):
        obj_map = ut.get_ad_value(ad_from, gt_key)
        row_idx = obj_map['row_self']
        col_idx = obj_map['row_target']
        n_rows,n_cols = obj_map['shape']


        T = coo_matrix((np.ones(n_rows), (row_idx, col_idx)), shape=(n_rows, n_cols))

        return dict(true = T.T)


    @classmethod
    @abstractmethod
    def score(cls, res: Dict[str,Any], *args, **kwargs) -> float:
        pass

    @classmethod
    def save(cls, value: float, out_path: str) -> None:
        with open(out_path, "w+") as f:
            f.writelines(str(value))


class MapJaccardDist(MapMetricClass):
    name = "jaccard"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @classmethod
    def score(cls,res: Dict[str,coo_matrix], *args, **kwargs) -> float:
        T_true = res['true']
        T_pred = res['pred']

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

        return jc


class MapAccuracy(MapMetricClass):
    name = "accuracy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def score(cls, res: Dict[str,coo_matrix], *args, **kwargs) -> float:
        T_true = res['true']
        T_pred = res['pred']


        # sparse matrices do not work with A * B
        inter = T_pred.multiply(T_true)
        inter = np.sum(inter)
        full = np.sum(T_true)
        acc = inter / full

        return acc

import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import coo_matrix


class MapMetricClass(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def eval_map(cls, T_pred: np.ndarray, T_true: np.ndarray, *args, **kwargs) -> float:
        pass


class MapJaccardDist(MapMetricClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def eval_map(cls, T_pred: coo_matrix, T_true: coo_matrix, *args, **kwargs) -> float:
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


class MapAccary(MapMetricClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def eval_map(cls, T_pred: coo_matrix, T_true: coo_matrix, *args, **kwargs) -> float:

        inter = np.sum(T_pred @ T_true)
        full = np.sum(T_true)
        acc = inter / full

        return acc



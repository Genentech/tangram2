from abc import ABC, abstractmethod

import anndata as ad
import CeLEry as cel
import numpy as np
import scanpy as sc
from scipy.sparse import spmatrix


class PPClass(ABC):
    @staticmethod
    @abstractmethod
    def pp(adata: ad.AnnData, *args, **kwargs):
        pass


class NormalizeTotal(PPClass):
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        sc.pp.normalize_total(adata, target_sum=float(kwargs.get("target_sum", 1e4)))


class StandardScanpy(PPClass):
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        if isinstance(adata.X, spmatrix):
            adata.X = adata.X.toarray()
        adata.X = adata.X.astype(np.float32)
        sc.pp.normalize_total(adata, target_sum=float(kwargs.get("target_sum", 1e4)))
        sc.pp.log1p(adata)


#TODO: define this class
class CeLEryPP(PPClass):
    pass

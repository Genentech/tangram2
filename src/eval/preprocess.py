import scanpy as sc
import anndata as ad
from abc import ABC, abstractmethod
from scipy.sparse import spmatrix
import numpy as np


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

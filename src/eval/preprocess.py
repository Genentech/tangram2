from abc import ABC, abstractmethod

import anndata as ad
import CeLEry as cel
import numpy as np
import scanpy as sc
import tangram as tg
import tangram2 as tg2
from scipy.sparse import spmatrix


class PPClass(ABC):
    @staticmethod
    @abstractmethod
    def pp(adata: ad.AnnData, *args, **kwargs):
        pass


class IdentityPP(PPClass):
    @staticmethod
    def pp(adata: ad.AnnData, *args, **kwargs):
        return adata


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


class CeLEryPP(PPClass):
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        if isinstance(adata.X, spmatrix):
            adata.X = adata.X.toarray()
        adata.X = adata.X.astype(np.float32)
        cel.get_zscore(adata)


class StandardTangramV1(PPClass):
    @staticmethod
    def pp(adata: ad.AnnData, input_type: str | None = None, **kwargs):
        # redunant rn but might change in future
        match input_type:
            case "X_from" | "sc":
                NormalizeTotal.pp(adata, **kwargs)
            case "X_to" | "sp":
                NormalizeTotal.pp(adata, **kwargs)
            case _:
                NormalizeTotal.pp(adata, **kwargs)


class StandardTangramV2(PPClass):
    @staticmethod
    def pp(adata: ad.AnnData, input_type: str | None = None, **kwargs):
        # redunant rn but might change in future
        match input_type:
            case "X_from" | "sc":
                IdentityPP.pp(adata)
            case "X_to" | "sp":
                IdentityPP.pp(adata)
            case _:
                IdentityPP.pp(adata)

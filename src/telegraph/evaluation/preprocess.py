from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import spmatrix


class PPClass(ABC):
    # Preprocessing Baseclass
    @staticmethod
    @abstractmethod
    def pp(adata: ad.AnnData, *args, **kwargs):
        pass


class IdentityPP(PPClass):
    # Does nothing to the data (identity transform)
    @staticmethod
    def pp(adata: ad.AnnData, *args, **kwargs):
        return adata


class NormalizeTotal(PPClass):
    # normalize total normalization
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        target_sum = kwargs.get("target_sum", None)
        if target_sum is not None:
            target_sum = float(target_sum)
        sc.pp.normalize_total(adata, target_sum=target_sum)


class ScanpyPCA(PPClass):
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):

        n_obs, n_var = adata.shape
        n_comps = kwargs.get("n_comps", None)

        if n_comps is not None:
            if n_obs < n_comps:
                return None
            if n_var < n_comps:
                return None

        sc.pp.pca(
            data=adata,
            n_comps=kwargs.get("n_comps", None),
            svd_solver=kwargs.get("svd_solver", "arpack"),
            random_state=kwargs.get("random_state", 0),
        )


class StandardScanpy(PPClass):
    # common normalization in scanpy
    # NormalizeTotal + Log1p
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        # check if sparse matrix
        if isinstance(adata.X, spmatrix):
            adata.X = adata.X.toarray()
        # change dtype to float32
        adata.X = adata.X.astype(np.float32)
        # normalize total
        target_sum = kwargs.get("target_sum", None)
        if target_sum is not None:
            target_sum = float(target_sum)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        # log1p transform
        sc.pp.log1p(adata)


class CeLEryPP(PPClass):
    # CeLEry normalization
    # Based on: https://github.com/QihuangZhang/CeLEry/blob/main/tutorial/tutorial.md
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        from CeLEry import get_zscore

        # check if sparse matrix
        if isinstance(adata.X, spmatrix):
            adata.X = adata.X.toarray()
        # change dtype to float32 (for autograd framework)
        adata.X = adata.X.astype(np.float32)
        # get zscore
        get_zscore(adata)


class StandardTangramV1(PPClass):
    # Tangram v1 recommended normalization
    @staticmethod
    def pp(adata: ad.AnnData, input_type: str | None = None, **kwargs):
        # redundant rn but might change in future
        # check input type
        match input_type:
            case "X_from" | "sc":
                NormalizeTotal.pp(adata, **kwargs)
            case "X_to" | "sp":
                NormalizeTotal.pp(adata, **kwargs)
            case _:
                NormalizeTotal.pp(adata, **kwargs)


class StandardTangramV2(PPClass):
    # Tangram v2 recommended normalization
    @staticmethod
    def pp(adata: ad.AnnData, input_type: str | None = None, **kwargs):
        # redundant rn but might change in future
        # check input type
        match input_type:
            case "X_from" | "sc":
                IdentityPP.pp(adata)
            case "X_to" | "sp":
                IdentityPP.pp(adata)
            case _:
                IdentityPP.pp(adata)


class StandardSpaOTsc(PPClass):
    # SpaOTsc normalization according to the manuscript
    @staticmethod
    def pp(adata: ad.AnnData, input_type: str | None = None, **kwargs):
        # redundant rn but might change in future
        # check input type
        match input_type:
            case "X_from" | "sc":
                # Normalize total
                sc.pp.normalize_total(
                    adata, target_sum=float(kwargs.get("target_sum", 1e4))
                )
                sc.pp.log1p(adata)
            case "X_to" | "sp":
                sc.pp.normalize_total(
                    adata, target_sum=float(kwargs.get("target_sum", 1e4))
                )

                sc.pp.log1p(adata)
            case _:
                # Normalize total
                sc.pp.normalize_total(
                    adata, target_sum=float(kwargs.get("target_sum", 1e4))
                )
                sc.pp.log1p(adata)


class LowercaseGenes(PPClass):
    # Method to make all genes lowercase
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        adata.var.index = [g.lower() for g in adata.var.index.tolist()]


class StandardMoscot(PPClass):
    # MOSCOT recommended normalization
    @staticmethod
    def pp(adata: ad.AnnData, input_type: str | None = None, **kwargs):
        match input_type:
            case "X_from" | "sc":
                LowercaseGenes.pp(adata)
                StandardScanpy.pp(adata, **kwargs.get("scanpy_pp", {}))
                ScanpyPCA.pp(adata, **kwargs.get("scanpy_pca", {}))
            case "X_to" | "sp":
                LowercaseGenes.pp(adata)
                StandardScanpy.pp(adata, **kwargs.get("scanpy_pp", {}))
                ScanpyPCA.pp(adata, **kwargs.get("scanpy_pca", {}))
            case _:
                LowercaseGenes.pp(adata)
                StandardScanpy.pp(adata, **kwargs.get("scanpy_pp", {}))
                ScanpyPCA.pp(adata, **kwargs.get("scanpy_pca", {}))


class FilterGenes(PPClass):
    # Method to filter genes using scanpy's filter genes method
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        sc.pp.filter_genes(
            data=adata,
            min_counts=kwargs.get("min_counts", None),
            min_cells=kwargs.get("min_cells", None),
            max_counts=kwargs.get("max_counts", None),
            max_cells=kwargs.get("max_cells", None)
        )


class StandardSpaGE(PPClass):
    # SpaGE recommended pre-processing
    @staticmethod
    def pp(adata: ad.AnnData, input_type:str | None = None, **kwargs):
        match input_type:
            case "X_from" | "sc":
                LowercaseGenes.pp(adata)
                FilterGenes.pp(adata, **kwargs)
                StandardScanpy.pp(adata, **kwargs)
            case "X_to" | "sc":
                LowercaseGenes.pp(adata)
                StandardScanpy.pp(adata, **kwargs)
            case _:
                StandardScanpy.pp(adata, **kwargs)


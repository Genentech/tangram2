from abc import ABC, abstractmethod

import anndata as ad
import CeLEry as cel
import numpy as np
import scanpy as sc
import tangram as tg
import tangram2 as tg2
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
        sc.pp.normalize_total(adata, target_sum=float(kwargs.get("target_sum", None)))


class ScanpyPCA(PPClass):
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        sc.tl.pca(
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
        sc.pp.normalize_total(adata, target_sum=float(kwargs.get("target_sum", 1e4)))
        # log1p transform
        sc.pp.log1p(adata)


class CeLEryPP(PPClass):
    # CeLEry normalization
    # Based on: https://github.com/QihuangZhang/CeLEry/blob/main/tutorial/tutorial.md
    @staticmethod
    def pp(adata: ad.AnnData, **kwargs):
        # check if sparse matrix
        if isinstance(adata.X, spmatrix):
            adata.X = adata.X.toarray()
        # change dtype to float32 (for autograd framework)
        adata.X = adata.X.astype(np.float32)
        # get zscore
        cel.get_zscore(adata)


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
                sc.pp.log1p(adata)
            case _:
                # Normalize total
                sc.pp.normalize_total(
                    adata, target_sum=float(kwargs.get("target_sum", 1e4))
                )
                sc.pp.log1p(adata)


# class StandardOTSpatial(PPClass):
#     # Standard PP for OT methods
#     @staticmethod
#     def pp(adata: ad.AnnData, **kwargs):
#         if kwargs is {}:
#             # TODO: warn that 'nan' coordinates are not removed
#             return adata
#         else:
#             keep = adata.obs[[kwargs["x_coords"], kwargs["y_coords"]]].dropna(axis=0).index
#             return adata[keep, :].copy()


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
                return adata
            case "X_to" | "sp":
                LowercaseGenes.pp(adata)
                StandardScanpy.pp(adata, **kwargs.get("scanpy_pp", {}))
                ScanpyPCA.pp(adata, **kwargs.get("scanpy_pca", {}))
                return adata
            case _:
                LowercaseGenes.pp(adata)
                StandardScanpy.pp(adata, **kwargs.get("scanpy_pp", {}))
                ScanpyPCA.pp(adata, **kwargs.get("scanpy_pca", {}))
                return adata

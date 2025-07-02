from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import spmatrix


def chainwrapper(func):

    def wrapper(cls, *args, **kwargs):
        if isinstance(args[0], dict):
            if len(args) > 1:
                if isinstance(args[1], str):
                    args[1] = [args[1]]
            return func(cls, *args, **kwargs)

        mod_args = args[0]
        keep_args = args[1:] if len(args) > 1 else []
        new_args = [dict(_tmp=mod_args), ["_tmp"]] + keep_args

        return func(cls, *new_args, **kwargs)

    return wrapper


class PPClass(ABC):
    # Preprocessing Baseclass

    @classmethod
    @chainwrapper
    def run(cls, input_dict, target_objs=["X_from", "X_to"], **kwargs):
        for obj_name in target_objs:
            obj = input_dict.get(obj_name)
            if obj is not None:
                obj = cls.pp(obj, obj_name=obj_name, **kwargs)
                if obj is not None:
                    input_dict[obj_name] = obj

    @classmethod
    @abstractmethod
    def pp(cls, input_dict, *args, **kwargs):
        pass


class IdentityPP(PPClass):
    # Does nothing to the data (identity transform)
    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        return None


class NormalizeTotal(PPClass):
    # normalize total normalization

    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        target_sum = kwargs.get("target_sum", 1e4)
        if isinstance(obj, ad.AnnData):
            if target_sum is not None:
                target_sum = float(target_sum)
            sc.pp.normalize_total(obj, target_sum=target_sum)
        else:
            obj = obj / (obj.values.sum(axis=1, keepdims=True) + 1e-7) * 1e4
            return obj


class Log1p(PPClass):
    # normalize total normalization

    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        if isinstance(obj, ad.AnnData):
            obj.X = np.log1p(obj.X)
        elif isinstance(obj, pd.DataFrame):
            obj = np.log1p(obj)
        return obj


class ScanpyPCA(PPClass):

    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        n_obs, n_var = obj.shape
        n_comps = kwargs.get("n_comps", None)

        if n_comps is not None:
            if n_obs < n_comps:
                return None
            if n_var < n_comps:
                return None

        sc.pp.pca(
            data=obj,
            n_comps=kwargs.get("n_comps", None),
            svd_solver=kwargs.get("svd_solver", "arpack"),
            random_state=kwargs.get("random_state", 0),
        )


class StandardScanpy(PPClass):
    # common normalization in scanpy
    # NormalizeTotal + Log1p

    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        obj = NormalizeTotal.pp(obj, obj_name, **kwargs)
        obj = Log1p.pp(obj, obj_name, **kwargs)
        return obj


class RegressType(PPClass):
    # common normalization in scanpy
    # NormalizeTotal + Log1p

    @staticmethod
    def pp(obj, obj_name="from", normalize: bool = False, **kwargs):
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        if isinstance(obj, ad.AnnData):
            obj = obj.to_df()

        obs_names = obj.index
        var_names = obj.columns

        if normalize:
            obj = obj.astype(np.float32)
            obj = obj / (obj.sum(axis=0) + 1e-8) * 1e4
            obj = np.log1p(obj)

        n_components = kwargs.get("n_components")
        pca = PCA(n_components=n_components)
        emb = pca.fit_transform(obj)
        n_clusters = kwargs.get("n_clusters", 20)
        km = KMeans(n_clusters=n_clusters)
        cidx = km.fit_predict(emb)

        group_means = np.array([obj[cidx == i].mean(axis=0) for i in np.unique(cidx)])
        expanded_means = group_means[cidx]

        obj_centered = obj - expanded_means
        obj_centered = pd.DataFrame(obj_centered, index=obs_names, columns=var_names)
        return obj_centered


class CeLEryPP(PPClass):
    # CeLEry normalization
    # Based on: https://github.com/QihuangZhang/CeLEry/blob/main/tutorial/tutorial.md

    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        from CeLEry import get_zscore

        # check if sparse matrix
        if isinstance(obj.X, spmatrix):
            obj.X = obj.X.toarray()
        # change dtype to float32 (for autograd framework)
        obj.X = obj.X.astype(np.float32)
        # get zscore
        get_zscore(obj)

        return obj


class StandardTangram1(PPClass):
    # Tangram v1 recommended normalization

    @classmethod
    def pp(cls, obj, obj_name=None, **kwargs):
        # redundant rn but might change in future
        # check input type
        match obj_name:
            case "X_from" | "sc":
                NormalizeTotal.pp(obj, **kwargs)
            case "X_to" | "sp":
                NormalizeTotal.pp(obj, **kwargs)
            case _:
                NormalizeTotal.pp(obj, **kwargs)


class StandardTangram2(PPClass):
    # Tangram v2 recommended normalization
    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        # redundant rn but might change in future
        # check input type
        match obj_name:
            case "X_from" | "sc":
                IdentityPP.pp(obj)
            case "X_to" | "sp":
                IdentityPP.pp(obj)
            case _:
                IdentityPP.pp(obj)


class StandardSpaOTsc(PPClass):
    # SpaOTsc normalization according to the manuscript
    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        match obj_name:
            case "X_from" | "sc":
                # Normalize total
                sc.pp.normalize_total(
                    obj, target_sum=float(kwargs.get("target_sum", 1e4))
                )
                sc.pp.log1p(obj)
            case "X_to" | "sp":
                sc.pp.normalize_total(
                    obj, target_sum=float(kwargs.get("target_sum", 1e4))
                )

                sc.pp.log1p(obj)
            case _:
                # Normalize total
                sc.pp.normalize_total(
                    obj, target_sum=float(kwargs.get("target_sum", 1e4))
                )
                sc.pp.log1p(obj)


class LowercaseGenes(PPClass):
    # Method to make all genes lowercase

    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        obj.var.index = [g.lower() for g in obj.var.index.tolist()]
        # remove duplicates
        keep = ~obj.var.index.duplicated()
        obj = obj[:, keep].copy()
        return obj


class StandardMoscot(PPClass):
    # MOSCOT recommended normalization
    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        match obj_name:
            case "X_from" | "sc":
                obj = LowercaseGenes.pp(obj)
                StandardScanpy.pp(obj, **kwargs.get("scanpy_pp", {}))
                ScanpyPCA.pp(obj, **kwargs.get("scanpy_pca", {}))
                return obj
            case "X_to" | "sp":
                obj = LowercaseGenes.pp(obj)
                StandardScanpy.pp(obj, **kwargs.get("scanpy_pp", {}))
                ScanpyPCA.pp(obj, **kwargs.get("scanpy_pca", {}))
                return obj
            case _:
                obj = LowercaseGenes.pp(obj)
                StandardScanpy.pp(obj, **kwargs.get("scanpy_pp", {}))
                ScanpyPCA.pp(obj, **kwargs.get("scanpy_pca", {}))
                return obj


class gimVIPP(PPClass):

    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        match obj_name:
            case "X_from" | "sc":
                # Remove cells with no gene count
                sc.pp.filter_cells(obj, min_counts=1)
            case "X_to" | "sp":
                # Remove cells with no gene count
                sc.pp.filter_cells(obj, min_counts=1)
            case _:
                # Remove cells with no gene count
                sc.pp.filter_cells(obj, min_counts=1)


class FilterGenes(PPClass):
    # Method to filter genes using scanpy's filter genes method
    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        sc.pp.filter_genes(
            data=obj,
            min_counts=kwargs.get("min_counts", None),
            min_cells=kwargs.get("min_cells", None),
            max_counts=kwargs.get("max_counts", None),
            max_cells=kwargs.get("max_cells", None),
        )


class StandardSpaGE(PPClass):
    # SpaGE recommended pre-processing

    @staticmethod
    def pp(obj, obj_name=None, **kwargs):
        match obj_name:
            case "X_from" | "sc":
                obj = LowercaseGenes.pp(obj)
                FilterGenes.pp(obj, **kwargs)
                StandardScanpy.pp(obj, **kwargs)
                return obj
            case "X_to" | "sc":
                obj = LowercaseGenes.pp(obj)
                StandardScanpy.pp(obj, **kwargs)
                return obj
            case _:
                StandardScanpy.pp(obj, **kwargs)

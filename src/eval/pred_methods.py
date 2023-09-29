from . import utils as ut
from eval._methods import MethodClass
import anndata as ad
import tangram as tg1
import tangram2 as tg2
from scipy.sparse import spmatrix
from typing import Any, Dict
from abs import abstractmethod
import numpy as np
import pandas as pd


class PredMethod(MethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def run(
        cls,
        X_to: Any,
        X_from: Any,
        S_from: np.ndarray | None,
        T: np.ndarray | spmatrix | None = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        pass


class TangramPred(PredMethod):
    tg = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        X_to: ad.AnnData,
        X_from: ad.AnnData,
        S_from: np.ndarray | None,
        T: np.ndarray | spmatrix | None,
        *args,
        spatial_key_to: str = "spatial",
        spatial_key_from: str = "spatial",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        if isinstance(T, spmatrix):
            T_soft = T.todense()
        else:
            T_soft = T.copy()

        ad_map = ad.AnnData(
            T_soft,
            obs=X_to.obs,
            var=X_from.obs,
        )

        ad_ge = cls.tg.project_genes(adata_map=ad_map, adata_sc=X_from)

        X_pred = ad_ge.to_df()

        return dict(X_to_pred=X_pred)


class TangramV1Pred(TangramPred):
    tg = tg1

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass


class TangramV2Pred(TangramPred):
    tg = tg2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pass

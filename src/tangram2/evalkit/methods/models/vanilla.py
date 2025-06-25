from abc import ABC, abstractmethod

import numpy as np
import torch as t
from torch import nn
from torch.nn.functional import cosine_similarity, softmax


class MapBase(nn.Module, ABC):
    def __init__(self, X, Y, device="cpu", temperature: float = 1, **kwargs):
        super().__init__(**kwargs)
        self.X = (
            X if isinstance(X, t.Tensor) else t.tensor(X.astype(np.float32))
        )  # [n_to] x [n_var]
        self.Y = (
            Y if isinstance(Y, t.Tensor) else t.tensor(Y.astype(np.float32))
        )  # [n_from] x [n_var]

        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have same dimension along axis 1")

        self.X = self.X.to(device)
        self.Y = self.Y.to(device)

        self.n_x = self.X.shape[0]
        self.n_y = self.Y.shape[0]
        self.n_v = self.X.shape[1]

        self._T = t.zeros(self.n_x, self.n_y).to(device)
        self._T = nn.Parameter(self._T)

        self.sp = nn.functional.softplus
        self.sigm = nn.functional.sigmoid
        self.sfmx = nn.functional.softmax
        self.temp = temperature

    @property
    def T(
        self,
    ):
        T = self.sfmx(self._T.detach() / self.temp, dim=0)
        return T.cpu().numpy()

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return Y

    def recon_loss(
        self,
    ):
        y_h = self.transform_Y(self.Y)
        T = self.sfmx(self._T / self.temp, dim=0)
        X_pred = t.mm(T, y_h)
        X_true = self.transform_X(self.X)

        loss = (
            -cosine_similarity(X_true, X_pred, dim=1).sum()
            - cosine_similarity(X_true, X_pred, dim=0).sum()
        )

        return loss

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass


class ScaleClass(nn.Module, ABC):
    def __init__(self, n_v, device="cpu", **kwargs):
        super().__init__(**kwargs)

        self._W = t.randn(1, n_v).to(device)
        self._W = nn.Parameter(self._W)

    @property
    def W(
        self,
    ):
        W = self.sigm(self._W.detach())
        return W.cpu().numpy()

    def transform_X(self, X):
        W = self.sigm(self._W.detach())
        return W * Z

    def transform_Y(self, Y):
        W = self.sigm(self._W.detach())
        return Y * W


class SimpleMap(MapBase):
    def __init__(self, X, Y, device="cpu", **kwargs):
        super().__init__(X, Y, device, **kwargs)

    def loss(
        self,
    ):
        loss = self.recon_loss()
        return loss


class SimpleScaleMap(SimpleMap, ScaleClass):
    def __init__(self, X, Y, device="cpu", **kwargs):
        n_v = X.shape[1]
        super().__init__(X=X, Y=Y, n_v=n_v, device=device, **kwargs)

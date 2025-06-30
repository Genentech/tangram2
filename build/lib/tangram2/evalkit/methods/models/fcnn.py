from typing import Callable, List

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import torch as t
from torch import nn
from torch.utils.data import Dataset


class FCNNDataset(Dataset):
    def __init__(self, X, y: np.ndarray | t.Tensor | None = None):
        self.X = X
        if not isinstance(self.X, t.Tensor):
            self.X = t.tensor(X)
        self.Y = y
        if self.Y is None:
            self.train = False
        else:
            self.train = True
            if not isinstance(self.Y, t.Tensor):
                self.Y = t.tensor(Y)
            assert len(self.Y) == len(self.X)
            self.F_out = self.Y.shape[1]

        self.N = len(self.X)
        self.F_in = self.X.shape[1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.train:
            return self.X[idx, :], self.Y[idx, :]
        else:
            return self.X[idx, :]


class FCNNImp(L.LightningModule):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: List[int],
        layer_norm: bool = True,
        dropout: float = 0.2,
        loss_fun: Callable | None = None,
    ):
        super().__init__()

        self.n_in = n_input
        self.n_out = n_output
        self.n_hidden = n_hidden
        self.ln = layer_norm
        self.p = dropout

        if loss_fun is None:
            self.loss_fun = nn.MSELoss()
        else:
            self.loss_fun = loss_fun

        self.linear = nn.Linear(self.n_in, self.n_hidden[0])

        layers = []

        for k in range(1, len(self.n_hidden)):
            if self.ln:
                layers.append(nn.LayerNorm(self.n_hidden[k - 1]))
            layers.append(nn.SELU())
            layers.append(nn.Dropout(self.p))
            layers.append(nn.Linear(self.n_hidden[k - 1], self.n_hidden[k]))

        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(self.n_hidden[-1], self.n_out)

    def forward(self, *batch):
        x = batch[0]
        h = self.linear(x)
        h = self.layers(h)
        y_pred = self.head(h)
        return y_pred

    def _step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.loss_fun(y_pred, y_true)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        train_loss = self._step(batch, batch_idx)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        val_loss = self._step(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=False)
        return val_loss

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# def run_simple_imputation(X_to,X_from, n_hidden, batch_size, log_every_n_steps,n_epochs):
#     pass

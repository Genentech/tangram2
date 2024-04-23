import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray | t.Tensor,
        cond: np.ndarray | t.Tensor | None = None,
    ):
        self.X = X
        self.N = len(self.X)

        if not isinstance(self.X, t.Tensor):
            self.X = t.tensor(X)

        if cond is not None:
            self.C = t.tensor(cond)
            self.n_C_var = self.C.shape[1]

        else:
            self.C = t.zeros((self.N, 1))
            self.n_C_var = 1

        self.n_X_var = self.X.shape[1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx, :], self.C[idx, :]


class CVAE(L.LightningModule):
    def __init__(
        self,
        x_dim,
        hidden_dim,
        z_dim,
        c_dim,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.x_dim = x_dim
        self.c_dim = c_dim
        self.z_dim = z_dim

        self.p = dropout
        # Encoder
        self.fc1 = t.nn.Linear(self.x_dim + self.c_dim, hidden_dim)
        self.fc21 = t.nn.Linear(hidden_dim, z_dim)  # Mean μ
        self.fc22 = t.nn.Linear(hidden_dim, z_dim)  # Log variance σ^2

        # Decoder
        self.fc3 = t.nn.Linear(self.z_dim + self.c_dim, hidden_dim)
        self.fc4 = t.nn.Linear(hidden_dim, self.x_dim)

    def encode(self, x, c):
        h0 = t.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(h0))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h0 = t.cat([z, c], dim=1)
        h3 = F.relu(self.fc3(h0))
        return self.fc4(h3)

    def corrected_features(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        c_new = t.zeros_like(c)
        x_pred = self.decode(z, c_new)

        return x_pred

    def forward(self, input):
        x, c = input
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, c), mu, log_var, z

    def loss(self, x, x_pred, mu, log_var):
        MSE = F.mse_loss(x_pred, x, reduction="sum")
        KLD = -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def _step(self, batch, batch_idx):
        x, c = batch
        x_pred, mu, log_var, z = self.forward(batch)
        loss = self.loss(x, x_pred, mu, log_var)
        batch_size = x.size(0)
        return loss / batch_size

    def training_step(self, batch, batch_idx):
        train_loss = self._step(batch, batch_idx)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        val_loss = self._step(batch, batch_idx)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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
        X,
    ):
        self.X = X
        if not isinstance(self.X, t.Tensor):
            self.X = t.tensor(X)
        self.N = len(self.X)
        self.F_in = self.X.shape[1]
        self.F_out = self.X.shape[1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx, :]


class VAE(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, z_dim, dropout: float = 0.2):
        super().__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim

        self.p = dropout
        # Encoder
        self.fc1 = t.nn.Linear(input_dim, hidden_dim)
        self.fc21 = t.nn.Linear(hidden_dim, z_dim)  # Mean μ
        self.fc22 = t.nn.Linear(hidden_dim, z_dim)  # Log variance σ^2

        # Decoder
        self.fc3 = t.nn.Linear(z_dim, hidden_dim)
        self.fc4 = t.nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, log_var):
        std = t.exp(0.5 * log_var)
        eps = t.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var, z

    def loss(self, x, x_pred, mu, log_var):
        MSE = F.mse_loss(x_pred, x.view(-1, self.input_dim), reduction="sum")
        KLD = -0.5 * t.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return MSE + KLD

    def _step(self, batch, batch_idx):
        x = batch
        x_pred, mu, log_var, z = self.forward(x)
        loss = self.loss(x, x_pred, mu, log_var)
        return loss

    def training_step(self, batch, batch_idx):
        train_loss = self._step(batch, batch_idx)
        self.log("train_loss", train_loss)
        return train_loss

    def training_step(self, batch, batch_idx):
        val_loss = self._step(batch, batch_idx)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

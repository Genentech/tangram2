import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions.normal import Normal


class InteractionModel(pl.LightningModule):
    def __init__(
        self, feature_dim, label_dim, learning_rate=0.001, seed: int = 42, **kwargs
    ):

        super().__init__()

        self.seed = seed

        self.L = label_dim
        self.F = feature_dim

        torch.manual_seed(self.seed)

        self.beta = nn.Parameter(torch.zeros(self.L, self.L, self.F))  # L x L x F
        self.gamma = nn.Parameter(torch.randn(1, self.F))  # F
        self.alpha = nn.Parameter(torch.randn(self.L, self.F))  # L x F
        self.log_scale = nn.Parameter(torch.randn(1, self.F))  # F

        self.lr = learning_rate

        self.eps = 1e7

    def forward(self, p):

        alpha_mu = torch.einsum("nl,lf->nf", p, self.alpha)
        beta_mu = torch.einsum("ni,nj,ijf->nf", p, p, self.beta)
        mu = self.gamma + alpha_mu + beta_mu
        mu = self.clip(mu)

        return mu

    def clip(self, x):
        return torch.clip(x, -self.eps, self.eps)

    def loss(self, x, mu):
        scale = torch.exp(self.clip(self.log_scale))
        log_prob = Normal(loc=mu, scale=scale).log_prob(x)

        return -log_prob.mean()

    def step(self, batch, batch_idx, prefix="train"):
        x, p = batch
        mu = self(p)
        loss = self.loss(x, mu)
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(
        self,
        batch,
        batch_idx,
    ):
        return self.step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

import pytorch_lightning as pl
import torch
from torch import nn
from torch.distributions.normal import Normal


class InteractionModel(pl.LightningModule):
    def __init__(
        self, feature_dim, label_dim, learning_rate=0.001, seed: int = 42, **kwargs
    ):
        """
        Initialize the InteractionModel with trainable parameters and hyperparameters.

        Args:
            feature_dim (int): Dimensionality of the input features.
            label_dim (int): Number of label categories.
            learning_rate (float): Learning rate for the optimizer. Default is 0.001.
            seed (int): Random seed for parameter initialization. Default is 42.
            **kwargs: Additional keyword arguments (currently unused).
        """

        super().__init__()

        self.seed = seed

        self.L = label_dim
        self.F = feature_dim

        torch.manual_seed(self.seed)

        self.beta = nn.Parameter(torch.randn(self.L, self.L, self.F))  # L x L x F
        self.gamma = nn.Parameter(torch.randn(1, self.F))  # F
        self.alpha = nn.Parameter(torch.randn(self.L, self.F))  # L x F
        self.log_scale = nn.Parameter(torch.randn(1, self.F))  # F

        self.lr = learning_rate

        self.eps = 1e7

    def forward(self, d: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model computing the predicted output.

        Args:
            d (torch.Tensor): Label indicator tensor of shape (n_samples, n_labels).
            p (torch.Tensor): Effective neighborhood tensor of shape (n_samples, n_labels).

        Returns:
            torch.Tensor: Predicted output of shape (n_samples, n_features).
        """

        alpha_mu = torch.einsum("nl,lf->nf", d, self.alpha)
        beta_mu = torch.einsum("ni,nj,ijf->nf", d, p, self.beta)
        mu = self.gamma + alpha_mu + beta_mu
        mu = self.clip(mu)

        return mu

    def clip(self, x):
        """
        Clip tensor values to lie within the range [-eps, eps].

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Clipped tensor.
        """

        return torch.clip(x, -self.eps, self.eps)

    def loss(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative log-likelihood loss based on a Normal distribution.

        Args:
            x (torch.Tensor): Ground truth tensor of shape (n_samples, n_features).
            mu (torch.Tensor): Predicted mean tensor of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Scalar loss value.
        """

        scale = torch.exp(self.clip(self.log_scale))
        log_prob = Normal(loc=mu, scale=scale).log_prob(x)

        return -log_prob.mean()

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: torch.Tensor,
        prefix: str = "train",
    ) -> torch.Tensor:
        """
        Perform a single optimization step: compute predictions, evaluate loss, and log it.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A batch containing (x, d, p).
            batch_idx (int): Index of the batch.
            prefix (str): Logging prefix (e.g., "train", "val"). Default is "train".

        Returns:
            torch.Tensor: Computed loss.
        """

        x, d, p = batch
        mu = self(d, p)
        loss = self.loss(x, mu)
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the training step used by PyTorch Lightning during model training.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A batch containing (x, d, p).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed training loss.
        """

        return self.step(batch, batch_idx)

    def configure_optimizers(self):
        """
        Configure the optimizer used for training.

        Returns:
            torch.optim.Optimizer: Initialized Adam optimizer with model parameters.
        """

        return torch.optim.Adam(self.parameters(), lr=self.lr)

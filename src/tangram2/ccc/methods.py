from typing import Any, Dict

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xarray as xr
from pytorch_lightning import loggers as pl_loggers
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from .base import MethodClass
from .models import InteractionModel as IM

__all__ = ["TangramCCC"]


class TangramCCC(MethodClass):
    """Tangram Cell-Cell Communication (CCC) method for cell-cell interaction analysis."""

    ins = ["X_from", "D_from"]
    outs = ["D_from"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def train(
        cls,
        X: np.ndarray,
        D: np.ndarray,
        P: np.ndarray,
        n_epochs: int = 1000,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        n_components: int = 500,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train the TangramCCC model on input feature data, interaction matrix, and property matrix.

        Args:
            X (np.ndarray): Feature matrix of shape (n_obs, n_features).
            D (np.ndarray): Interaction matrix of shape (n_obs, n_obs).
            P (np.ndarray): Property matrix of shape (n_obs, n_properties).
            n_epochs (int): Number of training epochs. Default is 1000.
            learning_rate (float): Learning rate for optimizer. Default is 0.001.
            batch_size (int): Batch size for training. Default is 256.
            n_components (int): Number of PCA components to use. Default is 500.
            seed (int): Random seed for reproducibility. Default is 42.
            verbose (bool): Whether to display training progress. Default is True.
            **kwargs: Additional keyword arguments for customization.

        Returns:
            Dict[str, Any]: Dictionary containing learned matrices 'alpha', 'beta', and 'gamma'.
        """

        n_obs, n_var = X.shape
        n_prop = P.shape[1]

        n_components = min(n_obs, n_var, n_components)
        batch_size = min(n_obs, batch_size)

        pca = PCA(n_components=n_components)

        Z = pca.fit_transform(X)
        V = pca.components_

        inX = torch.tensor(Z.astype(np.float32))
        inD = torch.tensor(D.values.astype(np.float32))
        inP = torch.tensor(P.values.astype(np.float32))

        dataset = TensorDataset(inX, inD, inP)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        model = IM(
            feature_dim=inX.shape[1],
            label_dim=inP.shape[1],
            learning_rate=learning_rate,
            seed=seed,
        )

        device = "gpu" if torch.cuda.is_available() else "cpu"

        trainer = pl.Trainer(
            max_epochs=n_epochs, accelerator=device, enable_progress_bar=verbose
        )
        trainer.fit(model, dataloader)

        beta = np.dot(model.beta.detach().cpu().numpy(), V)
        alpha = np.dot(model.alpha.detach().cpu().numpy(), V)
        gamma = np.dot(model.gamma.detach().cpu().numpy(), V)

        return dict(alpha=alpha, beta=beta, gamma=gamma)

    @classmethod
    def format_output(cls, res: Dict[str, Any], feature_names, prop_names):
        """
        Format model outputs into an xarray.Dataset with labeled dimensions.

        Args:
            res (Dict[str, Any]): Dictionary containing 'alpha', 'beta', and 'gamma' arrays.
            feature_names (List[str]): Names of the input features.
            prop_names (List[str]): Names of the predicted properties.

        Returns:
            xr.Dataset: Structured dataset with coordinates and dimensions.
        """

        coords = {
            "features": feature_names,
            "labels": prop_names,
            "labels_": prop_names,
        }

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                "alpha": (["labels", "features"], res.pop("alpha")),
                "gamma": (
                    ["features"],
                    res.pop("gamma").flatten(),
                ),  # Flatten gamma to 1D since it's Fx1
                "beta": (
                    ["labels", "labels_", "features"],
                    res.pop("beta"),
                ),  # Use L1 and L2 to differentiate dimensions in beta
            },
            coords=coords,
        )

        return ds

    @classmethod
    def run_with_adata(
        cls,
        adata: ad.AnnData,
        P: pd.DataFrame,
        label_col: str,
        layer: str | None = None,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Run the TangramCCC method using an AnnData object and a property matrix.

        Args:
            adata (ad.AnnData): Annotated data matrix with gene expression values.
            P (pd.DataFrame): Property matrix with cell type proportions.
            label_col (str): Column name indicating cell type labels.
            layer (str | None): Optional layer in AnnData to use for expression. Defaults to None.
            seed (int): Random seed for reproducibility. Default is 42.
            verbose (bool): Whether to display progress. Default is True.
            **kwargs: Additional arguments passed to the training method.

        Returns:
            xr.Dataset: Dataset containing formatted alpha, beta, and gamma matrices.
        """

        X = adata.to_df(layer=layer).values
        prop_names = P.columns.tolist()
        feature_names = X.columns.tolist()

        res = cls.train(X, P.values, seed=seed, verbose=verbose, **kwargs)

        res = cls.format_output(res, feature_names, prop_names)

        return res

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        seed: int = 42,
        verbose: bool = True,
        return_neighborhood: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Run the TangramCCC method using an AnnData object and a property matrix.

        Args:
            adata (ad.AnnData): Annotated data matrix with gene expression values.
            P (pd.DataFrame): Property matrix with cell type proportions.
            label_col (str): Column name indicating cell type labels.
            layer (str | None): Optional layer in AnnData to use for expression. Defaults to None.
            seed (int): Random seed for reproducibility. Default is 42.
            verbose (bool): Whether to display progress. Default is True.
            **kwargs: Additional arguments passed to the training method.

        Returns:
        xr.Dataset: Dataset containing formatted alpha, beta, and gamma matrices.

        """
        T = input_dict.get("T", None)
        D_from = input_dict.get("D_from", None)
        X_use = input_dict.get("X_from", None)

        for name, obj in zip(["T", "D_from", "X_to/X_to_pred"], [T, D_from, X_use]):
            if obj is None:
                raise ValueError(f"object {name} is None")

        if isinstance(X_use, ad.AnnData):
            X_use = X_use.to_df(layer=kwargs.get("layer", None))

        P = pd.DataFrame(
            np.dot(np.dot(T.T, T), D_from), index=T.columns, columns=D_from.columns
        )

        if input_dict.get("w") is not None:
            # TODO: this is unclear
            w = input_dict["w"]
            w.index = w["cell_type"]
            P = P / w.loc[P.columns, "coefficient"]

        P = P / P.values.sum(axis=1, keepdims=True)

        feature_names = X_use.columns.tolist()
        prop_names = P.columns

        res = cls.train(X_use, D_from, P, seed=seed, verbose=verbose, **kwargs)

        res = cls.format_output(res, feature_names, prop_names)

        if return_neighborhood:
            return res, P

        return res

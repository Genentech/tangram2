from typing import Any, Dict, List, Tuple

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import torch as t
from lightning.pytorch.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from ...methods._methods import MethodClass
from .models import CVAE, VAEDataset


class CVAEVanilla(MethodClass):
    ins = ["X_from", "D_from"]
    outs = ["D_from"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def embedd(
        cls,
        X: np.ndarray,
        C: np.ndarray,
        z_dim: int = 16,
        hidden_dim: int = 64,
        n_epochs: int = 400,
        p_train: float = 0.8,
        random_state: int = 13,
        batch_size=256,
        use_wandb: bool = False,
        **kwargs,
    ) -> np.ndarray:

        X = X.astype(np.float32)
        C = C.astype(np.float32)

        batch_size = min(batch_size, len(X))

        X_train_val, X_test, C_train_val, C_test = train_test_split(
            X, C, train_size=p_train, random_state=random_state
        )
        X_train, X_val, C_train, C_val = train_test_split(
            X_train_val, C_train_val, train_size=p_train, random_state=random_state
        )

        train_data = VAEDataset(X_train, C_train)
        val_data = VAEDataset(X_val, C_val)
        test_data = VAEDataset(X_test, C_test)
        pred_data = VAEDataset(X, C)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        pred_dataloader = DataLoader(pred_data, batch_size=batch_size, shuffle=False)

        logger = WandbLogger() if use_wandb else None

        trainer = L.Trainer(
            max_epochs=n_epochs,
            log_every_n_steps=None,
            logger=logger,
            callbacks=RichProgressBar(),
        )

        model_args = dict(
            x_dim=train_data.n_X_var,
            z_dim=z_dim,
            c_dim=train_data.n_C_var,
            hidden_dim=hidden_dim,
        )
        model = CVAE(**model_args)
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        pred = trainer.predict(model=model, dataloaders=pred_dataloader)
        Z = np.vstack([x[1].cpu().numpy() for x in pred])

        with t.no_grad():
            X_hat = np.vstack(
                [
                    model.corrected_features(x, c).cpu().numpy()
                    for x, c in pred_dataloader
                ]
            )

        return X_hat, Z

    @classmethod
    def stratify(
        cls,
        signal_values: np.ndarray,
        embedding: np.ndarray,
        label_values: np.ndarray,
        target_label: str,
        source_label: str = None,
        signal_name: str = "signal",
        min_group_count: int = 20,
        **kwargs,
    ):

        from scipy.spatial import cKDTree
        from sklearn.cluster import KMeans

        x = signal_values
        is_target = np.where(label_values == target_label)[0]
        km = KMeans(n_clusters=2, n_init="auto")

        if source_label is None:
            clu_idx = km.fit_predict(x[:, None])
            clu_cnt = km.cluster_centers_.flatten()
            ordr = np.argsort(clu_cnt)
            low_clu = ordr[0]
            high_clu = ordr[-1]
            is_high = clu_idx == high_clu
            is_low = clu_idx == low_clu
        else:
            is_source = np.where(label_values == source_label)[0]
            clu_idx = km.fit_predict(x[is_source, None])
            clu_cnt = km.cluster_centers_.flatten()
            ordr = np.argsort(clu_cnt)
            low_clu = ordr[0]
            high_clu = ordr[-1]
            is_high = is_source[clu_idx == high_clu]
            is_low = is_source[clu_idx == low_clu]

        Z = embedding
        kd = cKDTree(Z[is_target])
        _, high_idxs = kd.query(Z[is_high], k=5)
        high_idxs = np.unique(high_idxs.flatten())
        high_idxs = is_target[high_idxs]

        v_sr = np.array(["background"] * len(x), dtype=object)
        v_sr[is_high] = "sender"
        v_sr[is_target] = "non-receiver"
        v_sr[high_idxs] = "receiver"

        return v_sr

    @classmethod
    def run_with_adata(
        cls,
        adata: ad.AnnData,
        signal_and_label: List[Tuple[str, str]] | List[Tuple[str, str, str]],
        label_col: str,
        batch_col: str | None = None,
        use_existing_emb: bool = False,
        emb_key: str = "X_cvae",
        min_group_count: int = 20,
        **kwargs,
    ):

        X = adata.to_df().values
        X_col = adata.var_names

        if (use_existing_emb) and (emb_key in adata.obsm):
            E = adata.obsm[emb_key].copy()
        else:
            C = pd.get_dummies(adata.obs[label_col]).astype(int)
            if batch_col is not None:
                B = pd.get_dummies(adata.obs[batch_col]).astype(int)
                C = pd.concat((C, B), axis=1)

            C_col = C.columns
            C = C.values

            X_hat, E = cls.embedd(X, C, **kwargs)
            adata.obsm["X_cvae"] = E
            adata.layers["X_rec"] = X_hat

        if isinstance(signal_and_label[0], str):
            signal_and_label = [signal_and_label]

        labels = adata.obs[label_col].values

        for signal_name, *st_labels in signal_and_label:
            if signal_name not in X_col:
                continue

            if len(st_labels) == 2:
                target_label, source_label = st_labels
            else:
                source_label, target_label = None, st_labels[0]

            signal_values = adata.obs_vector(signal_name).flatten()
            signal_indicator = cls.stratify(
                signal_values,
                embedding=E,
                label_values=labels,
                target_label=target_label,
                source_label=source_label,
                signal_name=signal_name,
            )

            adata.obs[f"ixn_{signal_name}_{target_label}"] = signal_indicator

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        feature_name: List[str] | str,
        add_complement: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        pass

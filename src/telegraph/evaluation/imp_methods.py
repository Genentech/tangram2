import os.path as osp
import tempfile
from abc import abstractmethod
from typing import Any, Callable, Dict, List

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import torch as t
from sklearn.model_selection import train_test_split

import telegraph.evaluation.policies as pol
import telegraph.evaluation.utils as ut
from telegraph.evaluation._methods import MethodClass


class ImpMethodClass(MethodClass):
    # Imputation Method Base class
    ins = ["X_to", "X_from"]
    outs = ["X_to_pred"]

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
        input_dict: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        pass


class MeanImputation(ImpMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        **kwargs,
    ):
        from scipy.spatial import cKDTree

        X_to = input_dict.get("X_to")
        assert X_to is not None
        X_from = input_dict.get("X_from")
        assert X_from is not None

        if isinstance(X_to, ad.AnnData):
            X_to = X_to.to_df()
        elif isinstance(X_to, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        if isinstance(X_from, ad.AnnData):
            X_from = X_from.to_df()
        elif isinstance(X_from, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        to_var = set(X_to.columns)

        from_var = set(X_from.columns)

        X_from_mean = np.mean(X_from.values, axis=0, keepdims=True)

        X_to_pred = np.repeat(X_from_mean, X_to.shape[0], axis=0)

        X_to_pred = pd.DataFrame(
            X_to_pred,
            index=X_to.index,
            columns=X_from.columns,
        )

        out = dict(X_to_pred=X_to_pred)

        return out


class KNNImputation(ImpMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        n_neighs: int = 10,
        train_var: List[str] | None = None,
        **kwargs,
    ):
        from scipy.spatial import cKDTree

        X_to = input_dict.get("X_to")
        assert X_to is not None
        X_from = input_dict.get("X_from")
        assert X_from is not None

        if isinstance(X_to, ad.AnnData):
            X_to = X_to.to_df()
        elif isinstance(X_to, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        if isinstance(X_from, ad.AnnData):
            X_from = X_from.to_df()
        elif isinstance(X_from, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        to_var = set(X_to.columns)
        from_var = set(X_from.columns)
        inter_var = from_var.intersection(to_var)

        if train_var is None:
            train_var = list(inter_var.intersection(inter_var))

        kd = cKDTree(X_from.loc[:, train_var].values)

        dists, idxs = kd.query(X_to.loc[:, train_var].values, k=n_neighs)
        ndists = dists / dists.sum(axis=1, keepdims=True)

        X_to_pred = np.sum(X_from.values[idxs, :] * ndists[:, :, None], axis=1)

        X_to_pred = pd.DataFrame(
            X_to_pred,
            columns=list(from_var),
            index=X_to.index,
        )

        out = dict(X_to_pred=X_to_pred)

        return out


class TrainValImputationClass(ImpMethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def train_model(
        self,
        model,
        train_dataloader,
        val_dataloader,
        model_class,
        model_args,
        n_epochs,
        log_every_n_steps=1,
    ):
        from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=tmpdir,
                filename="test-{epoch:02d}-{val_loss}",
                save_top_k=1,
            )

            trainer = L.Trainer(
                max_epochs=n_epochs,
                log_every_n_steps=log_every_n_steps,
                callbacks=[checkpoint_callback, RichProgressBar()],
            )

            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            del model

            best_model_path = trainer.checkpoint_callback.best_model_path

            best_model = model_class.load_from_checkpoint(
                best_model_path,
                **model_args,
            )

        return best_model, trainer


class FCNNImputation(TrainValImputationClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        n_hidden: List[int] = 50,
        n_epochs: int = 100,
        log_every_n_steps: int = 1,
        batch_size: int | None = None,
        subset_pred_to_from_vars: bool = False,
        train_var: List[str] | None = None,
        **kwargs,
    ) -> Dict[str, Any]:

        from torch.utils.data import DataLoader

        from telegraph.evaluation.models.fcnn import FCNNDataset, FCNNImp

        X_to = input_dict.get("X_to")
        assert X_to is not None
        X_from = input_dict.get("X_from")
        assert X_from is not None

        if isinstance(X_to, ad.AnnData):
            X_to = X_to.to_df()
        elif isinstance(X_to, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        if isinstance(X_from, ad.AnnData):
            X_from = X_from.to_df()
        elif isinstance(X_from, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        to_var = set(X_to.columns)
        from_var = set(X_from.columns)
        inter_var = from_var.intersection(to_var)

        if train_var is None:
            train_var = list(inter_var)
            pred_var = list(from_var.difference(to_var))
        else:
            train_var = list(set(train_var).intersection(inter_var))
            pred_var = list(from_var.difference(train_var))

        to_names = X_to.index
        to_vars = X_to.columns

        # use U,V to avoid confusion with X_to,X_from values in input_dict
        U_from = t.tensor(X_from.loc[:, train_var].values.astype(np.float32))
        V_from = t.tensor(X_from.loc[:, pred_var].values.astype(np.float32))

        U_to = t.tensor(X_to.loc[:, train_var].values.astype(np.float32))

        U_from_train, U_from_val, V_from_train, V_from_val = train_test_split(
            U_from, V_from
        )

        train_data = FCNNDataset(U_from_train, V_from_train)
        val_data = FCNNDataset(U_from_val, V_from_val)
        pred_data = FCNNDataset(U_to)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        pred_dataloader = DataLoader(pred_data, batch_size=batch_size, shuffle=False)

        model_args = dict(
            n_input=train_data.F_in,
            n_output=train_data.F_out,
            n_hidden=ut.listify(n_hidden),
        )

        model = FCNNImp(**model_args)

        best_model, trainer = cls.train_model(
            model,
            train_dataloader,
            val_dataloader,
            FCNNImp,
            model_args,
            n_epochs,
            log_every_n_steps,
        )

        V_to = trainer.predict(model=best_model, dataloaders=pred_dataloader)
        V_to = [x.cpu().numpy() for x in V_to]
        V_to = np.vstack(V_to)

        X_to_pred = np.hstack((X_to.loc[:, train_var].values, V_to))

        X_to_pred = pd.DataFrame(
            X_to_pred,
            columns=train_var + pred_var,
            index=to_names,
        )

        if subset_pred_to_from_vars:
            X_to_pred = X_to_pred.loc[:, X_from.columns]

        return dict(X_to_pred=X_to_pred)


class VAEKNNImputation(TrainValImputationClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        n_latent: int = 32,
        n_hidden: int = 128,
        n_neighs: int = 10,
        batch_size: int | None = None,
        dropout: float = 0.2,
        n_epochs: int = 100,
        train_var: List[str] | None = None,
        log_every_n_steps: int = 1,
    ):

        from torch.utils.data import DataLoader

        from telegraph.evaluation.models import vae

        X_to = input_dict.get("X_to")
        assert X_to is not None
        X_from = input_dict.get("X_from")
        assert X_from is not None

        if isinstance(X_to, ad.AnnData):
            X_to = X_to.to_df()
        elif isinstance(X_to, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        if isinstance(X_from, ad.AnnData):
            X_from = X_from.to_df()
        elif isinstance(X_from, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        to_var = set(X_to.columns)
        from_var = set(X_from.columns)
        inter_var = from_var.intersection(to_var)

        if train_var is None:
            train_var = list(inter_var)
        else:
            train_var = list(train_var.intersection(inter_var))

        to_names = X_to.index
        to_vars = X_to.columns

        # use U,V to avoid confusion with X_to,X_from values in input_dict
        U_from = t.tensor(X_from.loc[:, train_var].values.astype(np.float32))
        U_to = t.tensor(X_to.loc[:, train_var].values.astype(np.float32))
        U_all = t.vstack((U_from, U_to))

        U_to_train, U_to_val = train_test_split(U_to)
        U_from_train, U_from_val = train_test_split(U_from)

        U_all_train = np.vstack((U_to_train, U_from_train))
        U_all_val = np.vstack((U_to_val, U_from_val))

        del U_to_train, U_to_val, U_from_train, U_from_val

        train_data = vae.VAEDataset(
            U_all_train,
        )
        val_data = vae.VAEDataset(U_all_val)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        model_args = dict(
            input_dim=train_data.F_in,
            hidden_dim=n_hidden,
            z_dim=n_latent,
            dropout=dropout,
        )

        model = vae.VAE(**model_args)

        best_model, trainer = cls.train_model(
            model,
            train_dataloader,
            val_dataloader,
            vae.VAE,
            model_args,
            n_epochs,
            log_every_n_steps,
        )

        del train_data
        del val_data
        del U_all

        to_dataset = vae.VAEDataset(U_to)
        to_dataloader = DataLoader(to_dataset, batch_size=batch_size, shuffle=False)

        from_dataset = vae.VAEDataset(U_from)
        from_dataloader = DataLoader(from_dataset, batch_size=batch_size, shuffle=False)

        out_from = trainer.predict(model=best_model, dataloaders=from_dataloader)
        out_from = np.vstack([x[-1].cpu().numpy() for x in out_from])
        out_to = trainer.predict(model=best_model, dataloaders=to_dataloader)
        out_to = np.vstack([x[-1].cpu().numpy() for x in out_to])

        from scipy.spatial import cKDTree

        kd = cKDTree(out_from)

        dists, idxs = kd.query(out_to, k=n_neighs)
        ndists = dists / dists.sum(axis=1, keepdims=True)

        n_from = X_from.shape[0]

        X_to_pred = np.sum(X_from.values[idxs, :] * ndists[:, :, None], axis=1)

        X_to_pred = pd.DataFrame(
            X_to_pred,
            columns=list(from_var),
            index=to_names,
        )

        out = dict(X_to_pred=X_to_pred)

        return out


class SpaGEImputation(ImpMethodClass):

    ins = ["X_to", "X_from"]
    outs = ["X_to_pred"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        genes: str | List | None = None,
        n_pca: int = 30,
        lowercase_var_names: bool = True,
        **kwargs
    ):
        from SpaGE.main import SpaGE

        # get single cell spatial data X_to
        X_to = input_dict.get("X_to")
        assert X_to is not None
        # get single cell RNASeq data X_from
        X_from = input_dict.get("X_from")
        assert X_from is not None

        if isinstance(X_to, ad.AnnData):
            X_to = X_to.to_df()
        elif isinstance(X_to, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        if isinstance(X_from, ad.AnnData):
            X_from = X_from.to_df()
        elif isinstance(X_from, pd.DataFrame):
            pass
        else:
            raise NotImplementedError

        if lowercase_var_names:
            X_to.columns = [x.lower() for x in X_to.columns]
            X_from.columns = [x.lower() for x in X_from.columns]

        # policy checks
        pol.check_values(X_to, "X_to")
        pol.check_type(X_to, "X_to")
        pol.check_values(X_from, "X_from")
        pol.check_type(X_from, "X_from")

        if genes is not None:
            genes = ut.listify(genes)
            genes = [g.lower() for g in genes]

        imp_genes = SpaGE(X_to, X_from, n_pv=n_pca, genes_to_predict=genes)
        X_to_pred = pd.concat((X_to.loc[:,X_to.columns.difference(imp_genes)], imp_genes), axis=1)

        out = dict(X_to_pred=X_to_pred)

        return out

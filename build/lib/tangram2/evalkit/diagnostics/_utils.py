import inspect
from functools import wraps
from typing import Dict

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.decomposition import PCA
from torch.cuda import is_available


def easy_input(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) < 1:
            first_arg = list(kwargs.values())[0]
        else:
            first_arg = args[0]

        if isinstance(first_arg, dict):

            input_dict = first_arg

            sig = inspect.signature(func)

            args_names = [
                x.name
                for x in sig.parameters.values()
                if x.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
            kwargs_names = [
                x.name
                for x in sig.parameters.values()
                if x.kind == inspect.Parameter.KEYWORD_ONLY
            ]

            # add_args = [v for k, v in input_dict.items() if k in args_names]
            add_args = [input_dict[k] for k in args_names if k in input_dict]

            add_kwargs = {k: v for k, v in input_dict.items() if k in kwargs_names}

            new_args = list(args[1::]) + add_args
            kwargs.update(add_kwargs)

            out = func(*new_args, **kwargs)
        else:
            out = func(*args, **kwargs)
        return out

    return wrapper


def _get_X_and_labels(
    X,
    D=None,
    labels=None,
    layer=None,
    obsm=None,
    group_col: str | None = None,
):

    assert any([D is not None, labels is not None])

    if labels is not None:
        assert labels.shape[0] == X.shape[0]

    if D is not None:
        assert D.shape[0] == X.shape[0]
        assert group_col is not None
        if len(group_col) == 1:
            labels = D[group_col[0]].values.flatten()
            labels = np.array([f"{group_col[0]}_{lab}" for lab in labels])
        else:
            labels = (
                D[group_col]
                .apply(lambda row: ", ".join(row.index[row == 1]), axis=1)
                .values
            )
            labels[labels == ""] = "background"

        if isinstance(X, ad.AnnData):
            if obsm is None:
                Xn = X.to_df(layer=layer)
            else:
                Xn = X.obsm[obsm]
                if isinstance(Xn, np.ndarray):
                    Xn = pd.DataFrame(Xn, index=X.obs_names)

    return Xn, labels


def _subset_helper(D=None, labels=None, subset_cols=None, subset_mode=None):

    if subset_cols is None:
        keep_idx = np.ones(len(D)) if D is not None else np.ones(len(labels))
        return keep_idx.astype(bool)

    subset_mode = "u" if subset_mode.startswith("uni") else "i"

    if D is not None:
        if subset_cols is not None:
            if isinstance(subset_cols, str):
                subset_cols = [subset_cols]

            if subset_mode == "i":
                keep_idx = np.ones(len(D))
                for col in subset_cols:
                    ind = D[col].values.astype(float).flatten()
                    keep_idx *= ind
                keep_idx = keep_idx.astype(bool)
            else:
                keep_idx = np.zeros(len(D))
                for col in subset_cols:
                    ind = D[col].values.astype(float).flatten()
                    keep_idx += ind
                keep_idx = keep_idx > 0
    elif labels is not None:
        keep_idx = labels == subset_cols

    return keep_idx


def pick_x_d(X_to, D_to, X_from, D_from, target, use_pred=False, X_to_pred=None):
    if target == "to":
        if use_pred and X_to_pred is not None:
            X, D = X_to_pred, D_to

        elif X_to is not None:
            X, D = X_to, D_to
        else:
            target = "from"

    if target == "from":
        if X_from is not None:
            X, D = X_from, D_from
        else:
            raise ValueError("One of X_to/X_from has to be provided.")
    return X, D


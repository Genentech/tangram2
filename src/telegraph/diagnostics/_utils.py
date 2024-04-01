import inspect
from functools import wraps
from typing import Dict

import anndata as ad
import numpy as np
import pandas as pd
from harmony import harmonize
from sklearn.decomposition import PCA
from torch.cuda import is_available


def easy_input(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], dict):
            input_dict = args[0]
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


def pick_x_d(X_to, D_to, X_from, D_from, target):
    if target == "to":
        if X_to is not None:
            X, D = X_to, D_to
        else:
            target = "from"

    if target == "from":
        if X_from is not None:
            X, D = X_from, D_from
        else:
            raise ValuError("One of X_to/X_from has to be provided.")
    return X, D


def harmony_helper(
    X,
    metadata,
    batch_key,
    metadata_is_D=True,
    n_components=2,
    normalize: bool = True,
):

    if isinstance(X, np.ndarray):
        X_n = X.copy()
    elif isinstance(X, pd.DataFrame):
        X_n = X.values.copy()
    elif isinstance(X, ad.AnnData):
        X_n = X.X.copy()
    else:
        raise ValueError(
            "X in wrong format. Provide X as either of {np.ndarray,pd.DataFrame,ad.AnnData}"
        )

    _n_components = min(X_n.shape[1], n_components)

    pca = PCA(n_components=_n_components)

    X_n = pca.fit_transform(X_n)

    labels = metadata[batch_key]
    if metadata_is_D:
        labels = pd.DataFrame(
            labels.apply(
                lambda row: "_".join(
                    [labels.columns[i] for i in range(len(row)) if row[i] == 1]
                ),
                axis=1,
            )
        )
        labels.columns = ["batch"]
        batch_key = "batch"

    if is_available():
        use_gpu = True
    else:
        use_gpu = False

    Z = harmonize(X_n, labels, batch_key=batch_key, use_gpu=use_gpu)

    return Z[:, 0:n_components]

import inspect
from functools import wraps
from typing import Dict

import anndata as ad
import numpy as np
import pandas as pd
from harmony import harmonize


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

            add_args = [v for k, v in input_dict.items() if k in args_names]
            add_kwargs = {k: v for k, v in input_dict.items() if k in kwargs_names}

            new_args = list(args[1::]) + add_args
            kwargs.update(add_kwargs)

            out = func(*new_args, **kwargs)
        else:
            out = func(*args, **kwargs)
        return out

    return wrapper


def harmony_helper(X, metadata, batch_key, metadata_is_D=True, n_componetns=2):

    if isinstance(X, np.ndarray):
        X_n = X.copy()
    elif isinstance(X, pd.DataFrame):
        X_n = X.values.copy()
    elif isinstance(X, ad.AnnData):
        X_n = X.X.copy()
    else:
        raise ValueError("X in wrong format")

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
        print(labels)

    Z = harmonize(X, labels, batch_key=batch_key)

    return Z[:, 0:n_components]

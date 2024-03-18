import inspect
from functools import wraps
from typing import Dict

import anndata as ad
import numpy as np


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

from typing import Any, Dict, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _parse_feature_list_or_dea_df(obj, to_lower: bool = False):
    if to_lower:
        obj_trans = lambda x: x.lower()
    else:
        obj_trans = lambda x: x

    if isinstance(obj, pd.DataFrame):
        features = obj.names.values.tolist()
    else:
        features = obj

    return [obj_trans(x) for x in features]


def tabulate_dea_overlap(
    *inputs,
) -> pd.DataFrame:
    names = list()
    sets = list()
    for obj_i in inputs:
        if isinstance(obj_i, str):
            names.append(obj_i)
        elif isinstance(obj_i, pd.DataFrame):
            sets.append(set([x.lower() for x in obj_i.names.values.tolist()]))
        elif isinstance(obj_i, list):
            sets.append(set(obj_i))
        elif isinstance(obj_i, set):
            sets.append(obj_i)
        else:
            raise ValueError("Can handle input of type {}".format(type(obj_i)))

    n_sets = len(sets)
    n_names = len(sets)

    if n_sets != n_names:
        names = ["Set {}".append(x) for x in range(n_sets)]

    omat = np.zeros((n_sets, n_sets), dtype=int)
    for i, set_i in enumerate(sets):
        for j, set_j in enumerate(sets):
            omat[i, j] = len(set_i.intersection(set_j))
    omat = pd.DataFrame(omat, index=names, columns=names)

    return omat


def plot_dea_overlap(*inputs, cmap=plt.cm.autumn_r):
    if len(inputs) == 1 and isinstance(inputs[0], pd.DataFrame):
        omat = inputs[0]
    else:
        omat = tabulate_dea_overlap(*inputs)

    sns.heatmap(omat, annot=True, fmt="d", cmap=cmap)
    plt.show()


def _interpret_dea_features_gprofiler(features, **kwargs):
    from gprofiler import GProfiler

    default_method_params = dict(organism="hsapiens")

    method_params = {k: v for k, v in kwargs.items()}
    for k, v in default_method_params.items():
        if k not in method_params:
            method_params[k] = v

    gp = GProfiler(return_dataframe=True)

    out = gp.profile(
        query=features,
        **method_params,
    )

    return out


def interpret_dea_features(
    input: pd.DataFrame | List[str],
    method: Literal["g:Profiler"] = "g:Profiler",
    **kwargs,
):

    if isinstance(input, dict):
        features = dict()
        for key, val in input.items():
            features[key] = _parse_feature_list_or_dea_df(val)
    else:
        features = _parse_feature_list_or_dea_df(input)

    lmet = method.lower()

    if lmet.startswith("g:prof") or lmet.startswith("gprof"):
        out = _interpret_dea_features_gprofiler(features, **kwargs)
    else:
        raise NotImplementedError(f"Method {method} if not implemented yet")

    return out

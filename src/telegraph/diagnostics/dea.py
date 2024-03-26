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
            raise ValueError("Can't handle input of type {}".format(type(obj_i)))

    n_sets = len(sets)
    n_names = len(sets)

    if n_sets != n_names:
        names = [f"Set {x}" for x in range(n_sets)]

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
        out["tg_id"] = out["query"].values
    else:
        raise NotImplementedError(f"Method {method} if not implemented yet")

    return out


def plot_top_k_enrichment_results(
    *dfs,
    name_col="name",
    value_col="p_value",
    value_is_p_value=True,
    max_name_len=None,
    top_k=20,
    id_col="tg_id",
):
    """Plot top K enrichement results

    Will plot the top_k enrichment results from an enrichment analysis (e.g., provided by the
    function `interpret_dea_features`).

    Args:
        dfs: pd.DataFrame (unlimited number)
        name_col: column where the geneset/pathway name can be found
        value_col: column where the values to sort by/pick top k from can be found
        value_is_p_value: whether values are p-values or not
        max_name_len: maximal number of characters of geneset/pathway name to be printed in plot
        top_k: number of top pathways to plot
        id_col: column to split analysis by
    Returns:
        Plots the top K genesets/pathways using matplotlib
    """

    n_rows = len(dfs)
    h_scale = top_k / 20
    fig, ax = plt.subplots(n_rows, 1, figsize=(12, 6 * n_rows * h_scale))
    if hasattr(ax, "__len__"):
        ax.flatten()

    for ii, df in enumerate(dfs):
        id = df[id_col].values[0]
        names = df[name_col].values.tolist()
        if max_name_len is None:
            max_name_len = np.inf
        names = np.array(
            [
                f"{k + 1}. " + x[0 : min(max_name_len, len(x))]
                for k, x in enumerate(names)
            ]
        )
        values = df[value_col].values
        ordr = np.argsort(values)
        names = names[ordr]
        values = values[ordr]
        k = min(top_k, len(values))
        names = names[0:k]
        values = values[0:k]

        if value_is_p_value:
            values = -np.log(values)

        ax[ii].barh(names, values, edgecolor="black", color="red")
        ax[ii].invert_yaxis()
        ax[ii].set_xlabel(f"-log[{value_col}]")
        ax[ii].set_title("Analysis: {}".format(id))
        ax[ii].spines["top"].set_visible(False)
        ax[ii].spines["right"].set_visible(False)

    fig.tight_layout()

    plt.show()

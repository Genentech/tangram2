from typing import Any, Dict, List, Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from . import _utils as ut


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


def compute_dea_auroc(
    dea_df: pd.DataFrame,
    effect: List[str],
    score_by: str = "logfoldchanges",
    pval_cutoff: float = None,
    min_overlap: int | None = None,
    abs_transform: bool = False,
    use_best_up_down: bool = False,
    only_use_overlap: bool = True,
    ascending: bool = True,
    name_col: str = "names",
    pval_col: str = "pvals_adj",
):

    out = dict(fpr=[], tpr=[], num_effect=len(effect), auroc=0)

    dedf = dea_df.copy()

    if pval_cutoff is not None:
        dedf = dedf[dedf[pval_col].values < pval_cutoff]

    if score_by not in dedf.columns:
        print(f"WARNING : {sort_by} not found in the columns of the dea object")
        return out
    else:
        if abs_transform:
            adj_score_key = f"abs_{score_by}"
            dedf[adj_score_key] = np.abs(dedf[score_by].values)
        else:
            adj_score_key = score_by

    y_true = np.array([x in effect for x in dedf[name_col].values]).astype(int)

    if len(np.unique(y_true)) < 2:
        print("No overlap between effect and threshold")
        return out

    if min_overlap is not None:
        if np.sum(y_true) < min_overlap:
            return out

    y_score = dedf[adj_score_key].values

    if not only_use_overlap:
        e_set = set(effect)
        n_set = set(dedf[name_col].values.tolist())
        n_e_diff = n_set.difference(e_set)
        n_diff = len(n_e_diff)
        y_true = np.append(y_true, np.ones(n_diff))
        y_score = np.append(y_score, y_score.min() * np.ones(n_diff))

    # up-reg
    fpr_up, tpr_up, _ = roc_curve(y_true, y_score)
    auroc_up = roc_auc_score(y_true, y_score)
    # down-reg

    if use_best_up_down and not abs_transform:
        fpr_down, tpr_down, _ = roc_curve(y_true, -y_score)
        auroc_down = roc_auc_score(y_true, -y_score)
    else:
        auroc_down = -np.inf

    if auroc_up > auroc_down:
        fpr, tpr, auroc = (fpr_up, tpr_up, auroc_up)
    else:
        fpr, tpr, auroc = (fpr_down, tpr_down, auroc_down)

    out["fpr"] = fpr
    out["tpr"] = tpr
    out["auroc"] = auroc

    return out


def plot_dea_auroc(
    dea_auroc_res: Dict[str, Any],
    side_size: float = 4,
    title: str = None,
    ax: plt.Axes | None = None,
    return_fig_ax: bool = False,
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(side_size, side_size))
    else:
        fig = None

    ax.plot(dea_auroc_res["fpr"], dea_auroc_res["tpr"], ".-")
    auroc = dea_auroc_res["auroc"]
    n_effect = dea_auroc_res["num_effect"]

    ax.set_title(f"{title} \n AUROC: {auroc:0.2f} | #effect : {n_effect}")
    ax.plot([0, 1], [0, 1], linestyle="dashed")

    fig.tight_layout()

    if return_fig_ax:
        return fig, ax


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


@ut.easy_input
def plot_signature_enrichment(
    group_col: str = None,
    *,
    X_to=None,
    D_to=None,
    X_from=None,
    D_from=None,
    labels=None,
    feature_list: List[str] = None,
    target: Literal["to", "from"] = "to",
    signature_score_params: Dict[str, Any] = {},
    score_name: str | None = None,
    subset_cols: List[str] | None = None,
    subset_mode: Literal["intersection", "union"] = "intersection",
    hide_background: bool = False,
):

    from scanpy.pl import violin
    from scanpy.tl import score_genes

    if feature_list is None:
        raise ValueError("a feature list must be provided")

    X, D = ut.pick_x_d(X_to, D_to, X_from, D_from, target)
    keep_idx = ut._subset_helper(
        D=D, labels=labels, subset_cols=subset_cols, subset_mode=subset_mode
    )
    Xn, labels = ut._get_X_and_labels(X, D=D, labels=labels, group_col=group_col)

    Xn = Xn.iloc[keep_idx]
    labels = labels[keep_idx]

    if hide_background:
        Xn = Xn.iloc[labels != "background"]
        labels = labels[labels != "background"]

    _adata = ad.AnnData(
        Xn,
        obs=pd.DataFrame([], index=Xn.index),
        var=pd.DataFrame([], index=Xn.columns),
    )

    _adata.obs["label"] = labels

    score_name = "score" if score_name is None else score_name
    score_genes(_adata, gene_list=feature_list, score_name=score_name)
    violin(_adata, keys=score_name, groupby="label")

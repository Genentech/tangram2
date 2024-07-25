from typing import Any, Dict, List, Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    ndcg_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)

from . import _utils as ut


def _precision_recall_curve(y_true, y_pred):
    y, x, _ = precision_recall_curve(y_true, y_pred)
    return y[1:-1], x[1:-1]


def _get_sample_weights(y_true):
    n_tot = len(y_true)
    w_pos = float(n_tot / np.sum(y_true))

    sample_weights = np.ones_like(y_true, dtype=float)
    sample_weights[y_true == 1] = w_pos
    return sample_weights.astype(float)


def _ndcg_fun(y_true, y_pred):
    return ndcg_score([y_true], [y_pred])


def _hypergeom_fun(y_true, y_pred):
    from scipy.stats import hypergeom

    ordr = np.argsort(y_pred)[::-1]
    y_true_tmp = y_true[ordr].astype(float)
    n_pos = int(np.sum(y_true_tmp))
    n_tot = len(y_true_tmp)

    cum_sum = np.cumsum(y_true_tmp)
    N = n_tot
    K = n_pos

    pvals = np.array([hypergeom.pmf(cum_sum[n], N, K, n + 1) for n in range(n_tot)])
    pvals = -np.log(pvals)

    return pvals


def _top_k_fun(y_true, y_pred):
    ordr = np.argsort(y_pred)[::-1]
    y_true_tmp = y_true[ordr].astype(float)

    return np.cumsum(y_true_tmp)


def _top_k_score(y_true, y_pred):

    score = _top_k_fun(y_true, y_pred)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    max_area = n_pos**2 / 2 + n_neg * n_pos
    obs_area = np.trapz(score)
    res = obs_area / max_area

    return res


def _top_k_curve(y_true, y_pred):

    score = _top_k_fun(y_true, y_pred)
    score = score / np.sum(y_true)
    x_vals = np.arange(len(score))

    return x_vals, score, None


score_types = dict(
    auroc={
        "score_fun": roc_auc_score,
        "curve_fun": roc_curve,
        "x_pos": 0,
        "y_pos": 1,
        "y_name": "TPR",
        "x_name": "FPR",
    },
    aupr={
        "score_fun": average_precision_score,
        "curve_fun": _precision_recall_curve,
        "x_pos": 1,
        "y_pos": 0,
        "y_name": "Precision",
        "x_name": "Recall",
    },
    topk={
        "score_fun": _top_k_score,
        "curve_fun": _top_k_curve,
        "x_pos": 0,
        "y_pos": 1,
        "x_name": "K",
        "y_name": "Overlap",
    },
    ndcg={
        "score_fun": _ndcg_fun,
        "curve_fun": _top_k_curve,
        "x_pos": 0,
        "y_pos": 1,
        "x_name": "Rank",
        "y_name": "Overlap",
    },
)


_recommended_reverse = {
    "logfoldchanges": False,
    "pvals_adj": True,
    "pvals": True,
    "score": False,
}


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


def _rank_curve(y_true, y_score):
    ordr = np.argsort(y_score)[::-1]
    y_true = y_true[ordr]

    n_pos = np.cumsum(y_true)
    n_tot = np.arange(len(y_true)) + 1
    max_ix = np.argmax(n_pos == n_pos.max())
    n_pos = n_pos[0:max_ix]
    n_tot = n_tot[0:max_ix]

    return n_tot, n_pos, None


def _rank_score(y_true, y_score):
    n_tot, n_pos, _ = _rank_curve(y_true, y_score)
    score = np.mean(n_pos / n_tot)
    return score


def _process_dea_df(
    dedf,
    score_by: str = "logfoldchanges",
    name_col: str = "names",
    pval_col: str = "pvals_adj",
    pval_cutoff: float | None = None,
    score_cutoff: float | None = None,
    abs_transform: bool = False,
    reverse: bool = False,
    top_k: int | None = None,
):

    if score_by in _recommended_reverse:
        if _recommended_reverse[score_by] != reverse:
            print(
                f'[WARNING] : You are using "score_by={score_by}" with setting "reverse={str(reverse)}". This is not the recommended setting.'
            )

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

    if reverse:
        dedf[adj_score_key] = -dedf[adj_score_key]

    if score_cutoff is not None:
        dedf = dedf[dedf[score_by].values > score_cutoff]

    if top_k is not None:
        dedf = dedf.sort_values(
            by=adj_score_key, ascending=_recomended_reverse[reverse]
        )
        dedf = dedf.iloc[0:top_k, :]

    dedf.index = dedf[name_col]

    return dedf, adj_score_key


def compute_dea_score(
    dea_df: pd.DataFrame,
    effect: List[str],
    method: Literal["auroc", "top_k_f1"],
    score_by: str = "logfoldchanges",
    score_cutoff: float | None = None,
    pval_cutoff: float = None,
    min_overlap: int | None = None,
    abs_transform: bool = False,
    use_best_up_down: bool = False,
    reverse: bool = False,
    name_col: str = "names",
    pval_col: str = "pvals_adj",
):

    dedf = dea_df.copy()

    og_names = dedf[name_col].values

    dedf, adj_score_key = _process_dea_df(
        dedf,
        score_by=score_by,
        name_col=name_col,
        pval_col=pval_col,
        score_cutoff=score_cutoff,
        abs_transform=abs_transform,
        reverse=reverse,
        pval_cutoff=pval_cutoff,
    )

    pred_names = dedf[name_col].values

    y_true = np.array([x in effect for x in og_names]).astype(float)
    y_pred = np.array(
        [
            float(dedf.loc[x, adj_score_key]) if x in pred_names else np.nan
            for x in og_names
        ]
    ).astype(float)

    y_pred[np.isnan(y_pred)] = np.min(y_pred[~np.isnan(y_pred)]) - 1

    score_fun, curve_fun = (
        score_types[method]["score_fun"],
        score_types[method]["curve_fun"],
    )

    score = score_fun(y_true, y_pred)
    curve_res = curve_fun(y_true, y_pred)
    curve_x, curve_y = (
        curve_res[score_types[method]["x_pos"]],
        curve_res[score_types[method]["y_pos"]],
    )
    num_effect = np.sum(y_true)
    num_og_effect = len(effect)

    return {
        "score": score,
        "x": curve_x,
        "y": curve_y,
        "num_effect_overlap": num_effect,
        "num_effect_og": num_og_effect,
        "score_info": {"name": method},
    }


def compute_dea_auroc_score(
    dea_df: pd.DataFrame,
    effect: List[str],
    score_by: str = "logfoldchanges",
    score_cutoff: float | None = None,
    pval_cutoff: float = None,
    min_overlap: int | None = None,
    abs_transform: bool = False,
    use_best_up_down: bool = False,
    only_use_overlap: bool = True,
    reverse: bool = False,
    name_col: str = "names",
    pval_col: str = "pvals_adj",
    au_type: Literal["auroc", "aupr", "rank"] = "auroc",
):

    print(
        '[WARNING] : This function is depracated. Please use "compute_dea_score" instead with "method = auroc".'
    )

    au_types = dict(
        auroc={
            "score_fun": roc_auc_score,
            "curve_fun": roc_curve,
            "x_pos": 0,
            "y_pos": 1,
            "y_name": "TPR",
            "x_name": "FPR",
        },
        aupr={
            "score_fun": average_precision_score,
            "curve_fun": precision_recall_curve,
            "x": 1,
            "y": 0,
            "y_name": "Precision",
            "x_name": "Recall",
        },
        topk={
            "score_fun": rank_score,
            "curve_fun": rank_curve,
            "x_pos": 0,
            "y_pos": 1,
            "x_name": "top K",
            "y_name": "Correct",
        },
    )

    score_fun, curve_fun = au_types[au_type]

    out = dict(fpr=[], tpr=[], num_effect=len(effect))
    out[au_type] = 0

    dedf = dea_df.copy()
    dedf, adj_score_key = _process_dea_df(
        dedf,
        score_by=score_by,
        name_col=name_col,
        pval_col=pval_col,
        score_cutoff=score_cutoff,
        abs_transform=abs_transform,
        reverse=reverse,
    )

    y_true = np.array([x in effect for x in dedf[name_col].values]).astype(int)

    if len(np.unique(y_true)) < 2:
        if np.sum(y_true) > 1:
            out[au_type] = 1
            return out
        else:
            print("No overlap between effect and results")
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

    if reverse:
        y_score = -y_score

    # up-reg
    fpr_up, tpr_up, _ = curve_fun(y_true, y_score)
    auroc_up = score_fun(y_true, y_score)

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
    out[au_type] = auroc

    return out


def plot_dea_score(
    dea_score_res: Dict[str, Any],
    side_size: float = 4,
    title: str = None,
    ax: plt.Axes | None = None,
    return_fig_ax: bool = False,
    plot_diagonal: bool = False,
    method: str | None = None,
):

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(side_size, side_size))
    else:
        fig = None

    ax.plot(dea_score_res["x"], dea_score_res["y"], ".-")
    score = dea_score_res["score"]
    n_effect = dea_score_res["num_effect_overlap"]
    if method is None:
        method = dea_score_res["score_info"]["name"]

    if "score_info" in dea_score_res:
        method = dea_score_res["score_info"]["name"]
        x_label = score_types[method].get("x_name", "X")
        y_label = score_types[method].get("y_name", "Y")
    else:
        method = "Unknown"
        x_label = "X"
        y_label = "Y"

    ax.set_title(f"{method}: {score:0.4f} | #effect : {n_effect}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if plot_diagonal:
        ax.plot([0, 1], [0, 1], linestyle="dashed")

    if fig is not None:
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
    else:
        ax = [ax]

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
    X_to_pred=None,
    labels=None,
    feature_list: List[str] = None,
    target: Literal["to", "from"] = "to",
    signature_score_params: Dict[str, Any] = {},
    score_name: str | None = None,
    subset_cols: List[str] | None = None,
    subset_mode: Literal["intersection", "union"] = "intersection",
    hide_background: bool = False,
    use_pred: bool = False,
):

    from scanpy.pl import violin
    from scanpy.tl import score_genes

    if feature_list is None:
        raise ValueError("a feature list must be provided")

    X, D = ut.pick_x_d(X_to, D_to, X_from, D_from, target, use_pred, X_to_pred)
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
    violin(_adata, keys=score_name, groupby="label", rotation=90)

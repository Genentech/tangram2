from typing import Any, Dict, List

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import _utils as ut


def _get_T(T: pd.DataFrame | np.ndarray, normalize_T: bool = True):
    if isinstance(T, pd.DataFrame):
        T_n = T.values
    else:
        T_n = T.copy()

    if normalize_T:
        T_n = T_n / (T_n.sum(axis=0, keepdims=True) + 1e-9)

    return T_n


@ut.easy_input
def plot_top_k_distribution(
    T: pd.DataFrame | np.ndarray,
    ks: int | List[int] = 1,
    bins: int = 100,
    ax=None,
    normalize_T: bool = True,
    plot: bool = True,
    alpha: float = 0.3,
    edgecolor=None,
):

    # T: [n_to] x [n_from]

    T_n = _get_T(
        T,
        normalize_T=normalize_T,
    )

    if not isinstance(ks, (list, tuple)):
        ks = [ks]

    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    for i, k in enumerate(ks):
        T_k_max = np.sort(T_n, axis=0)
        T_k_max = np.flipud(T_k_max)[k - 1, :]

        ax.hist(
            T_k_max,
            bins=bins,
            label="top {}".format(k),
            alpha=alpha,
            edgecolor=edgecolor,
            density=True,
        )

    ax.legend()

    if plot:
        plt.show()

    else:
        return fig, ax


@ut.easy_input
def plot_cells_per_spot(
    T: np.ndarray | pd.DataFrame,
    bins: int | None = 25,
    normalize_T: bool = True,
    plot: bool = True,
    plt_kwargs: Dict[str, Any] | None = None,
):

    T_n = _get_T(T, normalize_T)

    T_hard_idx = np.argmax(T_n, axis=0)
    T_hard = np.zeros(T_n.shape)
    T_hard[T_hard_idx, np.arange(T_hard.shape[1])] = 1

    cells_per_spot_soft = np.sum(T_n, axis=1)
    cells_per_spot_hard = np.sum(T_hard, axis=1)

    if plt_kwargs is None:
        _plt_kwargs = {}
    else:
        _plt_kwargs = {k: v for k, v in plt_kwargs.items()}

    plt_kwargs_defaults = dict(edgecolor="black", facecolor="blue")

    for k, v in plt_kwargs_defaults.items():
        if k not in _plt_kwargs:
            _plt_kwargs[k] = v

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title("effective number of cells assigned to each spot (soft)")
    ax[0].hist(
        cells_per_spot_soft.flatten(),
        bins=bins,
        **_plt_kwargs,
    )

    ax[1].set_title("absolute number of cells assigned to each spot (hard)")
    ax[1].hist(
        cells_per_spot_hard.flatten(),
        bins=bins,
        **_plt_kwargs,
    )

    fig.tight_layout()

    if plot:
        plt.show()
    else:
        return fig, ax


@ut.easy_input
def plot_from_feature_on_to(
    S_to: np.ndarray | pd.DataFrame,
    T: pd.DataFrame | np.ndarray,
    features: pd.DataFrame | pd.Series,
    check_index: bool = True,
    n_cols: int = 4,
    marker_size=10,
):

    if isinstance(S_to, pd.DataFrame) and isinstance(T, pd.DataFrame):

        if check_index:
            inter = S_to.index.intersection(T.index)
            union = S_to.index.union(T.index)
            assert len(union) == len(inter)
            S_u = S_to.loc[inter, :].values
            T_u = T.loc[inter, :].values

    else:
        if isinstance(S_to, pd.DataFrame):
            S_u = S_to.values
        else:
            S_u = S_to
        if isinstance(T, pd.DataFrame):
            T_u = T.values
        else:
            T_u = T

    if isinstance(features, (pd.DataFrame, pd.Series)):
        if len(features.shape) == 1:
            if isinstance(features.values[0], str):
                M_v = pd.get_dummies(features).astype(float)
            else:
                M_v = np.reshape(features, (-1, 1))
        else:
            M_v = features
        cat_names = M_v.columns
    else:
        if len(features.shape) == 1:
            if isinstance(features[0], str):
                M_v = pd.get_dummies(
                    pd.DataFrame(features, columns=["feature"]),
                    prefix="",
                    prefix_sep="",
                ).astype(float)
                cat_names = M_v.columns
            else:
                M_v = features.reshape(-1, 1)
        else:
            M_v = features
            cat_names = ["feat_{}".format(x) for x in range(M_v.shape[1])]

    M_u = np.dot(T, M_v)  #  [to x from ] x [from x feat]

    n_cols = min(n_cols, M_u.shape[1])
    n_rows = int(np.ceil(M_u.shape[1] / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if hasattr(ax, "__len__"):
        ax = ax.flatten()
    else:
        ax = [ax]

    for k in range(M_u.shape[1]):

        ax[k].scatter(S_u[:, 0], S_u[:, 1], c=M_u[:, k], s=marker_size)
        ax[k].set_title(cat_names[k])
        ax[k].set_xticks([])
        ax[k].set_xticklabels([])
        ax[k].set_yticks([])
        ax[k].set_yticklabels([])

    for axx in ax[k + 1 :]:
        axx.set_visible(False)

    plt.show()

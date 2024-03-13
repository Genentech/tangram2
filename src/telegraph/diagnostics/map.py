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
        T_n = T_n / T_n.sum(axis=0, keepdims=True)

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

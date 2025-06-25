import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import _utils as ut


def casify(x, ref):
    if x in ref:
        return x
    elif x.lower() in ref:
        return x.lower()
    elif x.upper() in ref:
        return x.upper()
    else:
        raise ValueError("{} was not found".format(x))


@ut.easy_input
def plot_pred_vs_obs(
    X_to,
    X_to_pred,
    S_to,
    feature_names,
    marker_size=0.1,
    log_values=False,
    cmap=plt.cm.magma,
):
    if not isinstance(feature_names, list):
        feature_name = [feature_names]
    n_features = len(feature_names)

    fig, ax = plt.subplots(n_features, 2, figsize=(10.5, 5 * n_features))
    for row, feature_name in enumerate(feature_names):
        if isinstance(X_to, ad.AnnData):
            _feature_name = casify(feature_name, X_to.var_names)
            val_og = X_to.obs_vector(_feature_name)
        else:
            _feature_name = casify(feature_name, X_to.columns)
            val_og = X_to[_feature_name].values
        if isinstance(X_to_pred, ad.AnnData):
            _feature_name = casify(feature_name, X_to_pred.var_names)
            val_pred = X_to_pred.obs_vector(_feature_name)
        else:
            _feature_name = casify(feature_name, X_to_pred.columns)
            val_pred = X_to_pred[_feature_name].values

        if log_values and np.all(val_pred > 0) and np.all(val_og > 0):
            val_pred = np.log1p(val_pred)
            val_og = np.log1p(val_og)

        vmax = np.max((val_pred.max(), val_og.max()))
        vmin = np.min((val_pred.min(), val_og.min()))

        for col, val in enumerate([val_og, val_pred]):
            val_ordr = np.argsort(val)[::-1]

            og_sc = ax[row, col].scatter(
                (
                    S_to.values[val_ordr, 0]
                    if isinstance(S_to, pd.DataFrame)
                    else S_to[val_ordr, 0]
                ),
                (
                    S_to.values[val_ordr, 1]
                    if isinstance(S_to, pd.DataFrame)
                    else S_to[val_ordr, 1]
                ),
                c=val[val_ordr],
                cmap=cmap,
                s=marker_size,
                vmin=vmin,
                vmax=vmax,
            )

            fig.colorbar(og_sc, ax=ax[row, col])

        ax[row, 0].set_title("Observed : {}".format(feature_name))
        ax[row, 1].set_title("Predicted : {}".format(feature_name))

    plt.show()

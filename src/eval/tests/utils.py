import anndata as ad
import numpy as np
import pandas as pd


def make_fake_adata(n_obs, n_features, n_labels, random_seed=42):
    rng = np.random.default_rng(random_seed)

    X = rng.random((n_obs, n_features))
    S = rng.random((n_obs, 2))
    labels = rng.integers(0, n_labels + 1, size=n_obs)

    var_names = ["feature_{}".format(x) for x in range(n_features)]
    obs_names = ["obs_{}".format(x) for x in range(n_obs)]

    adata = ad.AnnData(
        X,
        var=pd.DataFrame(var_names, index=var_names, columns=["features"]),
        obs=pd.DataFrame(labels, index=obs_names, columns=["labels"]),
    )

    adata.obsm["spatial"] = S

    return adata


def make_fake_map_input(
    n_to=10,
    n_from=12,
    n_features_to=15,
    n_features_from=15,
    n_labels_to=5,
    n_labels_from=3,
):
    X_to = make_fake_adata(n_to, n_features_to, n_labels_to)
    X_from = make_fake_adata(n_from, n_features_from, n_labels_from)

    res_dict = dict(X_to=X_to, X_from=X_from)
    return res_dict

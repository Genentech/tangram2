import anndata as ad
import numpy as np
import pandas as pd

from . import utils as ut


class DEA(Enum):
    adj_p_value = "pvals_adj"
    logfold = "log2fc"
    p_value = "pvals"
    feature = "feature"
    score = "score"
    coeff = "score"
    agg_p_value = "pvals_agg"


def get_empty_dea_df():
    return pd.DataFrame(
        [],
        columns=[
            DEA.p_value.value,
            DEA.adj_p_value.value,
            DEA.feature_value.value,
            DEA.score.value,
        ],
    )


def scanpy_dea_labels_from_D(D: pd.DataFrame, group_pair, new_col_name="label"):

    col_names = D.columns.tolist()
    label = np.array(["background"] * len(D), dtype=object)

    if isinstance(group_pair, str):
        grp_1 = group_pair
        grp_2 = None
    elif isinstance(group_pair, (list, tuple)):
        grp_1, grp_2 = group_pair

    def helper(grp_i, labels):
        grp_i_l = ut.listify(grp_i)
        grp_i_fl = [x for x in grp_i_l if x in col_names]
        if len(grp_i_fl) < 1:
            raise ValueError("group covariates are not in design matrix")
        is_grp_i = np.all(D[grp_i_fl].values, axis=1)
        grp_i_name = "_".join(grp_i_fl)
        labels[is_grp_i] = grp_i_name
        return labels, grp_i_name

    label, grp_1_name = helper(grp_1, label)

    if grp_2 is not None:
        label, grp_2_name = helper(grp_2, label)
    else:
        grp_2_name = "rest"

    D_new = pd.DataFrame(label, index=D.index, columns=[new_col_name])

    return D_new, grp_1_name, grp_2_name


def anndata_from_X_and_D(X, D):

    if isinstance(X, ad.AnnData):
        var = X.var
        X = X.to_df()
    else:
        var = pd.DataFrame(
            X.columns,
            index=X.columns,
            columns=["features"],
        )

    adata = ad.AnnData(
        X,
        obs=D,
        var=var,
    )

    return adata

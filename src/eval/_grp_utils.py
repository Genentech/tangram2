from typing import Any, Dict, List, Tuple

import pandas as pd

from . import utils as ut


def add_covariates(func, *args, **kwargs):
    # this is a decorator that can be added to any group method that allows
    # the user to add covariates that already exist in the anndata object
    # and doesn't need any calculation
    def wrapper(
        cls,
        input_dict: Dict[str, Any],
        *args,
        add_covariates: Dict[str, List[str]]
        | Dict[str, str]
        | List[str]
        | str
        | None = None,
        subset: List[str] | str = None,
        merge: bool = False,
        **kwargs,
    ):
        _covs = add_covariates
        ss = ut.listify(subset)

        res_dict = func(cls, input_dict, *args, **kwargs)

        if _covs is None:
            return res_dict

        if not isinstance(_covs, dict):
            cov_dict = ut.listify(_covs)
            cov_dict = {"to": covs, "from": covs}
        else:
            cov_dict = {key: ut.listify(val) for key, val in _covs.items()}

        for tgt, covs in cov_dict.items():
            X = input_dict.get(f"X_{tgt}", None)
            D = res_dict.get(f"D_{tgt}", pd.DataFrame([], index=X.obs.index))
            if X is None:
                continue

            for cov in covs:
                labels = ut.get_ad_value(X, cov, to_np=False)
                if labels is None:
                    pass

                if isinstance(labels.values[0], str):
                    D_add = pd.get_dummies(labels).astype(int)
                    if subset is not None:
                        keep = [c for c in D_add.columns if c in subset]
                        D_add = D_add.loc[:, keep]
                    if not merge:
                        D = pd.concat((D, D_add), axis=1)
                    else:
                        D_new = dict()
                        for name_i in D.columns:
                            col_i = D[name_i].values
                            for name_j in D_add.columns:
                                col_j = D_add[name_j].values
                                col_ij = col_i * col_j
                                D_new[f"{name_i}_{name_j}"] = col_ij
                        D = pd.DataFrame(D_new, index=D.index)
                        del D_new
                else:
                    D[cov] = labels.values

            res_dict[f"D_{tgt}"] = D

        return res_dict

    return wrapper

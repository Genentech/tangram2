from typing import Any, Dict, List, Tuple

import numpy as np
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
        add_covariates: (
            Dict[str, List[str]] | Dict[str, str] | List[str] | str | None
        ) = None,
        subset: List[str] | str = None,
        merge: bool = False,
        **kwargs,
    ):

        # 'add_covariates' is expected to be a dictionary
        # of the form: {'from': [from_key_1,...from_key_k,], 'to' : [to_key_1,...,to_key_2]}
        # where the keys are the names of the columns in the anndata objects

        # rename for convenience, I'm lazy
        _covs = add_covariates

        # execute inner function
        res_dict = func(cls, input_dict, *args, **kwargs)

        # if no covs just return result from the inner function
        if _covs is None:
            return res_dict

        # make sure subset is in list format
        ss = ut.listify(subset)

        # if covariates is not a dict then set same covariates
        # for "to" and "from"
        if not isinstance(_covs, dict):
            covs = ut.listify(_covs)
            cov_dict = {"to": covs, "from": covs}
        else:
            cov_dict = {key: ut.listify(val) for key, val in _covs.items()}

        # add covariates for "to" and "from"
        for tgt, covs in cov_dict.items():
            # get anndata for target ("to" or "from")
            X = input_dict.get(f"X_{tgt}", None)
            # get design matrix for target
            D = res_dict.get(f"D_{tgt}", pd.DataFrame([], index=X.obs.index))
            # if target not in input dict then move on
            if X is None:
                continue

            # for each covariate create indicator
            for cov in covs:
                labels = ut.get_ad_value(X, cov, to_np=False)
                if labels is None:
                    pass

                # if labels are discrete then create indicators
                if isinstance(labels.values[0], str):
                    D_add = pd.get_dummies(labels).astype(int)
                    if subset is not None:
                        keep = [c for c in D_add.columns if c in subset]
                        D_add = D_add.loc[:, keep]
                    # merge is getting all combinations of the old and new covariates
                    if not merge:
                        D = pd.concat((D, D_add), axis=1)
                    else:
                        D_new = dict()
                        # iterate over old covariates
                        for name_i in D.columns:
                            col_i = D[name_i].values
                            # iterate over new covariates
                            for name_j in D_add.columns:
                                col_j = D_add[name_j].values
                                col_ij = col_i * col_j
                                # create merged entry
                                D_new[f"{name_i}_{name_j}"] = col_ij
                        # new design matrix
                        D = pd.DataFrame(D_new, index=D.index)
                        del D_new
                # if continuous covariate just set to values
                else:
                    D[cov] = labels.values

            # overwrite the design matrix
            res_dict[f"D_{tgt}"] = D

        return res_dict

    return wrapper

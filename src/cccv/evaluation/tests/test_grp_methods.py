from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
import pytest

from cccv.evaluation import grp_methods as gm
from cccv.evaluation.tests import utils as ut


class TestThresholdGroup:
    @pytest.fixture(autouse=True)
    def method(
        self,
    ):
        def func(*args, **kwargs):
            return

        return gm.ThresholdGroup

    @pytest.mark.parametrize(
        "thres_t,thres_x,n_covariates",
        [(0.5, 0.5, 0), (0.5, 0.5, 1), (0.5, 0.5, 2), ([0.49, 0.51], [0.51, 0.49], 0)],
    )
    def test_default(
        self,
        method,
        n_covariates,
        thres_t,
        thres_x,
        tmp_path,
        n_to=10,
        n_from=12,
        n_features_to=15,
        n_features_from=15,
    ):
        res_dict = ut.make_fake_X(
            n_to, n_from, n_features_to, n_features_from, n_labels_from=4
        )
        res_dict = ut.make_fake_T(res_dict=res_dict)
        res_dict["X_to_pred"] = res_dict["X_to"].copy()

        n_features_to = res_dict["X_to_pred"].shape[1]
        feature_id = np.random.choice(n_features_to)
        feature_name = res_dict["X_from"].var_names[feature_id]

        if n_covariates > 0:
            label_col = res_dict["X_from"].obs.columns[0]
            add_covariates = {"from": [label_col]}
        else:
            add_covariates = None

        out = method.run(
            res_dict,
            feature_name=feature_name,
            thres_x=thres_x,
            thres_t=thres_t,
            add_covariates=add_covariates,
        )

        res_dict.update(out)

        method.save(res_dict, tmp_path)

        ut.check_out(method, res_dict)

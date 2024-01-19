from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
import pytest

from cccv.evaluation import dea_methods as dm
from cccv.evaluation.tests import utils as ut
from cccv.evaluation.utils import design_matrix_to_labels


class BaseTestDEAMethods:
    def _make_base_input(
        self,
    ):
        n_to = 204
        n_from = 202
        res_dict = ut.make_fake_X(include_pred=True, n_to=n_to, n_from=n_from)
        ut.make_fake_D(res_dict=res_dict, n_grp_from=2, n_grp_to=3)
        return res_dict

    def _method_base(self, method, tmp_path, res_dict=None, method_params=None):

        if method_params is None:
            method_params = {}

        if res_dict is None:
            res_dict = self._make_base_input()

        out = method.run(res_dict, **method_params)
        res_dict.update(out)

        method.save(res_dict, tmp_path)


class TestScanpyDEA(BaseTestDEAMethods):
    @pytest.fixture(autouse=True)
    def method(
        self,
    ):
        return dm.ScanpyDEA

    def test_default(self, method, tmp_path):
        self._method_base(method, tmp_path)

    @pytest.mark.parametrize(
        "target,n_groups,test_method,normalize,mode,do_subset",
        [
            ("from", 0, "wilcoxon", False, "pos", False),
            ("to", 0, "wilcoxon", False, "pos", False),
            ("from", 2, "wilcoxon", False, "pos", False),
            ("from", 0, "wilcoxon", False, "neg", False),
            ("from", 0, "t-test", False, "neg", False),
            ("to", 0, "wilcoxon", True, "both", True),
        ],
    )
    def test_custom(
        self,
        method,
        tmp_path,
        test_method,
        normalize,
        n_groups,
        do_subset,
        mode,
        target,
    ):
        res_dict = self._make_base_input()

        no_target = {"to": "from", "from": "to"}[target]
        res_dict[f"X_{no_target}"] = None
        res_dict[f"X_{no_target}_pred"] = None
        res_dict[f"D_{no_target}"] = None

        D = res_dict.get(f"D_{target}").copy()
        if do_subset:
            keep_cols = D.columns[1::].tolist()
            D = D.loc[:, keep_cols]
            subset_features = {target: keep_cols}
        else:
            subset_features = None

        if n_groups > 0:
            labels = design_matrix_to_labels(D)
            uni_labels = np.unique(labels)
            groups = [
                tuple(np.random.choice(uni_labels, replace=False, size=2))
                for _ in range(n_groups)
            ]
            groups = {target: groups}
        else:
            groups = "all"

        method_params = dict(
            method=test_method,
            normalize=normalize,
            groups=groups,
            subset_features=subset_features,
            mode=mode,
        )

        self._method_base(
            method, tmp_path, res_dict=res_dict, method_params=method_params
        )

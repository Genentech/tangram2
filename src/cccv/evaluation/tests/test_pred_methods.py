import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pytest
from cccv.evaluation import pred_methods as pm
from cccv.evaluation.tests import utils as ut


class BaseTestPredMethods:
    def _method_base(self, method, tmp_path, res_dict=None, **method_params):
        if res_dict is None:
            res_dict = ut.make_fake_map_input()
            res_dict = ut.make_fake_T(res_dict=res_dict)
            # TODO: Add X_from_scaled to fake data

        out = method.run(res_dict, **method_params)
        res_dict.update(out)

        assert all([x in res_dict for x in method.outs])

        if "X_to_pred" in res_dict:
            X_to_pred_row, X_to_pred_col = res_dict["X_to_pred"].shape
            assert (X_to_pred_row == res_dict["X_to"].shape[0]) and (
                    X_to_pred_col == res_dict["X_from"].shape[1]
            )
        if "X_from_pred" in res_dict:
            assert res_dict["X_from_pred"] is None

        method.save(res_dict, tmp_path)


class TestTangramPred(BaseTestPredMethods):
    @pytest.fixture(autouse=True)
    def method(
        self,
    ):
        return pm.TangramPred

    def test_default(self, method, tmp_path):
        # test that structure and dims of output is correct
        # checks default params case

        self._method_base(method, tmp_path)


class TestMoscotPred(BaseTestPredMethods):
    @pytest.fixture(autouse=True)
    def method(
        self,
    ):
        return pm.MoscotPred

    def test_default(self, method, tmp_path):
        # test that structure and dims of output is correct
        # checks default params case

        self._method_base(method, tmp_path)

    @pytest.mark.parametrize(
        "prediction_genes",
        [["feature_2"], ["feature_1", "feature_4"]],
    )
    def test_custom(self, method, tmp_path, prediction_genes):
        # test that structure and dims of output is correct
        # checks custom params case

        self._method_base(method, tmp_path, prediction_genes=prediction_genes)


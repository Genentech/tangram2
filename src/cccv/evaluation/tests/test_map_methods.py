from abc import ABC, abstractmethod

import anndata as ad
import numpy as np
import pytest

from cccv.evaluation import map_methods as mm
from cccv.evaluation.tests import utils as ut


class BaseTestMapMethods:
    def _method_base(self, method, tmp_path, res_dict=None, **method_params):
        if res_dict is None:
            res_dict = ut.make_fake_map_input()

        out = method.run(res_dict, **method_params)
        res_dict.update(out)

        assert all([x in res_dict for x in method.outs])

        if "T" in res_dict:
            T_row, T_col = res_dict["T"].shape
            assert (T_row == res_dict["X_to"].shape[0]) and (
                T_col == res_dict["X_from"].shape[0]
            )

        for suffix in ["to", "from"]:
            S = res_dict.get("S_{}".format(suffix), None)
            if S is not None:
                S_row = res_dict["S_{}".format(suffix)].shape[0]
                assert S_row == res_dict["X_{}".format(suffix)].shape[0]

        method.save(res_dict, tmp_path)


class TestRandomMap(BaseTestMapMethods):
    @pytest.fixture(autouse=True)
    def method(
        self,
    ):
        return mm.RandomMap

    def test_default(self, method, tmp_path):
        # test that structure and dims of output is correct
        # checks default params case

        self._method_base(method, tmp_path)

    @pytest.mark.parametrize(
        "seed,return_sparse",
        [(3, False), (1, True)],
    )
    def test_custom(self, method, tmp_path, seed, return_sparse):
        # test that structure and dims of output is correct
        # checks custom params case

        self._method_base(method, tmp_path, seed=seed, return_sparse=return_sparse)


class TestArgMaxCorrMap(BaseTestMapMethods):
    @pytest.fixture(autouse=True)
    def method(
        self,
    ):
        return mm.ArgMaxCorrMap

    def test_structure_default(self, method, tmp_path):
        # test that structure and dims of output is correct
        # checks default params case
        self._method_base(method, tmp_path)

    @pytest.mark.parametrize(
        "return_sparse",
        [(False), (True)],
    )
    def test_structure_custom(self, method, tmp_path, return_sparse):
        # test that structure and dims of output is correct
        # checks custom params case

        self._method_base(method, tmp_path, return_sparse=return_sparse)

    def test_method_output(self, method):
        # tests that the output makes sense
        # if X_to and X_from are same matrix
        # mapping by ArgMaxCorrMap shoul give 1-1
        # mapping

        res_dict = ut.make_fake_map_input()
        res_dict["X_to"] = res_dict["X_from"]
        out = method.run(res_dict, return_sparse=True)
        np.testing.assert_array_equal(out["T"].row, out["T"].col)

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import anndata as ad
import numpy as np
import pytest

import cccv.evaluation.constants as C
from cccv.evaluation import metrics as mx
from cccv.evaluation.tests import utils as ut


class BaseTestMetric:
    @classmethod
    @abstractmethod
    def _make_input(*args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    def _metric_base(
        cls, metric, tmp_path, res_dict=None, ref_dict=None, **method_params
    ):

        match (res_dict, ref_dict):
            case (None, None):
                res_dict, ref_dict = ut.make_input()
            case (_, None):
                _, ref_dict = ut.make_input()
            case (None, _):
                res_dict, _ = ut.make_input()

        score_dict = metric.score(res_dict, **method_params)

        metric.save(score_dict, tmp_path)


class TestMapMetric(BaseTestMetric):
    @classmethod
    def _make_input(return_sparse=False):
        res_dict = ut.make_fake_T(return_sparse=return_sparse)

        ut.make_fake_S(res_dict)

        ref_dict = copy.deepcopy(res_dict)
        return (res_dict, ref_dict)


class TestMapMetric(BaseTestMetric):
    @pytest.mark.parametrize(
        "metric,return_sparse,",
        ["map_jaccard", True],
        ["map_jaccard", False],
        ["map_accuracy", True],
        ["map_rmse", False],
    )
    def test_default(
        cls,
        metric,
        return_spare,
        tmp_path,
        **kwargs,
    ):

        _metric_base(metric, tmp_path)

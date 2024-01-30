import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import anndata as ad
import numpy as np
import pytest

import telegraph.evaluation.constants as C
from telegraph.evaluation.tests import utils as ut


class BaseTestMetric(ABC):
    @classmethod
    @abstractmethod
    def _make_input(cls, *args, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    @classmethod
    def _metric_base(
        cls,
        metric,
        tmp_path,
        res_dict=None,
        ref_dict=None,
        **kwargs,
    ):
        match (res_dict, ref_dict):
            case (None, None):
                res_dict, ref_dict = cls._make_input()
            case (_, None):
                _, ref_dict = cls._make_input()
            case (None, _):
                res_dict, _ = cls._make_input()

        score_dict = metric.score(res_dict=res_dict, ref_dict=ref_dict, **kwargs)

        metric.save(score_dict, tmp_path)

    def test_default(*args, **kwargs):
        pass


class TestMapMetric(BaseTestMetric):
    @classmethod
    def _make_input(cls, return_sparse=False):
        res_dict = ut.make_fake_T(return_sparse=return_sparse)

        ut.make_fake_S(res_dict=res_dict)

        ref_dict = copy.deepcopy(res_dict)

        return (res_dict, ref_dict)

    @pytest.mark.parametrize(
        "metric_name,return_sparse",
        (
            ["map_jaccard", True],
            ["map_jaccard", False],
            ["map_accuracy", True],
            ["map_rmse", False],
        ),
    )
    def test_run(
        cls,
        metric_name,
        return_sparse,
        tmp_path,
        **kwargs,
    ):
        # tests for runtime errors

        metric = C.METRICS["OPTIONS"].value[metric_name]

        ref_dict, res_dict = cls._make_input(return_sparse)

        cls._metric_base(
            metric,
            tmp_path,
            res_dict=res_dict,
            ref_dict=ref_dict,
        )

    def test_output(*args, **kwargs):
        # asserts that the output is as expected
        pass

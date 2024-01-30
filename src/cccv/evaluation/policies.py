from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List

import numpy as np
from anndata import AnnData
from pandas import DataFrame
from scipy.sparse import spmatrix


def dim_match(shape_1, shape_2):
    for d1, d2 in zip(shape_1, shape_2):
        match (d1, d2):
            case (None, _):
                pass
            case (_, None):
                pass
            case (_, _):
                if d1 != d2:
                    return False
    return True


class BasePolicy(ABC):
    object_name = ""
    object_type = None

    @classmethod
    def test_dimensions(
        cls, obj, exp_shape, obj_shape: List[int] | None = None, **kwargs
    ) -> None:

        if obj_shape is None:
            assert hasattr(
                obj, "shape"
            ), f"{cls.object_name} does not have a shape attribute, provide shape manually"
            obj_shape = obj.shape
        if not isinstance(obj_shape, (list, tuple, np.ndarray)):
            obj_shape = [obj_shape]
        if not isinstance(exp_shape, (list, tuple, np.ndarray)):
            exp_shape = [exp_shape]

        # check some number of dimensions
        assert len(exp_shape) == len(
            obj_shape
        ), f"number of dimensions mismatch for {cls.object_name}"

        # check same dims values
        assert dim_match(
            exp_shape, obj_shape
        ), f"{cls.object_name} does not have the correct dimensions"

    @classmethod
    def test_type(cls, obj: Any, exp_type: Any | None = None):
        obj_type = type(obj)
        if exp_type is None:
            exp_type = cls.object_type

        assert isinstance(
            obj, exp_type
        ), f"{cls.object_name} is of the wrong type. Expected {str(exp_type)}, got {str(obj_type)}"

    @classmethod
    @abstractmethod
    def test_values(cls, obj, *args, **kwargs):
        pass


class PolicyT(BasePolicy):
    object_name = "T"
    object_type = (np.ndarray, spmatrix)

    @classmethod
    def test_values(cls, obj: np.ndarray | spmatrix):

        is_neg = np.sum(obj < 0)
        assert is_neg == 0, f"{cls.object_name} has negative values"


class PolicyX(BasePolicy):
    object_name = "X"

    @classmethod
    def test_values(cls, *args, **kwargs):
        pass


class PolicyXto(PolicyX):
    object_name = "X_to"
    object_type = (DataFrame, AnnData)


class PolicyXtopred(PolicyX):
    object_name = "X_to_pred"
    object_type = (DataFrame, AnnData)


class PolicyXfrom(PolicyX):
    object_name = "X_from"
    object_type = (DataFrame, AnnData)


class PolicyD(BasePolicy):
    object_name = "D"
    object_type = DataFrame

    @classmethod
    def test_values(cls, *args, **kwargs):
        pass


class PolicyDto(PolicyD):
    object_name = "D_to"


class PolicyDfrom(PolicyD):
    object_name = "D_to"


class PolicyDict(Enum):
    T = PolicyT
    X_to = PolicyXto
    X_from = PolicyXfrom
    D_to = PolicyDto
    D_from = PolicyDfrom
    X_to_pred = PolicyXtopred


def check_values(obj, obj_name: str, *args, **kwargs):
    PolicyDict[obj_name].value.test_values(obj, *args, **kwargs)


def check_dimensions(obj, obj_name: str, *args, **kwargs):
    PolicyDict[obj_name].value.test_dimensions(obj, *args, **kwargs)


def check_type(obj, obj_name: str, *args, **kwargs):
    PolicyDict[obj_name].value.test_type(obj, *args, **kwargs)

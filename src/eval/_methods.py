from abc import ABC, abstractmethod
from typing import Any, Dict


class MethodClass(ABC):
    # Method Baseclass
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pass

    @classmethod
    @abstractmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        # run methods runs the actual method
        # this should return a dictionary with
        # the name of the object as a key and the object as value
        pass

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        **kwargs,
    ) -> None:
        # save methods takes the output from run methods
        # or a dictionary which the run method output is a subset of.
        # the save method should save the output as file
        return None

    @staticmethod
    def get_kwargs(*args, **kwargs):
        # TODO: discontinue this
        return dict()

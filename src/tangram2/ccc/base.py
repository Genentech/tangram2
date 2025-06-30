from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List



class MethodClass(ABC):
    # Method Baseclass
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        # elements that are required as input
        ins = []
        # elements that are given as output
        outs = []

    @classmethod
    @property
    def custom_save_funcs(cls) -> Dict[str, Callable]:
        # update this method if you want custom
        # save functions
        _funcs = dict()
        return _funcs

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

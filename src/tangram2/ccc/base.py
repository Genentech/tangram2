from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List


class MethodClass(ABC):
    """ """

    # Method Baseclass
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the method class."""
        # elements that are required as input
        ins = []
        # elements that are given as output
        outs = []

    @classmethod
    @property
    def custom_save_funcs(cls) -> Dict[str, Callable]:
        """Return a dictionary of custom save functions."""
        _funcs = dict()
        return _funcs

    @classmethod
    @abstractmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the method"""
        pass

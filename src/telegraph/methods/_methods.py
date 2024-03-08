from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from telegraph.methods.save_methods import StandardSaveMethods

from . import utils as ut


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

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        save_keys: List[str] | str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        if save_keys is not None:
            _save_keys = ut.listify(save_keys)
            _save_keys = [x for x in _save_key if res_dict.get(x) is not None]
        else:
            _save_keys = cls.outs

        save_funcs = {
            **cls.custom_save_funcs,
            **StandardSaveMethods.standard_save_funcs,
        }

        for key in _save_keys:
            if key in save_funcs:
                if verbose:
                    print(f"Saving : {key}")
                save_funcs[key](res_dict, out_dir, **kwargs)

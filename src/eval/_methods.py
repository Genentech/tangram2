from abc import ABC, abstractmethod
from typing import Any, Dict


class MethodClass(ABC):
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
        pass

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        **kwargs,
    ) -> None:
        return None

    @staticmethod
    def get_kwargs(*args, **kwargs):
        return dict()

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
    ) -> Any:
        pass

    @staticmethod
    def get_kwargs(*args, **kwargs):
        return dict()

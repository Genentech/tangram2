from abc import ABC, abstractmethod
from typing import Any


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
        self,
        *args,
        **kwargs,
    ) -> Any:
        pass

    @staticmethod
    def get_kwargs(*args, **kwargs):
        return dict()

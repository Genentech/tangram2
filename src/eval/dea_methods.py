from . import utils as ut
from eval._methods import MethodClass
from typing import Any, Dict


class DEAMethod(MethodClass):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    @classmethod
    @abstractmethod
    def run(
        cls,
            Y: Any,
        design_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        pass



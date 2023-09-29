from . import utils as ut
from eval._methods import MethodClass
from scipy.sparse import spmatrix



def PredMethodClass(MethodClass):
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
        X_to: Any,
        X_from: Any,
        T_from: np.ndarray | spmatrix,
        *args,
        S_to: np.ndarray | None = None,
        S_from: np.ndarray | None = None,
        **kwargs,
    ) -> Dict[str, np.ndarray] | Dict[str, spmatrix]:
        pass


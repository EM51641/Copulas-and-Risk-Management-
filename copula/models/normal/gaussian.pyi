import numpy as np
from numpy.typing import NDArray

class GaussianCopula:
    initial_weights: NDArray[np.float64]
    returns: NDArray[np.float64]
    size: int
    var: float
    cvar: float
    alpha: float

    def __init__(
        self,
        initial_weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        size: int = 10000,
        alpha: float = 0.01,
    ) -> None: ...
    def fit(self) -> None: ...

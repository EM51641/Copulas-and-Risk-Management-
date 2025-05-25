from typing import Any

import numpy as np
from numpy.typing import NDArray

class StudentTCopula:
    var: float
    cvar: float
    alpha: float
    df: int
    size: int
    initial_weights: NDArray[np.float64]
    returns: NDArray[np.float64]

    def __init__(
        self,
        initial_weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        size: int = 10000,
        df: int = 4,
        alpha: float = 0.01,
    ) -> None: ...
    def fit(self) -> None: ...
    def _empirical_quantile(
        self, data: NDArray[np.float64], quantiles: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

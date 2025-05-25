from typing import Any

import numpy as np
from numpy.typing import NDArray

class StudentTCopula:
    var: float
    cvar: float

    def __init__(
        self,
        initial_weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        size: int = 10000,
    ) -> None: ...
    def fit(self) -> None: ...
    def _empirical_quantile(
        self, data: NDArray[np.float64], quantiles: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

from typing import Any

import numpy as np
from numpy.typing import NDArray

class StudentTCopula:
    """
    Student's t Copula model for portfolio risk management.

    This class implements the Student's t Copula model for portfolio risk management.
    It is used to calculate the Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    of a portfolio.
    """

    var: float
    cvar: float
    alpha: float
    df: int
    size: int
    weights: NDArray[np.float64]
    returns: NDArray[np.float64]

    def __init__(
        self,
        weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        size: int = 10000,
        df: int = 4,
        alpha: float = 0.01,
    ) -> None:
        """
        Initialize the StudentTCopula model.

        Args:
            weights: The weights of the portfolio.
            returns: The returns of the portfolio.
            size: The number of samples to generate.
            df: The degrees of freedom for the Student's t distribution.
            alpha: The significance level for the VaR and CVaR calculations.
        """
        ...

    def fit(self) -> None:
        """
        Fit the StudentTCopula model.

        This method fits the StudentTCopula model to the data.
        It calculates the Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        of the portfolio.
        """
        ...

    def __str__(self) -> str:
        """Return a string representation of the StudentTCopula."""
        ...

    def __repr__(self) -> str:
        """Return a detailed string representation of the StudentTCopula."""
        ...

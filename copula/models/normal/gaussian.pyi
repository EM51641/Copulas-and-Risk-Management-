import numpy as np
from numpy.typing import NDArray

class GaussianCopula:
    """
    Gaussian Copula model for portfolio risk management.

    This class implements the Gaussian Copula model for portfolio risk management.
    It is used to calculate the Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    of a portfolio.

    """

    weights: NDArray[np.float64]
    returns: NDArray[np.float64]
    size: int
    var: float
    cvar: float
    alpha: float

    def __init__(
        self,
        weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        size: int = 10000,
        alpha: float = 0.01,
    ) -> None:
        """
        Initialize the GaussianCopula model.

        Args:
            weights: The weights of the portfolio.
            returns: The returns of the portfolio.
            size: The number of samples to generate.
            alpha: The significance level for the VaR and CVaR calculations.
        """
        ...

    def fit(self) -> None:
        """
        Fit the GaussianCopula model.

        This method fits the GaussianCopula model to the data.
        It calculates the Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        of the portfolio.
        """
        ...

    def __str__(self) -> str:
        """Return a string representation of the GaussianCopula."""
        ...

    def __repr__(self) -> str:
        """Return a detailed string representation of the GaussianCopula."""
        ...

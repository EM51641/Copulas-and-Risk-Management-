import numpy as np
import pyvinecopulib as pv
from numpy.typing import NDArray


class ClaytonCopula:
    """
    Clayton Copula implementation.

    The Clayton copula is an Archimedean copula that exhibits asymmetric tail dependence,
    with stronger lower tail dependence than upper tail dependence. It is particularly
    useful for modeling scenarios where extreme negative events tend to occur together
    more frequently than extreme positive events.

    Properties:
        - Lower tail dependent
        - Upper tail independent
        - Asymmetric dependence structure
        - Parameter θ > 0 controls the strength of dependence
        - Perfect positive dependence as θ → ∞
        - Independence as θ → 0

    The Clayton copula is widely used in financial risk management, insurance,
    and other fields where modeling lower tail dependence is crucial.
    """

    def __init__(
        self,
        weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        size: int = 10000,
        alpha: float = 0.01,
    ) -> None:

        assert weights.ndim == 1, "weights must be a 1D array"
        assert returns.ndim == 2, "returns must be a 2D array"
        assert (
            weights.shape[0] == returns.shape[1]
        ), "weights and returns must have the same number of assets"
        assert size > 0, "size must be greater than 0"
        assert alpha > 0 and alpha < 1, "alpha must be between 0 and 1"

        self.weights = weights
        self.returns = returns
        self.size = size
        self.var = 0.0
        self.cvar = 0.0
        self.alpha = alpha

    def fit(self) -> None:
        """
        Fits the Clayton copula to the return data and calculates VaR/CVaR.

        This method performs the following steps:
        1. Transforms the return data into pseudo-observations using empirical distribution
        2. Fits a Clayton copula to capture the dependence structure
        3. Generates uniform samples with Clayton's dependence
        4. Transforms back to original returns using empirical quantiles
        5. Calculates VaR and CVaR from the rearranged returns
        """
        # Transform copula data using the empirical distribution
        U_empirical = pv.to_pseudo_obs(self.returns)

        # Fit a Clayton copula to get the dependence parameter
        controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.clayton])  # type: ignore
        cop = pv.Vinecop.from_data(U_empirical, controls=controls)

        # Generate uniform samples with Clayton's dependence
        U: NDArray[np.float64] = cop.simulate(self.size)  # type: ignore

        # Transform back to original returns using empirical quantiles
        U_scaled = np.empty_like(U)
        for i in range(self.returns.shape[1]):
            U_scaled[:, i] = np.quantile(self.returns[:, i], U[:, i])

        # Calculate portfolio returns
        portfolio_returns = U_scaled.dot(self.weights)

        # Calculate risk metrics
        negative_returns = portfolio_returns[portfolio_returns < 0]
        if negative_returns.size == 0:
            self.var = 0.0
            self.cvar = 0.0
            return

        self.var = float(np.quantile(negative_returns, self.alpha))
        self.cvar = float(negative_returns[negative_returns <= self.var].mean())

    def __str__(self) -> str:
        return f"ClaytonCopula(size={self.size}, alpha={self.alpha})"

    def __repr__(self) -> str:
        return f"ClaytonCopula(size={self.size}, alpha={self.alpha})"

import numpy as np
import scipy
from numpy.typing import NDArray


class GaussianCopula:
    def __init__(
        self,
        initial_weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        size: int = 10000,
    ):
        self.initial_weights = initial_weights
        self.returns = returns
        self.size = size

    def fit(self):
        covariance_matrix = np.cov(self.returns)

        simulated_returns = np.random.multivariate_normal(
            self.returns.mean(), covariance_matrix, self.size
        )

        correlation_matrix = np.corrcoef(simulated_returns)

        lower_triangular_matrix = np.linalg.cholesky(correlation_matrix)

        quantiles = np.zeros(len(self.returns))

        for i in range(len(self.returns)):
            z = np.random.standard_normal(self.size)
            g = np.dot(lower_triangular_matrix, z)
            quantiles[i] = scipy.stats.norm.cdf(g)

        inverse_transformation = np.array(
            [
                np.quantile(self.returns[:, i], quantiles[i])
                for i in range(self.returns.shape[1])
            ]
        )

        forecasted_returns = np.dot(
            np.array(self.initial_weights), inverse_transformation.T
        )
        negative_returns = forecasted_returns[forecasted_returns < 0]

        value_at_risk = np.quantile(negative_returns, 0.01)
        conditional_value_at_risk = np.mean(
            negative_returns[negative_returns < value_at_risk]
        )

        self.var = value_at_risk
        self.cvar = conditional_value_at_risk

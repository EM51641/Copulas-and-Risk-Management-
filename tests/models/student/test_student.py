import numpy as np
import pytest

from copula.models.student.student import StudentTCopula


@pytest.fixture
def sample_data():
    # Generate sample returns data
    np.random.seed(42)
    n_assets = 3
    n_observations = 1000
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets),
        cov=np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]),
        size=n_observations,
    )
    weights = np.array([0.4, 0.3, 0.3])
    return returns, weights


def test_initialization(sample_data):
    returns, weights = sample_data
    copula = StudentTCopula(
        initial_weights=weights, returns=returns, size=10000, df=4, alpha=0.01
    )

    assert copula.initial_weights.shape == weights.shape
    assert copula.returns.shape == returns.shape
    assert copula.size == 10000
    assert copula.df == 4
    assert copula.alpha == 0.01
    assert copula.var == 0.0  # Initial value
    assert copula.cvar == 0.0  # Initial value


def test_fit(sample_data):
    returns, weights = sample_data
    copula = StudentTCopula(
        initial_weights=weights, returns=returns, size=10000, df=4, alpha=0.01
    )

    copula.fit()

    # Check that VaR and CVaR are calculated
    assert copula.var != 0.0
    assert copula.cvar != 0.0
    # VaR should be less than or equal to CVaR
    assert copula.var > copula.cvar
    # Both should be negative (since we're looking at losses)
    assert copula.var < 0
    assert copula.cvar < 0


def test_different_degrees_of_freedom(sample_data):
    returns, weights = sample_data
    df_values = [3, 4, 5, 10]
    var_results = []
    cvar_results = []

    for df in df_values:
        copula = StudentTCopula(
            initial_weights=weights, returns=returns, size=10000, df=df, alpha=0.01
        )
        copula.fit()
        var_results.append(copula.var)
        cvar_results.append(copula.cvar)

    # Check that results are different for different df values
    assert len(set(var_results)) == len(df_values)
    assert len(set(cvar_results)) == len(df_values)


def test_different_alpha_values(sample_data):
    returns, weights = sample_data
    alpha_values = [0.01, 0.05, 0.1]
    var_results = []
    cvar_results = []

    for alpha in alpha_values:
        copula = StudentTCopula(
            initial_weights=weights, returns=returns, size=10000, df=4, alpha=alpha
        )
        copula.fit()
        var_results.append(copula.var)
        cvar_results.append(copula.cvar)

    # Check that VaR becomes less negative as alpha increases
    assert np.all(np.diff(var_results) > 0)
    # Check that CVaR follows the same pattern
    assert np.all(np.diff(cvar_results) > 0)


def test_input_validation(sample_data):
    returns, weights = sample_data

    # Test invalid weights shape
    with pytest.raises(AssertionError):
        StudentTCopula(
            initial_weights=np.array([0.5, 0.5]),  # Wrong number of weights
            returns=returns,
            size=10000,
            df=4,
        )

    # Test invalid returns shape
    with pytest.raises(AssertionError):
        StudentTCopula(
            initial_weights=weights,
            returns=returns[:, :2],  # Wrong number of assets
            size=10000,
            df=4,
        )

    # Test invalid df value
    with pytest.raises(AssertionError):
        StudentTCopula(
            initial_weights=weights, returns=returns, size=10000, df=1  # df must be > 2
        )

    # Test invalid alpha value
    with pytest.raises(AssertionError):
        StudentTCopula(
            initial_weights=weights,
            returns=returns,
            size=10000,
            alpha=0.0,
        )

# Copulas and Risk Management

A high-performance Python library for modeling financial risk using copula-based approaches. This library implements various copula models for portfolio risk assessment, with a focus on Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations.

## Features

- **Multiple Copula Models**:
  - Gaussian Copula
  - Student's t Copula
  - Clayton Copula
- **High-Performance Implementation**:
  - Cython-optimized core computations
  - Efficient memory management
  - Type-safe operations
- **Risk Metrics**:
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
  - Customizable confidence levels
- **Comprehensive Testing**:
  - Unit tests for all models
  - Input validation

## Installation

```bash
# Clone the repository
git clone https://github.com/EM51641/Copulas-and-Risk-Management-.git
cd Copulas-and-Risk-Management-

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Or install with all development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from copula.models.normal.gaussian import GaussianCopula
from copula.models.student.student import StudentTCopula
from copula.models.clayton.clayton import ClaytonCopula

# Generate sample returns data
n_assets = 3
n_observations = 1000
returns = np.random.multivariate_normal(
    mean=np.zeros(n_assets),
    cov=np.array([[1.0, 0.5, 0.3],
                 [0.5, 1.0, 0.4],
                 [0.3, 0.4, 1.0]]),
    size=n_observations
)
weights = np.array([0.4, 0.3, 0.3])

# Gaussian Copula
gaussian_copula = GaussianCopula(
    weights=weights,
    returns=returns,
    size=10000,
    alpha=0.01
)
gaussian_copula.fit()
print(f"Gaussian VaR: {gaussian_copula.var}")
print(f"Gaussian CVaR: {gaussian_copula.cvar}")

# Student's t Copula
student_copula = StudentTCopula(
    weights=weights,
    returns=returns,
    size=10000,
    df=4,
    alpha=0.01
)
student_copula.fit()
print(f"Student's t VaR: {student_copula.var}")
print(f"Student's t CVaR: {student_copula.cvar}")

# Clayton Copula
clayton_copula = ClaytonCopula(
    weights=weights,
    returns=returns,
    size=10000,
    alpha=0.01
)
clayton_copula.fit()
print(f"Clayton VaR: {clayton_copula.var}")
print(f"Clayton CVaR: {clayton_copula.cvar}")
```

## Key Features of Each Copula

### Gaussian Copula
- Symmetric tail dependence
- Based on multivariate normal distribution
- Suitable for moderate dependence structures

### Student's t Copula
- Heavy tails
- Symmetric tail dependence
- Controlled by degrees of freedom parameter
- Better for extreme events

### Clayton Copula
- Asymmetric tail dependence
- Strong lower tail dependence
- Upper tail independence
- Good for modeling joint extreme losses

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

**IMPORTANT: This software is for educational and research purposes only. It is not intended to provide investment advice.**

- This library is provided "as is" without any warranties
- The authors are not responsible for any financial losses or damages
- Users should consult with qualified financial professionals before making any investment decisions
- The risk models implemented are simplified versions of real-world scenarios
- Past performance is not indicative of future results
- All investment decisions should be made at the user's own risk

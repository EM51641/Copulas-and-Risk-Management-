# Copulas and Risk Management

A Python library for modeling financial risk using copula-based approaches. This library implements various copula models for portfolio risk assessment, with a focus on Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations.

## Features

- Implementation of different copula models:
  - Gaussian Copula
  - Student's t Copula
- Risk metrics calculation:
  - Value at Risk (VaR)
  - Conditional Value at Risk (CVaR)
- High-performance implementation using Cython
- Comprehensive test suite

## Installation

```bash
uv pip install .
```

## Usage

```python
import numpy as np
from copula.models.normal.gaussian import GaussianCopula
from copula.models.student.student import StudentTCopula

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
    initial_weights=weights,
    returns=returns,
    size=10000,
    alpha=0.01
)
gaussian_copula.fit()
print(f"Gaussian VaR: {gaussian_copula.var}")
print(f"Gaussian CVaR: {gaussian_copula.cvar}")

# Student's t Copula
student_copula = StudentTCopula(
    initial_weights=weights,
    returns=returns,
    size=10000,
    df=4,
    alpha=0.01
)
student_copula.fit()
print(f"Student's t VaR: {student_copula.var}")
print(f"Student's t CVaR: {student_copula.cvar}")
```

## Project Structure

```
copula/
├── models/
│   ├── normal/
│   │   ├── gaussian.pyx    # Gaussian Copula implementation
│   │   └── gaussian.pyi    # Type hints for Gaussian Copula
│   └── student/
│       ├── student.pyx     # Student's t Copula implementation
│       └── student.pyi     # Type hints for Student's t Copula
└── tests/
    └── models/
        ├── normal/
        │   └── test_gauss.py    # Tests for Gaussian Copula
        └── student/
            └── test_student.py  # Tests for Student's t Copula
```

## Development

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Cython
- pytest (for testing)

### Running Tests

```bash
pytest tests/
```

### Building Cython Extensions

```bash
python setup.py build_ext --inplace
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

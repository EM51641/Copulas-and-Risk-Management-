# gaussian_copula.pyx
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt
from scipy.stats import norm

ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class GaussianCopula:
    cdef:
        cnp.ndarray initial_weights
        cnp.ndarray returns
        int size
        public double var
        public double cvar
        public double alpha
    
    def __init__(self, cnp.ndarray[DTYPE_t, ndim=1] initial_weights, 
                 cnp.ndarray[DTYPE_t, ndim=2] returns, 
                 int size=10000,
                 double alpha=0.01):
        self.initial_weights = np.ascontiguousarray(initial_weights, dtype=np.float64)
        self.returns = np.ascontiguousarray(returns, dtype=np.float64)
        self.size = size
        self.var = 0.0
        self.cvar = 0.0
        self.alpha = alpha

    cpdef fit(self):
        cdef:
            int n_assets = self.returns.shape[1]
            int i, j
            cnp.ndarray[DTYPE_t, ndim=2] corr_matrix
            cnp.ndarray[DTYPE_t, ndim=2] cholesky_matrix
            cnp.ndarray[DTYPE_t, ndim=2] Z
            cnp.ndarray[DTYPE_t, ndim=2] X
            cnp.ndarray[DTYPE_t, ndim=2] U
            cnp.ndarray[DTYPE_t, ndim=2] U_scaled
            cnp.ndarray[DTYPE_t, ndim=1] portfolio_returns
            cnp.ndarray[DTYPE_t, ndim=1] negative_returns
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(self.returns, rowvar=False)
        cholesky_matrix = np.linalg.cholesky(corr_matrix)

        # Generate multivariate normal samples
        Z = np.random.multivariate_normal(
            np.zeros(n_assets), 
            np.eye(n_assets), 
            self.size
        )
        X = Z.dot(cholesky_matrix.T)
        
        # Transform to uniform using normal CDF
        U = norm.cdf(X)
        
        # Inverse transform using empirical quantiles
        U_scaled = np.empty_like(U)
        for i in range(n_assets):
            U_scaled[:, i] = self._empirical_quantile(
                self.returns[:, i], 
                U[:, i]
            )

        # Calculate portfolio returns
        portfolio_returns = U_scaled.dot(self.initial_weights)
        
        # Calculate risk metrics
        negative_returns = portfolio_returns[portfolio_returns < 0]
        if negative_returns.size == 0:
            self.var = 0.0
            self.cvar = 0.0
            return

        self.var = np.quantile(negative_returns, self.alpha)
        self.cvar = negative_returns[negative_returns <= self.var].mean()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cnp.ndarray[DTYPE_t, ndim=1] _empirical_quantile(self, 
                                                         cnp.ndarray[DTYPE_t, ndim=1] data, 
                                                         cnp.ndarray[DTYPE_t, ndim=1] quantiles):
        cdef:
            cnp.ndarray[DTYPE_t, ndim=1] sorted_data = np.sort(data)
            cnp.ndarray[DTYPE_t, ndim=1] result = np.empty_like(quantiles)
            int n = data.shape[0]
            int idx

        for i in range(quantiles.shape[0]):
            idx = min(int(quantiles[i] * n), n-1)
            result[i] = sorted_data[idx]
        return result

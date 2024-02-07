from math import ceil

import numpy as np

class BackwardSDE:
    """This class is used to represent a Partial Derivative Equation as a Backward Stochastic Differential Equation (BSDE).
    """

    def __init__(self, X0, delta_t, T) -> None:
        self.dim = X0.shape[0]
        self.N = int(ceil(T / delta_t))

        self.delta_t = delta_t
        self.time = np.linspace(0, T, self.N + 1)

    def b(self, x, t):
        pass

    def sigma(self, x, t):
        pass

    def h(self, x, t, y, z):
        pass

    def g(self, x):
        pass


class BlackScholes(BackwardSDE):

        def __init__(self, X0, delta_t, T, r, sigma, S0) -> None:
            super().__init__(X0, delta_t, T)

            self.r = r
            self.sigma_ = sigma
            self.S0 = S0

        def b(self, x, t):
            return 0

        def sigma(self, x, t):
            return self.sigma_ * x

        def h(self, x, t, y, z):
            return -self.r * (y - z.T @ x)

        def g(self, x):
            return np.linalg.norm(x, axis=1) ** 2

class MultiAssetGaussian(BackwardSDE):

    def __init__(self, X0, delta_t, T, r, sigma, rho, mu, S0) -> None:
        super().__init__(X0, delta_t, T)

        self.r = r
        self.sigma = sigma
        self.rho = rho
        self.mu = mu
        self.S0 = S0

    def b(self, x, t):
        return self.r * x

    def sigma(self, x, t):
        return self.sigma * x

    def h(self, x, t, y, z):
        return -self.r * y

    def g(self, x):
        return np.sum(self.S0 * x, axis=1)
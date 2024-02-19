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

        self.T = T

    def b(self, x, t):
        pass

    def sigma(self, x, t):
        pass

    def h(self, x, t, y, z):
        pass

    def g(self, x):
        pass

    def price(self, X, t):
        pass


class BlackScholes(BackwardSDE):

        def __init__(self, X0, delta_t, T, r, sigma) -> None:
            super().__init__(X0, delta_t, T)

            self.r = r
            self.sigma_ = sigma

        def b(self, x, t):
            return np.zeros_like(x)

        def sigma(self, x, t):
            return self.sigma_ * np.diag(x)

        def h(self, x, t, y, z):
            return -self.r * (y - np.sum(x * z, axis=1))

        def g(self, x):
            return np.linalg.norm(x, axis=1) ** 2

        def price(self, X, t):
            return np.exp((self.r + self.sigma_**2)*(self.T-t)) * np.mean(self.g(X))

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

class HJB(BackwardSDE):

    def __init__(self, X0, delta_t, T, sigma) -> None:
        super().__init__(X0, delta_t, T)

        self.dim = X0.shape[-1]
        self.sigma_ = sigma

    def b(self, x, t):
        return np.zeros_like(x)

    def sigma(self, x, t):
        return self.sigma_ * np.eye(self.dim)

    def h(self, x, t, y, z):
        return - np.sum(z**2, axis=1)#-np.linalg.norm(z, axis=1) ** 2#

    def g(self, x):
        return np.log(1/2 + 1/2 * np.sum(x**2, axis=1))#np.linalg.norm(x, axis=1) ** 2)#

    def price(self, X, t, batch_size=1e3):
        noise = np.random.randn(X.shape[0], int(batch_size), X.shape[1])
        X_T = self.sigma_ * np.sqrt(self.T - t) * noise + np.broadcast_to(X, (int(batch_size), X.shape[0], X.shape[1])).transpose(1, 0, 2)
        return -np.log(np.mean(np.exp(-self.g(X_T)), axis=-1))

class DoubleWellHJB(BackwardSDE):

    def __init__(self, X0, delta_t, T, nu) -> None:
        super().__init__(X0, delta_t, T)

        self.dim = X0.shape[-1]
        self.C = 0.1 * np.eye(self.dim)
        self.nu = nu

    def b(self, x, t):
        return -(2 * x * ((x**2 - 1) @ self.C) + 2 * x * ((x**2 - 1) @ self.C.T))

    def sigma(self, x, t):
        return np.sqrt(2) * np.eye(self.dim)

    def h(self, x, t, y, z):
        return -1/2 * np.sum(z**2, axis=1)

    def g(self, x):
        return np.sum(self.nu * (x - 1)**2, axis=1)

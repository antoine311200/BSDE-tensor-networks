from math import ceil

import numpy as np

class BackwardSDE:
    """This class is used to represent a Partial Derivative Equation as a Backward Stochastic Differential Equation (BSDE).

    That is for a given parabolic PDE of the form:
    $$(\partial_t + L) V(x, t) + h(x, t, V(x, t), (\sigma^\intercal \Nabla V)(x, t)) = 0$$
    with
    $$L = \frac{1}{2} \sum_{i, j = 1}^d a_{i, j}(x, t) \partial_{x_i} \partial_{x_j} + \sum_{i = 1}^d b_i(x, t) \partial_{x_i}$$
    and $a_{i, j} = ((\sigma \sigma^\intercal)_{i, j})$.

    The terminal condition is given by $V(x, T) = g(x)$.
    """

    def __init__(self, X0, delta_t, T) -> None:
        self.dim = X0.shape[1]
        self.N = int(ceil(T / delta_t))

    def b(self, x, t):
        pass

    def sigma(self, x, t):
        pass

    def h(self, x, t, y, z):
        pass

    def g(self, x):
        pass

    def X(self, batch_size: int = 64, seed: int = 42):
        """Generate the backward sequence of X_t for t in [0, T] with a given time step.

        X_n+1 = X_n + b(X_n, t_n) * delta_t + sigma(X_n, t_n) * xi_n+1 * (W_{t_{n+1}} - W_{t_n})

        Args:
            batch_size (int, optional): Batch size. Defaults to 64.
            seed (int, optional): A random seed. Defaults to 42.
        """
        np.random.seed(seed)

        x = np.zeros((batch_size, self.N + 1, self.dim))
        xi = np.random.normal(size=(batch_size, self.N, self.dim))

        for n in range(self.N):
            x[:, n + 1] = x[:, n] + \
                self.b(x[:, n], n * self.delta_t) * self.delta_t + \
                self.sigma(x[:, n], n * self.delta_t) * np.sqrt(self.delta_t) * xi[:, n]

        return x

    def Y(self, batch_size: int = 64, seed: int = 42):
        """Generate the backward sequence of Y_t for t in [0, T] with a given time step.

        Y_n = Y_n+1 - h(X_n, t_n, Y_n, Z_n) * delta_t - Z_n * (W_{t_{n+1}} - W_{t_n})

        Args:
            batch_size (int, optional): Batch size. Defaults to 64.
            seed (int, optional): A random seed. Defaults to 42.
        """
        np.random.seed(seed)

        y = np.zeros((batch_size, self.N + 1))
        xi = np.random.normal(size=(batch_size, self.N, self.dim))

        for n in range(self.N - 1, -1, -1):
            y[:, n] = y[:, n + 1] - \
                self.h(self.X[:, n], n * self.delta_t, y[:, n + 1], self.Z[:, n]) * self.delta_t - \
                np.sum(self.Z[:, n] * np.sqrt(self.delta_t) * xi[:, n], axis=1)

        return y
import numpy as np

from bsde_solver.bsde import BackwardSDE
from bsde_solver.stochastic.path import generate_trajectories

import matplotlib.pyplot as plt

class HJB:

    def b(self, x, t):
        return np.zeros(x.shape)

    def sigma(self, x, t):
        return np.sqrt(2) * np.eye(x.shape[0])

    def h(self, x, y, z, t):
        # print(np.linalg.norm(z, axis=0) ** 2)
        # print(np.trace(z @ z.T))
        return -1/2 * np.linalg.norm(z, axis=0) ** 2

    def g(self, x):
        print(np.linalg.norm(x, axis=0) ** 2)
        print(np.trace(x @ x.T))
        return np.log(1/2 + 1/2 * np.linalg.norm(x, axis=0) ** 2)

if __name__ == "__main__":
    X0 = np.array([1, 3, 2])
    delta_t = 0.01
    T = 1
    N = int(T / delta_t)
    dim = 3
    batch_size = 4

    model = HJB()
    x = generate_trajectories(batch_size, N, dim, X0, model, delta_t)

    plt.figure(figsize=(8, 8), dpi=100)

    colormap = plt.cm.viridis
    time_range = np.linspace(0, T, N + 1)

    for i in range(batch_size):
        plt.subplot(batch_size, 1, i + 1)
        plt.plot(time_range, x[i, :, 0], color=colormap(i / batch_size), lw=0.8)
        plt.plot(time_range, x[i, :, 1], color=colormap(i / batch_size), lw=0.8)
        plt.plot(time_range, x[i, :, 2], color=colormap(i / batch_size), lw=0.8)
        plt.xlabel("Time")
        plt.ylabel("Value")
    plt.suptitle("Trajectories of the Hamilton-Jacobi-Bellman equation")
    plt.show()


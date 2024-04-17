from bsde_solver import xp

from bsde_solver.bsde import BackwardSDE, HJB
from bsde_solver.stochastic.path import generate_trajectories

import matplotlib.pyplot as plt

if __name__ == "__main__":
    X0 = xp.array([1, 3, 2])
    delta_t = 0.01
    T = 1
    N = int(T / delta_t)
    dim = 3
    batch_size = 4

    model = HJB(X0, delta_t, T)
    x = generate_trajectories(batch_size, N, dim, X0, model, delta_t)

    plt.figure(figsize=(8, 8), dpi=100)

    colormap = plt.cm.viridis
    time_range = xp.linspace(0, T, N + 1)

    for i in range(batch_size):
        plt.subplot(batch_size, 1, i + 1)
        plt.plot(time_range, x[i, :, 0], color=colormap(i / batch_size), lw=0.8)
        plt.plot(time_range, x[i, :, 1], color=colormap(i / batch_size), lw=0.8)
        plt.plot(time_range, x[i, :, 2], color=colormap(i / batch_size), lw=0.8)
        plt.xlabel("Time")
        plt.ylabel("Value")
    plt.suptitle("Trajectories of the Hamilton-Jacobi-Bellman equation")
    plt.show()


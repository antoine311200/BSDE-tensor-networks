import numpy as np

from bsde_solver.bsde import BackwardSDE


def generate_trajectories(batch_size, N, dim, X0, model, delta_t=0.01):
    """Generate the backward sequence of X_t for t in [0, T] with a given time step.

    X_n+1 = X_n + b(X_n, t_n) * delta_t + sigma(X_n, t_n) * xi_n+1 * (W_{t_{n+1}} - W_{t_n})

    Args:
        batch_size (int, optional): Batch size. Defaults to 64.
        N (int): Number of time steps.
        dim (int): Dimension of the process.
        X0 (np.ndarray): Initial condition.
        model (BackwardSDE): Model to simulate.
        delta_t (float, optional): Time step. Defaults to 0.01.

    Returns:
        np.ndarray: Trajectories of the process.
        np.ndarray: Noise.
    """
    x = np.zeros((batch_size, N + 1, dim))
    xi = np.random.normal(size=(batch_size, N + 1, dim))

    x[:, 0] = np.broadcast_to(X0, (batch_size, dim))

    for n in range(N):
        sigma = model.sigma(x[:, n], n*delta_t)
        if len(sigma.shape) != 1:
            x[:, n + 1] = x[:, n] + \
                model.b(x[:, n], n*delta_t) * delta_t + \
                xi[:, n] @ sigma * np.sqrt(delta_t)
        else:
            x[:, n + 1] = x[:, n] + \
                model.b(x[:, n], n*delta_t) * delta_t + \
                (xi[:, n] * sigma) * np.sqrt(delta_t)

    return x, xi
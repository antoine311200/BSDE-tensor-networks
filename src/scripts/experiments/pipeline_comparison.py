import numpy as np
import cProfile
from pstats import Stats
from functools import partial
from itertools import product

import matplotlib.pyplot as plt

from bsde_solver.bsde import BackwardSDE, HJB, DoubleWellHJB, BlackScholes, AllenCahn
from bsde_solver.stochastic.path import generate_trajectories
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.tensor.tensor_train import BatchTensorTrain
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.calculus.derivative import batch_derivative, multi_derivative
from bsde_solver.core.calculus.hessian import hessian
from bsde_solver.core.optimisation.als import ALS, SALSA
from bsde_solver.core.optimisation.mals import MALS
from bsde_solver.core.calculus.basis import LegendreBasis, PolynomialBasis
from bsde_solver.loss import PDELoss
from bsde_solver.utils import flatten, fast_contract, callable_name

import time
profiler = cProfile.Profile()
profiler.enable()

def initialize(X, n_assets, degree):
    basis = PolynomialBasis(degree)

    phi_X = []
    dphi_X = []
    ddphi_X = []

    # Compute phi_X and dphi_X for each time step
    for n in range(N + 1):
        phi_X_n = [TensorCore(basis.eval(X[:, n, i]), name=f"phi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(n_assets)]
        dphi_X_n = [TensorCore(basis.grad(X[:, n, i]), name=f"dphi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(n_assets)]
        ddphi_X_n = [TensorCore(basis.dgrad(X[:, n, i]), name=f"ddphi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(n_assets)]
        phi_X.append(phi_X_n)
        dphi_X.append(dphi_X_n)
        ddphi_X.append(ddphi_X_n)

    return phi_X, dphi_X, ddphi_X

def run_pipeline(model, phi_X, dphi_X, ddphi_X, n_asset, batch_size, degree, n_iter, rank, N, T, algo):
    Y = np.zeros((batch_size, N + 1))
    Y[:, -1] = model.g(X[:, -1])  # (batch_size, )

    start_time = time.perf_counter()
    V = [None for _ in range(N + 1)]
    V_N = algo(phi_X[-1], Y[:, -1])
    V[-1] = V_N
    print("Time to compute V_N:", f"{time.perf_counter() - start_time:.2f}s")

    check_V = fast_contract(V_N, phi_X[-1]).view(np.ndarray).squeeze()

    print("Mean reconstruction error at N:", f"{np.abs(np.mean(check_V - Y[:, -1])):.2e}")
    print("Prediction at N:", f"{np.mean(Y[:, -1]):.4f} | Value at N:", f"{np.mean(check_V):.4f}")

    print("Compute true prices")
    prices = []

    for n in range(N + 1):
        prices.append(model.price(X[:, n, :], n*dt))

    print("Start optimisation")
    start_time = time.perf_counter()

    step_times = []
    relative_errors = []
    errors = []

    relative_errors.append(np.abs(prices[0] - np.mean(Y[:, 0]) / prices[0]))

    for n in range(N - 1, -1, -1):
        print("Step:", n)
        step_start_time = time.perf_counter()
        # Compute Y = V_n+1(X_n+1)
        # Compute Z = grad_x V_n+1(X_n+1)
        # Compute h_n(X_n, t_n, Y, Z)
        # Compute dt * h_n(X_n, t_n, Y, Z) + V_n+1(X_n+1)
        V_n1 = V[n + 1]
        Y_n1 = Y[:, n + 1]  # (batch_size, )

        X_n1 = X[:, n + 1, :]  # (batch_size, num_assets)
        phi_X_n1 = phi_X[n+1]  # tensor core of shape (batch_size, degree) * num_assets
        dphi_X_n1 = dphi_X[n+1]  # tensor core of shape (batch_size, degree) * num_assets

        X_n = X[:, n, :]  # (batch_size, num_assets)
        phi_X_n = phi_X[n]  # tensor core of shape (batch_size, degree) * num_assets
        dphi_X_n = dphi_X[n]  # tensor core of shape (batch_size, degree) * num_assets

        Z_n1 = multi_derivative(V_n1, phi_X_n1, dphi_X_n1)  # (batch_size, num_assets)
        sigma_n1 = model.sigma(X_n1, (n+1) *dt)
        h_n1 = model.h(X_n1, (n+1) * dt, Y_n1, Z_n1)  # (batch_size, )

        step_n1 = h_n1*dt + Y_n1
        if n == 0 and callable_name(algo) == "SALSA":
            algo = partial(algo, do_reg=False)
        V_n = algo(phi_X_n, step_n1, n_iter=n_iter, ranks=ranks, init_tt=V_n1)
        V[n] = V_n
        Y_n = fast_contract(V_n, phi_X_n)

        Y[:, n] = Y_n.view(np.ndarray).squeeze()

        step_times.append(time.perf_counter() - step_start_time)

        print("Mean reconstruction error at n:", f"{np.mean(np.abs(prices[n] - Y_n)):.2e}, Max reconstruction error at n:", f"{np.max(np.abs(prices[n] - Y_n)):.2e}")
        print("Step time:", f"{time.perf_counter() - step_start_time:.2f}s")

        price_n = model.price(X_n, n*dt)
        relative_errors.append(np.abs(price_n - np.mean(Y[:, n])) / price_n)
        errors.append(np.abs(price_n - np.mean(Y[:, n])))

    print("End")

    end_time = time.perf_counter()

    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Mean step time: {np.mean(step_times):.2f}s")

    print()
    price = np.mean(model.price(X0_batch, 0))
    print("Price at 0", price)
    print("Predicted Price:", np.mean(Y[:, 0]))

    print("Relative error:", f"{np.abs(price - np.mean(Y[:, 0])) / price * 100:.2f}%")


    return Y, V, step_times, relative_errors, errors

if __name__ == "__main__":
    seed = 54
    np.random.seed(seed)

    result_dict = {}

    n_assets = [4]
    batch_sizes = [5000]
    degrees = [2]
    n_iters = [25]  # [1, 10, 25, 50, 100, 250, 500]#
    ranks = [2]
    Ns = [50]
    Ts = [1]

    algos = [
        ALS,
        partial(SALSA, max_rank=5),
    ]


    for n_asset, batch_size, degree, n_iter, rank, N, T, algo in product(
        n_assets, batch_sizes, degrees, n_iters, ranks, Ns, Ts, algos
    ):
        print(
            f"n_assets: {n_asset}, batch_size: {batch_size}, degree: {degree}, n_iter: {n_iter}, rank: {rank}"
        )

        X0 = np.array(flatten([(1, 0.5) for _ in range(n_asset//2)])) # Black-Scholes initial condition
        X0_batch = np.broadcast_to(X0, (batch_size, n_asset))

        sigma = 0.4
        r = 0.05
        dt = T / N
        model = BlackScholes(X0, dt, T, r, sigma)
        pde_loss = PDELoss(model)

        # Compute trajectories
        X, noise = generate_trajectories(X0_batch, N, model)
        phi_X, dphi_X, ddphi_X = initialize(X, n_asset, degree)

        # Get the algo name from the partial function
        algo = partial(algo, n_iter=n_iter, ranks=(1,) + (rank,) * (n_asset - 1) + (1,))
        result_dict[(callable_name(algo), n_asset, batch_size, degree, n_iter, rank)] = run_pipeline(
            model,
            phi_X, dphi_X, ddphi_X,
            n_asset, batch_size, degree,
            n_iter, rank,
            N, T,
            algo
        )

    plt.figure(figsize=(10, 5))
    n_simulations = 3
    colormap = plt.cm.viridis

    simulation_indices = np.random.choice(batch_size, n_simulations, replace=False)

    linestyles = ["dashed", "dashdot", "dotted", (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1))]
    lws = [0.8, 1.4]
    for k, idx in enumerate(simulation_indices):
        ground_prices = [model.price(np.array([X[idx, i]]), i * dt) for i in range(N + 1)]
        for l, (key, (Y, V, step_times, relative_errors, errors)) in enumerate(result_dict.items()):
            algo_name, n_asset, batch_size, degree, n_iter, rank = key
            predicted_prices = [Y[idx, i] for i in range(N + 1)]

            linestyle = linestyles[l]
            label = f"{algo_name} - {n_asset} assets"
            plt.plot(predicted_prices, linestyle=linestyle, lw=0.8, label=label if k == 0 else None, color=colormap(k / n_simulations))

        plt.plot(ground_prices, label="Ground truth" if k == 0 else None, lw=0.8, color=colormap(k / n_simulations))

    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    # Show relative errors
    for key, (Y, V, step_times, relative_errors, errors) in result_dict.items():
        algo_name, n_asset, batch_size, degree, n_iter, rank = key
        plt.plot(np.arange(N + 1), relative_errors, label=f"{algo_name} - {n_asset} assets", lw=0.8)

    plt.xlabel("Time")
    plt.ylabel("Relative error")
    plt.legend()
    plt.show()

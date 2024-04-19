from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.optimisation.als import ALS, SALSA
from bsde_solver.core.optimisation.mals import MALS
from bsde_solver.utils import fast_contract, callable_name

import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from functools import partial
from itertools import product
import sys


def poly(x, degree=10):
    return np.array([x**i for i in range(degree + 1)]).T


def initialize(n_assets, batch_size, degree, func):
    X, phi_X = [], []
    for i in range(batch_size):
        x = (np.random.rand(n_assets) - 1 / 2)
        phi = [
            TensorCore(
                poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",)
            )
            for i in range(n_assets)
        ]
        X.append(x)
        phi_X.append(phi)

    X = np.array(X)
    phi_X = np.array(phi_X).transpose(1, 0, 2)
    phi_X = [
        TensorCore(phi_X[i], name=f"phi_{i+1}", indices=("batch", f"m_{i+1}"))
        for i in range(n_assets)
    ]

    b = func(X)

    return phi_X, b


def run(X, Y, algo):
    start_time_batch = perf_counter()
    A = algo(X, Y)
    end_time_batch = perf_counter() - start_time_batch

    print("Time:", end_time_batch)

    batch_phis = TensorNetwork(cores=X)
    result = fast_contract(A, batch_phis).view(np.ndarray).squeeze()

    l2 = np.linalg.norm(result - Y)
    l1 = np.linalg.norm(result - Y, ord=1)

    print("Reconstruction error (total)   L2:", round(l2, 4), "   L1:", round(l1, 4))
    print("Mean reconstruction error ", round(np.mean(np.abs(result - Y)), 4))
    print("Maximum reconstruction error:", round(np.max(np.abs(result - Y)), 4))
    print("Ground truth samples:", [round(c, 3) for c in Y[:10]])
    print("Reconstruction samples:", [round(c, 3) for c in result[:10]])

    return A, result, l2, l1


if __name__ == "__main__":
    seed = 54
    np.random.seed(seed)

    result_dict = {}

    n_assets = [20]#[2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    batch_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    # degrees = [1, 2, 3, 4, 5]
    degrees = [2]
    n_iters = [25]  # [1, 10, 25, 50, 100, 250, 500]#
    ranks = [2]

    algos = [
        ALS,
        partial(
            SALSA,
            max_rank=5,
        ),
    ]

    for n_asset, batch_size, degree, n_iter, rank, algo in product(
        n_assets, batch_sizes, degrees, n_iters, ranks, algos
    ):
        print(
            f"n_assets: {n_asset}, batch_size: {batch_size}, degree: {degree}, n_iter: {n_iter}, rank: {rank}"
        )
        phi_X, b = initialize(
            n_asset,
            batch_size,
            degree,
            lambda x: np.linalg.norm(np.array(x), axis=1) ** 2,
        )
        # Get the algo name from the partial function
        algo = partial(algo, n_iter=n_iter, ranks=(1,) + (rank,) * (n_asset - 1) + (1,))
        result_dict[(callable_name(algo), n_asset, batch_size, degree, n_iter, rank)] = run(phi_X, b, algo)

    plt.figure(figsize=(10, 10))
    algo_errors = {}
    for key, (A, result, l2, l1) in result_dict.items():
        algo_name, n_asset, batch_size, degree, n_iter, rank = key
        algo_errors.setdefault(algo_name, []).append(l2)

    algo_colors = {
        "ALS": "blue",
        "SALSA": "red",
    }
    for algo_name, errors in algo_errors.items():
        # plt.plot(n_assets, errors, label=algo_name, color=algo_colors[algo_name], marker="o", linestyle="-", lw=0.8)
        # plt.plot(n_iters, errors, label=algo_name, color=algo_colors[algo_name], marker="o", linestyle="-", lw=0.8)
        plt.plot(batch_sizes, errors, label=algo_name, color=algo_colors[algo_name], marker="o", linestyle="-", lw=0.8)

    plt.xlabel("Batch size")
    plt.ylabel("L2 error")
    plt.legend()
    plt.show()
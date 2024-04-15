from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.optimisation.als import ALS, ALS_regularized
from bsde_solver.core.optimisation.mals import MALS
from bsde_solver.utils import fast_contract

import numpy as np
from time import perf_counter
from functools import partial
from itertools import product

def poly(x, degree=10):
    return np.array([x**i for i in range(degree+1)]).T

def initialize(n_assets, batch_size, degree, func):
    X, phi_X = [], []
    for i in range(batch_size):
        x = (np.random.rand(n_assets)-1/2)
        phi = [TensorCore(poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",)) for i in range(n_assets)]
        X.append(x)
        phi_X.append(phi)

    X = np.array(X)
    phi_X = np.array(phi_X).transpose(1, 0, 2)
    phi_X = [TensorCore(phi_X[i], name=f"phi_{i+1}", indices=("batch", f"m_{i+1}")) for i in range(n_assets)]

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
    seed = 216540
    np.random.seed(seed)

    result_dict = {}

    n_assets = [4]
    batch_sizes = [1000]
    degrees = [2, 3, 4, 5]
    rank = 3

    # funcs = [lambda x: np.sum(x, axis=1), lambda x: np.sum(x**2, axis=1), lambda x: np.sum(x**3, axis=1), lambda x: np.sum(x**4, axis=1)]
    # func(x1, ..., xn) = x1**4 + x2**4 + ... + xn**4
    # func(x1, ..., xn) = (x1+x2+...+xn)**4
    # func = lambda x: np.sum(x**4, axis=1)
    func = lambda x: np.sum(x**2, axis=1)**2

    for n_asset, batch_size, degree in product(n_assets, batch_sizes, degrees):
        print('~'*50)
        print(f"n_asset: {n_asset}, batch_size: {batch_size}, degree: {degree}")

        ranks = (1, ) + (rank,) * (n_asset - 1) + (1, )
        algo = partial(
            ALS,
            n_iter=25,
            ranks=ranks,
            optimizer="lstsq",
        )

        X, Y = initialize(n_asset, batch_size, degree, func)
        A, result, l2, l1 = run(X, Y, algo)
        result_dict[(n_asset, batch_size, degree)] = (A, result, l2, l1)

    # Plot the L2 and L1 errors
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    for k, v in result_dict.items():
        degree = k[2]
        ax[0].plot(degree, v[2], 'o', label=f"Degree {degree}")
        ax[1].plot(degree, v[3], 'o', label=f"Degree {degree}")

    ax[0].set_title("L2 error")
    ax[1].set_title("L1 error")
    ax[0].legend()
    ax[1].legend()
    plt.show()
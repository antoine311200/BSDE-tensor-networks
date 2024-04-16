from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.optimisation.als import ALS, SALSA
from bsde_solver.core.optimisation.mals import MALS
from bsde_solver.utils import fast_contract

import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from functools import partial
from itertools import product
import sys

def poly(x, degree=10):
    return np.array([x**i for i in range(degree+1)]).T

def initialize(n_assets, batch_size, degree, func):
    X, phi_X = [], []
    for i in range(batch_size):
        x = (np.random.rand(n_assets)-1/2) * 10
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
    seed = 654
    np.random.seed(seed)

    result_dict = {}

    n_assets = [4]
    batch_sizes = [1000]
    # degrees = [1, 2, 3, 4, 5]
    degrees = [4]
    n_iters = [1, 2, 3, 4, 6, 8, 10, 25, 50]#[1, 10, 25, 50, 100, 250, 500]#
    ranks = [2]

    # funcs = [lambda x: np.sum(x, axis=1), lambda x: np.sum(x**2, axis=1), lambda x: np.sum(x**3, axis=1), lambda x: np.sum(x**4, axis=1)]
    # func(x1, ..., xn) = x1**4 + x2**4 + ... + xn**4
    # func(x1, ..., xn) = (x1+x2+...+xn)**4
    # func(x1, ..., xn) = x1*x2
    # func_tex = r"x_1 \cdot x_2^2"
    # func = lambda x: x[:, 0] * x[:, 1] ** 2
    #np.sum(x**4, axis=1)

    func_tex = "$\sum_{i=1}^n (x_i^2 + x_i^4)$"
    func = lambda x: np.sum(x**2, axis=1) + np.sum(x**4, axis=1)

    # func_tex = "$\prod_{i=1}^n x_i$"
    # func = lambda x: np.prod(x, axis=1) # np.sum(x**4, axis=1) + np.sum(x**2, axis=1) #+ 

    # func prod x_2i + prod x_2i+1**3
    # func_tex = r"$\prod_{i=1}^n x_{2i} + \prod_{i=1}^n x_{2i+1}^3$"
    # func = lambda x: np.prod(x[:, ::2], axis=1) + np.prod(x[:, 1::2] ** 3, axis=1)

    # func = lambda x: np.sum(x**2, axis=1)**2
    # func = lambda x: np.linalg.norm(x, axis=1)**2

    for n_asset, batch_size, degree, rank, n_iter in product(n_assets, batch_sizes, degrees, ranks, n_iters):
        print('~'*50)
        print(f"n_asset: {n_asset}, batch_size: {batch_size}, degree: {degree}")

        # algo = partial(
        #     ALS,
        #     n_iter=n_iter,
        #     ranks=(1, ) + (rank,) * (n_asset - 1) + (1, ),
        #     optimizer="lstsq",
        # )
        algo = partial(
            SALSA,
            n_iter=n_iter,
            ranks=(1, ) + (rank,) * (n_asset - 1) + (1, ),
            max_rank=2,
            optimizer="lstsq",
        )

        X, Y = initialize(n_asset, batch_size, degree, func)
        A, result, l2, l1 = run(X, Y, algo)
        result_dict[(n_asset, batch_size, degree, rank, n_iter)] = (A, result, l2, l1)

        reduced_densities = []

        for core in A.cores.values():
            c = core.copy()
            cT = core.copy().rename("m_*", 'n_*')
            rho = TensorNetwork(cores=[c, cT], names=["C", "C_T"]).contract()
            print(np.round(np.diag(rho.view(np.ndarray)), 4))
            reduced_densities.append(np.diag(rho.view(np.ndarray)))

        full_rdm = np.array(reduced_densities)

        # Plot a grid as an image of the reduced density matrices
        plt.figure(figsize=(10, 10), dpi=200)
        # binarize full_rdm
        # full_rdm = (full_rdm > 1e-2).astype(int)
        plt.imshow(full_rdm, cmap='viridis', interpolation='nearest')
        plt.xticks(range(full_rdm.shape[1]), [f"Degree {i}" for i in range(full_rdm.shape[1])])
        plt.yticks(range(full_rdm.shape[0]), [f"Asset {i+1}" for i in range(full_rdm.shape[0])])
        plt.title(r"Reduced density matrices for {}".format(func_tex))
        # plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"../Images/rdm_asset_{n_asset}_bs_{batch_size}_d_{degree}_r_{rank}_it_{n_iter}.png", transparent=True, dpi=200)
        # plt.show()
        plt.close()

    # sys.exit()

    # Plot the L2 and L1 errors as bar plots
    import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    colors = ['skyblue', 'lightcoral']
    edgecolors = ['blue', 'red']
    for i, metric in enumerate(['L2', 'L1']):
        plt.figure(figsize=(10, 8), dpi=100)
        data = []
        for key, value in result_dict.items():
            data.append(value[i+2])

        plt.bar(range(len(data)), data, linewidth=0.8, edgecolor=edgecolors[i], color=colors[i], alpha=0.8)
        for j, value in enumerate(data):
            plt.text(j, value, round(value, 4), ha='center', va='bottom')
        # .set_xticks(range(len(data)))
        # plt.xticks(range(len(data)), [f"Assets {key[0]}" for key in result_dict.keys()], rotation=45, fontsize=8)
        # plt.xlabel("Assets")

        plt.xticks(range(len(data)), [f"Iterations {key[-1]}" for key in result_dict.keys()], rotation=45, fontsize=8)
        plt.xlabel("Iterations")

        # plt.xticks(range(len(data)), [f"Rank {key[-2]}" for key in result_dict.keys()], rotation=45, fontsize=8)
        # plt.xlabel("Ranks")

        # plt.xticks(range(len(data)), [f"Iterations {key[-1]}" for key in result_dict.keys()], rotation=45, fontsize=8)
        # ax[i].set_xticklabels([f"Iterations {key[-1]}" for key in result_dict.keys()], rotation=45, fontsize=8)
        # ax[i].set_xticklabels([f"Degree {key[2]}" for key in result_dict.keys()], rotation=45, fontsize=8)
        # ax[i].set_title(f"{metric} errors")
        # ax[i].set_ylabel(f"{metric} error")
        plt.title(f"{metric} errors")
        plt.ylabel(f"{metric} error")
        plt.tight_layout()
        # plt.xlabel("Degree")
        plt.savefig(f"../Images/{metric}_errors.png", transparent=True, dpi=200)
        # plt.show()
        plt.close()

    # plt.show()
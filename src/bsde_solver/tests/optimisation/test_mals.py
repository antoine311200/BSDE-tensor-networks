from time import perf_counter

import numpy as np

from bsde_solver.tensor.tensor_core import TensorCore
from bsde_solver.tensor.tensor_network import TensorNetwork
from bsde_solver.optimisation.mals import scalar_MALS

def poly(x, degree=10):
    return np.array([x**i for i in range(degree)]).T

def poly_derivative(x, degree=10):
    return np.array([i * x ** (i - 1) for i in range(degree)]).T


if __name__ == "__main__":

    seed = 55
    degree = 8
    num_assets = 8

    shape = [degree for _ in range(num_assets)]
    dim = 2
    ranks = (1, ) + (dim,) * (num_assets - 1) + (1, )

    n_simulations = 1
    phis, dphis = [], []

    np.random.seed(seed)
    for i in range(n_simulations):
        x = np.random.rand(num_assets)*5

        phi = [TensorCore(poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",)) for i in range(num_assets)]
        dphi = [TensorCore(poly_derivative(x[i], degree=degree), name=f"dphi_{i+1}", indices=(f"m_{i+1}",),) for i in range(num_assets)]

        phis.append(phi)
        dphis.append(dphi)

    b = np.random.rand(n_simulations)*10
    print(b)

    #################### Single ALS ####################

    start_time = perf_counter()
    min_tt = []
    for i in range(n_simulations):
        new_tt = scalar_MALS(phis[i], result=b[i], n_iter=1, ranks=ranks)
        min_tt.append(new_tt)

    end_time = perf_counter() - start_time
    print("Time:", end_time)

    results = []
    for i in range(n_simulations):
        result = TensorNetwork(cores=[min_tt[i]]+phis[i], names=["tt"]+[f"phi_{i+1}" for i in range(num_assets)]).contract().view(np.ndarray).squeeze()
        print("Reconstruction error:", np.linalg.norm(result - b[i]))
        # print("Result:", np.round(result, 5), "Expected:", np.round(b[i], 5))

        results.append(float(result))

    print(results)
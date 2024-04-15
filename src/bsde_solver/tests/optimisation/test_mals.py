from time import perf_counter

from bsde_solver import xp

from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.optimisation.mals import scalar_MALS

def poly(x, degree=10):
    return xp.array([x**i for i in range(degree)]).T

def poly_derivative(x, degree=10):
    return xp.array([i * x ** (i - 1) for i in range(degree)]).T


if __name__ == "__main__":

    seed = 55
    degree = 8
    num_assets = 8

    shape = [degree for _ in range(num_assets)]
    dim = 3
    ranks = (1, ) + (dim,) * (num_assets - 1) + (1, )

    n_simulations = 1
    phis, dphis = [], []

    xp.random.seed(seed)
    for i in range(n_simulations):
        x = xp.random.rand(num_assets)*4

        phi = [TensorCore(poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",)) for i in range(num_assets)]
        dphi = [TensorCore(poly_derivative(x[i], degree=degree), name=f"dphi_{i+1}", indices=(f"m_{i+1}",),) for i in range(num_assets)]

        phis.append(phi)
        dphis.append(dphi)

    b = xp.random.rand(n_simulations)*10
    print(b)

    #################### Single ALS ####################

    start_time = perf_counter()
    min_tt = []
    for i in range(n_simulations):
        new_tt = scalar_MALS(phis[i], result=b[i], n_iter=2, ranks=ranks)
        min_tt.append(new_tt)

    end_time = perf_counter() - start_time
    print("Time:", end_time)

    results = []
    for i in range(n_simulations):
        result = TensorNetwork(cores=[min_tt[i]]+phis[i], names=["tt"]+[f"phi_{i+1}" for i in range(num_assets)]).contract().view(xp.ndarray).squeeze()
        print("Reconstruction error:", xp.linalg.norm(result - b[i]))
        # print("Result:", xp.round(result, 5), "Expected:", xp.round(b[i], 5))

        results.append(float(result))

    print(results)

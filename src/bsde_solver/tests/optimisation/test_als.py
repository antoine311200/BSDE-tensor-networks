from bsde_solver.tensor.tensor_train import BatchTensorTrain, TensorTrain, left_unfold, right_unfold
from bsde_solver.tensor.tensor_core import TensorCore
from bsde_solver.tensor.tensor_network import TensorNetwork
from bsde_solver.utils import batch_qr
from bsde_solver.optimisation.scalar_als import scalar_ALS, batch_scalar_ALS

import numpy as np
from time import perf_counter

if __name__ == "__main__":

    seed = 855
    degree = 8
    num_assets = 50

    shape = [degree for _ in range(num_assets)]
    dim = 3
    ranks = (1, ) + (dim,) * (num_assets - 1) + (1, )

    def poly(x, degree=10):
        return np.array([x**i for i in range(degree)]).T

    def poly_derivative(x, degree=10):
        return np.array([i * x ** (i - 1) for i in range(degree)]).T

    n_simulations = 5
    phis, dphis = [], []

    np.random.seed(seed)
    for i in range(n_simulations):
        x = np.random.rand(num_assets)*4

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
        new_tt = scalar_ALS(phis[i], result=b[i], n_iter=5, ranks=ranks)
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

    #################### Batch ALS ####################

    phis, dphis = [], []
    np.random.seed(seed)
    for i in range(n_simulations):
        x = np.random.rand(num_assets)*4

        phi = [poly(x[i], degree=degree) for i in range(num_assets)]
        dphi = [poly_derivative(x[i], degree=degree) for i in range(num_assets)]
        phis.append(phi)
        dphis.append(dphi)

    phis = np.array(phis) # (n_simulations, tt.order, degree)
    dphis = np.array(dphis) # (n_simulations, tt.order, degree)

    phis = phis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    dphis = dphis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    phis = [TensorCore(phis[i], name=f"phi_{i+1}", indices=("batch", f"m_{i+1}")) for i in range(num_assets)]
    dphis = [TensorCore(dphis[i], name=f"dphi_{i+1}", indices=("batch", f"m_{i+1}")) for i in range(num_assets)]

    b = np.random.rand(n_simulations)*10

    start_time_batch = perf_counter()
    batch_als = batch_scalar_ALS(phis, b, n_iter=5, ranks=ranks)
    end_time_batch = perf_counter() - start_time_batch
    print("Time:", end_time_batch)

    unbatch_als = batch_als.unbatch()

    results = []
    for i in range(n_simulations):
        sample_phis = [TensorCore(phis[j][i], name=f"phi_{j+1}", indices=(f"m_{j+1}",))
            for j in range(num_assets)
        ]
        result = TensorNetwork(cores=[unbatch_als[i]]+sample_phis, names=["tt"]+[f"phi_{i+1}" for i in range(num_assets)]).contract().view(np.ndarray).squeeze()
        print("Reconstruction error:", np.linalg.norm(result - b[i]))
        # print("Result:", np.round(result, 5), "Expected:", np.round(b[i], 5))
        results.append(float(result))

    print(results)

    # print("Speed up factor:", end_time / end_time_batch)
from bsde_solver.core.tensor.tensor_train import BatchTensorTrain, TensorTrain, left_unfold, right_unfold
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.optimisation.scalar_als import scalar_ALS, batch_scalar_ALS, multi_als
from bsde_solver.utils import batch_qr

import numpy as np
from time import perf_counter

if __name__ == "__main__":

    seed = 5454
    degree = 10
    num_assets = 5

    shape = [degree for _ in range(num_assets)]
    dim = 5
    ranks = (1, ) + (dim,) * (num_assets - 1) + (1, )

    def poly(x, degree=10):
        return np.array([x**i for i in range(degree)]).T

    def poly_derivative(x, degree=10):
        return np.array([i * x ** (i - 1) for i in range(degree)]).T

    n_simulations = 10000
    xs, phis, dphis = [], [], []

    np.random.seed(seed)
    for i in range(n_simulations):
        x = np.random.rand(num_assets)-1/2
        phi = [TensorCore(poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",)) for i in range(num_assets)]
        dphi = [TensorCore(poly_derivative(x[i], degree=degree), name=f"dphi_{i+1}", indices=(f"m_{i+1}",),) for i in range(num_assets)]

        xs.append(x)
        phis.append(phi)
        dphis.append(dphi)

    # Relu of x
    # b = np.maximum(0, np.mean(np.array(xs), axis=1))
    b = np.log(1/2+1/2*np.linalg.norm(np.array(xs)**2, axis=1))

    #################### Single ALS ####################

    # print("Start Single Start")
    # start_time = perf_counter()
    # min_tt = []
    # for i in range(n_simulations):
    #     new_tt = scalar_ALS(phis[i], result=b[i], n_iter=25, ranks=ranks)
    #     min_tt.append(new_tt)

    # end_time = perf_counter() - start_time
    # print("Time:", end_time)

    # results = []
    # for i in range(n_simulations):
    #     result = TensorNetwork(cores=[min_tt[i]]+phis[i], names=["tt"]+[f"phi_{i+1}" for i in range(num_assets)]).contract().view(np.ndarray).squeeze()
    #     results.append(float(result))

    # print("Reconstruction error (batch):", np.linalg.norm(results - b))
    # print("Mean reconstruction error (batch):", np.mean(np.abs(results - b)))
    # print("Max reconstruction error (batch):", np.max(np.abs(results - b)))

    #################### Batch ALS ####################

    phis, dphis = [], []
    for i in range(n_simulations):
        x = xs[i]
        phi = [poly(x[j], degree=degree) for j in range(num_assets)]
        dphi = [poly_derivative(x[j], degree=degree) for j in range(num_assets)]
        phis.append(phi)
        dphis.append(dphi)

    phis = np.array(phis) # (n_simulations, tt.order, degree)
    dphis = np.array(dphis) # (n_simulations, tt.order, degree)

    phis = phis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    dphis = dphis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    phis = [TensorCore(phis[i], name=f"phi_{i+1}", indices=("batch", f"m_{i+1}")) for i in range(num_assets)]
    dphis = [TensorCore(dphis[i], name=f"dphi_{i+1}", indices=("batch", f"m_{i+1}")) for i in range(num_assets)]

    print("Start ALS")
    start_time_batch = perf_counter()
    multi_als_result = multi_als(phis, b, n_iter=10, ranks=ranks)
    end_time_batch = perf_counter() - start_time_batch
    print("Time:", end_time_batch)

    batch_phis = TensorNetwork(cores=phis)
    result = TensorNetwork(cores=[multi_als_result, batch_phis], names=["tt", "phi"]).contract(batch=True).view(np.ndarray).squeeze()

    print("Reconstruction error (batch):", np.linalg.norm(result - b))
    print("Mean reconstruction error (batch):", np.mean(np.abs(result - b)))
    print("Max reconstruction error (batch):", np.max(np.abs(result - b)))

    # print("Speed up factor:", end_time / end_time_batch)
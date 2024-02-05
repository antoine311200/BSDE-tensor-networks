from bsde_solver.tensor.tensor_train import BatchTensorTrain, TensorTrain, left_unfold, right_unfold
from bsde_solver.tensor.tensor_core import TensorCore
from bsde_solver.tensor.tensor_network import TensorNetwork
from bsde_solver.utils import batch_qr

import numpy as np


def retraction_operator(tt, i):
    operator = tt.extract([f"core_{j}" for j in range(tt.order) if j != i])
    return operator

def scalar_ALS(phis: list[TensorCore], result: float, n_iter=10, ranks=None):
    shape = tuple([phi.shape[0] for phi in phis])
    tt = TensorTrain(shape, ranks).randomize()
    tt.orthonormalize(mode="right", start=1)

    phis = TensorNetwork(cores=phis)

    def micro_optimization(tt, j):
        P = retraction_operator(tt, j)
        V = TensorNetwork(cores=[P, phis], names=["P", "phi"]).contract(indices=(f"r_{j}", f"m_{j+1}", f"r_{j+1}"))
        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")

        # The operator A can be non invertible, so we need to use some regularization
        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract()
        A = A.unfold((f"s_{j}", f"n_{j+1}", f"s_{j+1}"), (f"r_{j}", f"m_{j+1}", f"r_{j+1}"))
        V = V.unfold((f"r_{j}", f"m_{j+1}", f"r_{j+1}"))

        # Add regularization
        A += 0.1 * np.random.rand(A.shape[1], A.shape[1])
        V *= result

        # X = np.linalg.solve(A.view(np.ndarray), V.view(np.ndarray))
        X = np.linalg.lstsq(A.view(np.ndarray), V.view(np.ndarray), rcond=None)[0]
        X = TensorCore.like(X, tt.cores[f"core_{j}"])
        return X

    for _ in range(n_iter):
        # Left half sweep
        for j in range(tt.order - 1):
            # Micro optimization
            V = micro_optimization(tt, j)

            core_curr = tt.cores[f"core_{j}"]
            core_next = tt.cores[f"core_{j+1}"]

            L = left_unfold(V).view(np.ndarray)
            R = right_unfold(core_next).view(np.ndarray)

            Q, S = np.linalg.qr(L)
            W = S @ R

            tt.cores[f"core_{j}"] = TensorCore.like(Q, core_curr)
            tt.cores[f"core_{j+1}"] = TensorCore.like(W, core_next)

        # Right half sweep
        for j in range(tt.order - 1, 0, -1):
            # Micro optimization
            V = micro_optimization(tt, j)

            core_prev = tt.cores[f"core_{j-1}"]
            core_curr = tt.cores[f"core_{j}"]

            L = left_unfold(core_prev).view(np.ndarray)
            R = right_unfold(V).view(np.ndarray)

            Q, S = np.linalg.qr(R.T)
            W = L @ S.T

            tt.cores[f"core_{j-1}"] = TensorCore.like(W, core_prev)
            tt.cores[f"core_{j}"] = TensorCore.like(Q.T, core_curr)

    return tt


def batch_scalar_ALS(phis: list[TensorCore], result: list[float], n_iter=10, ranks=None):
    shape = tuple([phi.shape[1] for phi in phis])
    batch_size = len(result)

    batch_tt = BatchTensorTrain(batch_size, shape, ranks).randomize()
    batch_tt.orthonormalize(mode="right", start=1)

    phis = TensorNetwork(cores=phis)
    result = np.array(result)[:, np.newaxis]

    def micro_optimization(tt, j):
        P = retraction_operator(tt, j)
        V = TensorNetwork(cores=[P, phis], names=["P", "phi"]).contract(batch=True, indices=(f"r_{j}", f"m_{j+1}", f"r_{j+1}"))
        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")

        # The operator A can be non invertible, so we need to use some regularization
        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract(batch=True)
        A = A.unfold("batch", (f"s_{j}", f"n_{j+1}", f"s_{j+1}"), (f"r_{j}", f"m_{j+1}", f"r_{j+1}"))
        V = V.unfold("batch", (f"r_{j}", f"m_{j+1}", f"r_{j+1}"))

        # Add regularization
        A += 0.1 * np.stack([
            np.random.rand(A.shape[1], A.shape[1])
            for _ in range(batch_size)
        ])
        V *= result

        X = np.linalg.solve(A.view(np.ndarray), V.view(np.ndarray))
        X = TensorCore.like(X, tt.cores[f"core_{j}"])
        return X

    for _ in range(n_iter):
        # Left half sweep
        for j in range(batch_tt.order - 1):
            # Micro optimization
            V = micro_optimization(batch_tt, j)

            core_curr = batch_tt.cores[f"core_{j}"]
            core_next = batch_tt.cores[f"core_{j+1}"]

            L = V.unfold(0, (1, 2), 3).view(np.ndarray)
            R = core_next.unfold(0, 1, (2, 3)).view(np.ndarray)

            Q, S = batch_qr(L)
            W = np.einsum("bij,bjk->bik", S, R)

            batch_tt.cores[f"core_{j}"] = TensorCore.like(Q, core_curr)
            batch_tt.cores[f"core_{j+1}"] = TensorCore.like(W, core_next)

        # Right half sweep
        for j in range(batch_tt.order - 1, 0, -1):
            # Micro optimization
            V = micro_optimization(batch_tt, j)

            core_prev = batch_tt.cores[f"core_{j-1}"]
            core_curr = batch_tt.cores[f"core_{j}"]

            L = core_prev.unfold(0, (1, 2), 3).view(np.ndarray)
            R = V.unfold(0, (2, 3), 1).view(np.ndarray)

            Q, S = batch_qr(R)
            W = np.einsum("bij,bkj->bik", L, S)

            batch_tt.cores[f"core_{j-1}"] = TensorCore.like(W, core_prev)
            batch_tt.cores[f"core_{j}"] = TensorCore.like(Q.transpose(0, 2, 1), core_curr)

    return batch_tt

if __name__ == "__main__":
    from time import perf_counter

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

    n_simulations = 2
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
        result = TensorNetwork(cores=[new_tt]+phis[i], names=["tt"]+[f"phi_{i+1}" for i in range(num_assets)]).contract().view(np.ndarray).squeeze()
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
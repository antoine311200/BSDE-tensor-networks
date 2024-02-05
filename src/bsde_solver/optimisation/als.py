from typing import Union

from bsde_solver.tensor.tensor_train import TensorTrain, left_unfold, right_unfold
from bsde_solver.tensor.tensor_core import TensorCore
from bsde_solver.tensor.tensor_network import TensorNetwork

import numpy as np


def retraction_operator(tt, i):
    operator = tt.extract([f"core_{j}" for j in range(tt.order) if j != i])
    return operator

def scalar_ALS(X, phis: list[TensorCore], b: float, n_iter=10, ranks=None):
    tt = TensorTrain(X.shape, ranks).randomize()
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
        A += 0.1 * np.eye(A.shape[0])
        V *= b

        X = np.linalg.solve(A.view(np.ndarray), V.view(np.ndarray))
        X = TensorCore.like(X, tt.cores[f"core_{j}"])
        return X

    for i in range(n_iter):
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


def linear_ALS(
    b: list[TensorCore],
    n_iter=10,
    ranks=None,
):
    """Alternating Least cheme for Tensor train format.

    This implementation follows the work of Thorsten Rohwedder and Reinhold Schneider
    in their paper "The Alternating Linear Scheme for Tensor Optimization in the Tensor Train Format", 2011.

    Args:
        A (TensorNetwork): Tensor network representing the linear operator.
        b (TensorNetwork): Tensor network representing the right hand side.
        n_iter (int): Number of iterations to perform.
        ranks (list): List of ranks for the tensor train format.
    """

    shape = b[0].shape * len(b)
    tt = TensorTrain(shape, ranks).randomize()
    tt.orthonormalize(mode="right", start=1)

    b = TensorNetwork(cores=b)

    def micro_optimization(tt, j):
        P = retraction_operator(tt, j)
        V = TensorNetwork(cores=[P, b], names=["P", "b"]).contract(indices=(f"r_{j}", f"m_{j+1}", f"r_{j+1}"))
        return V

    for i in range(n_iter):
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





if __name__ == "__main__":
    from time import perf_counter

    degree = 4
    num_assets = 5

    shape = [degree for _ in range(num_assets)]
    ranks = [1, 3, 3, 3, 3, 1]
    tt = TensorTrain(shape, ranks)
    tt.randomize()

    def poly(x, degree=10):
        return np.array([x**i for i in range(degree)]).T

    def poly_derivative(x, degree=10):
        return np.array([i * x ** (i - 1) for i in range(degree)]).T

    n_simulations = 4
    phis, dphis = [], []

    # np.random.seed(4752)
    for i in range(n_simulations):
        x = np.random.rand(num_assets)*5

        phi = [TensorCore(poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",)) for i in range(tt.order)]
        dphi = [TensorCore(poly_derivative(x[i], degree=degree), name=f"dphi_{i+1}", indices=(f"m_{i+1}",),) for i in range(tt.order)]

        phis.append(phi)
        dphis.append(dphi)

    #################### Single ALS ####################

    start_time = perf_counter()
    min_tt = []
    b = -5
    for i in range(n_simulations):
        new_tt = scalar_ALS(tt, phis[i], b=b, n_iter=10, ranks=ranks)
        min_tt.append(new_tt)

        result = TensorNetwork(cores=[new_tt]+phis[i], names=["tt"]+[f"phi_{i+1}" for i in range(tt.order)]).contract().view(np.ndarray).squeeze()
        print("Reconstruction error:", np.linalg.norm(result - b))
    end_time = perf_counter() - start_time
    print("Time:", end_time)

    #################### Batch ALS ####################

    # phis, dphis = [], []
    # np.random.seed(0)
    # for i in range(n_simulations):
    #     x = np.random.rand(num_assets)

    #     phi = [poly(x[i], degree=degree) for i in range(tt.order)]
    #     dphi = [poly_derivative(x[i], degree=degree) for i in range(tt.order)]
    #     phis.append(phi)
    #     dphis.append(dphi)
    # phis = np.array(phis) # (n_simulations, tt.order, degree)
    # dphis = np.array(dphis) # (n_simulations, tt.order, degree)

    # phis = phis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    # dphis = dphis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    # phis = [
    #     TensorCore(
    #         phis[i], name=f"phi_{i+1}", indices=("batch", f"m_{i+1}")
    #     )
    #     for i in range(tt.order)
    # ]
    # dphis = [
    #     TensorCore(
    #         dphis[i], name=f"dphi_{i+1}", indices=("batch", f"m_{i+1}")
    #     )
    #     for i in range(tt.order)
    # ]

    # batch_tt = BatchTensorTrain.dupplicate(n_simulations, tt)

    # start_time_batch = perf_counter()
    # batch_derivatives = batch_derivative(batch_tt, phis, dphis)
    # end_time_batch = perf_counter() - start_time_batch
    # print("Time:", end_time_batch)

    # print("Speed up factor:", end_time / end_time_batch)

    # # Check if the results are the same
    # print(np.allclose(derivatives, batch_derivatives))
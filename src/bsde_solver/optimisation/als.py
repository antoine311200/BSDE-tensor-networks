from typing import Union

from bsde_solver.tensor.tensor_train import TensorTrain, left_unfold, right_unfold
from bsde_solver.tensor.tensor_core import TensorCore
from bsde_solver.tensor.tensor_network import TensorNetwork

import numpy as np


def retraction_operator(tt, i):
    operator = tt.extract([f"core_{j}" for j in range(tt.order) if j != i])
    return operator

def scalar_ALS(X, b: float, n_iter=10, ranks=None):
    tt = TensorTrain(X.shape, ranks).randomize()
    tt.orthonormalize(mode="right", start=1)

    def micro_optimization(tt, j):
        P = retraction_operator(tt, j)
        V = TensorNetwork(cores=[P, b], names=["P", "b"]).contract(indices=get_idx(j))
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
        print(V)
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
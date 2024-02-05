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
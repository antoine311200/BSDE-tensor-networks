from bsde_solver.core.tensor.tensor_train import BatchTensorTrain, TensorTrain, left_unfold, right_unfold
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.utils import batch_qr

import numpy as np
import sys

# def second_retraction_operator(tt, i):
#     operator = tt.extract([f"core_{j}" for j in range(tt.order) if j != i and j != i + 1])
#     return operator

def second_retraction_operator(tt, phis, j):
    results = []
    for i, (core, phi) in enumerate(zip(tt.cores.values(), phis.cores.values())):
        if i == j or i == j + 1:
            result = phi.copy()
        else:
            result = TensorNetwork(
                cores=[core, phi], names=[f"core_{i}", f"phi_{i}"]
            ).contract(indices=("batch", f"r_{i}", f"m_{i+1}", f"r_{i+1}"))
        results.append(result)

    V = TensorNetwork(cores=results, names=[f"V_{i}" for i in range(len(phis.cores))])
    V = V.contract(batch=True, indices=(f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"))
    print(V)
    return V

def MALS(phis: list[TensorCore], result: list[float], init_tt=None, n_iter=10, ranks=None):
    shape = tuple([phi.shape[1] for phi in phis])
    tt = init_tt if init_tt else TensorTrain(shape, ranks)
    tt.orthonormalize(mode="right", start=1)

    phis = TensorNetwork(cores=phis)
    result = TensorCore(np.array(result), name="result", indices=("batch",))

    def micro_optimization(tt, j):
        V = second_retraction_operator(tt, phis, j)
        # print(P)
        # V = TensorNetwork(cores=[P, phis], names=["P", "phi"])
        # print(f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}")
        # print(V)
        # V = V.contract(indices=(f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"))
        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")

        # The operator A can be non invertible, so we need to use some regularization
        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract()
        A = A.unfold((f"s_{j}", f"n_{j+1}", f"n_{j+2}", f"s_{j+2}"), (f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"))
        V = V.unfold((f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"))

        # Add regularization
        # A += 0.1 * np.random.rand(A.shape[1], A.shape[1])
        # V *= result
        Y = TensorNetwork(cores=[V, result], names=["V", "result"]).contract()


        print(A)

        # W = np.linalg.lstsq(A.view(np.ndarray), V.view(np.ndarray), rcond=None)[0]
        W = np.linalg.solve(A.view(np.ndarray), Y.view(np.ndarray))
        W = W.reshape(tt.ranks[j] * tt.shape[j], tt.shape[j+1] * tt.ranks[j+2])

        return W

    for _ in range(n_iter):
        print("Left half sweep")
        # Left half sweep
        for j in range(tt.order - 1):
            # Micro optimization
            W = micro_optimization(tt, j)

            Q, R = np.linalg.qr(W)
            Q = Q[:, :tt.ranks[j+1]]
            R = R[:tt.ranks[j+1], :]

            Y = TensorCore.like(Q, tt.cores[f"core_{j}"])
            Z = TensorCore.like(R, tt.cores[f"core_{j+1}"])

            tt.cores[f"core_{j}"] = Y
            tt.cores[f"core_{j+1}"] = Z
        print("Rigth half sweep")
        # Right half sweep
        for j in range(tt.order - 1, 0, -1):
            # Micro optimization
            W = micro_optimization(tt, j-1)

            Q, R = np.linalg.qr(W.T)

            Q = Q[:, :tt.ranks[j]]
            R = R[:tt.ranks[j], :]

            Y = TensorCore.like(R.T, tt.cores[f"core_{j-1}"])
            Z = TensorCore.like(Q.T, tt.cores[f"core_{j}"])

            tt.cores[f"core_{j-1}"] = Y
            tt.cores[f"core_{j}"] = Z

    return tt
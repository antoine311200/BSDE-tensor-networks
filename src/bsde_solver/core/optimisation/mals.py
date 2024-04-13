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
    return V

def MALS(phis: list[TensorCore], result: list[float], init_tt=None, n_iter=10, ranks=None, threshold=0.5):
    shape = tuple([phi.shape[1] for phi in phis])
    tt = init_tt if init_tt else TensorTrain(shape, ranks)
    tt.orthonormalize(mode="right", start=1)
    print(tt)
    phis = TensorNetwork(cores=phis)
    result = TensorCore(np.array(result), name="result", indices=("batch",))

    def micro_optimization(tt, j):
        V = second_retraction_operator(tt, phis, j)

        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")

        # The operator A can be non invertible, so we need to use some regularization
        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract()
        A = A.unfold((f"s_{j}", f"n_{j+1}", f"n_{j+2}", f"s_{j+2}"), (f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"))

        V = V.unfold("batch", (f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"))

        # Add regularization
        # A += 0.1 * np.random.rand(A.shape[1], A.shape[1])
        # V *= result
        Y = TensorNetwork(cores=[V, result], names=["V", "result"]).contract()

        # W = np.linalg.lstsq(A.view(np.ndarray), V.view(np.ndarray), rcond=None)[0]
        # W = np.linalg.solve(A.view(np.ndarray), Y.view(np.ndarray))
        # W = W.reshape(tt.ranks[j] * tt.shape[j], tt.shape[j+1] * tt.ranks[j+2])
        # print(Y)
        W = np.linalg.lstsq(A.view(np.ndarray), Y.view(np.ndarray), rcond=None)[0]
        W = W.reshape(tt.ranks[j] * tt.shape[j], tt.shape[j+1] * tt.ranks[j+2])

        return W

    for _ in range(n_iter):
        # print("Left half sweep")
        # Left half sweep
        for j in range(tt.order - 1):
            # Micro optimization
            # print("j", j)
            W = micro_optimization(tt, j)

            # Q, R = np.linalg.qr(W)
            # print(Q.shape, R.shape)
            # print(tt.ranks[j+1])
            # Q = Q[:, :tt.ranks[j+1]]
            # R = R[:tt.ranks[j+1], :]

            # Compute SVD
            U, S, V = np.linalg.svd(W)
            # print(U.shape, S.shape, V.shape)
            # print(S)

            # Compute the new rank as the number of singular values that are greater than a threshold
            new_rank = np.sum(S / S[0] > threshold) | 1
            # np.sum(S > 0.9 * np.sum(S)) | 1
            # tt.ranks[j+1]
            # np.sum(S > 0.9 * np.sum(S))

            Q = U[:, :new_rank]
            R = np.diag(S[:new_rank]) @ V[:new_rank, :]

            # print(tt.cores[f"core_{j}"])
            # print(TensorCore.dummy((tt.ranks[j], tt.shape[j], new_rank), indices=tt.cores[f"core_{j}"].indices))
            # print(tt.ranks)
            tt.ranks = tt.ranks[:j+1] + (new_rank,) + tt.ranks[j+2:]
            # print(tt.ranks)
            dummy_core1 = TensorCore.dummy((tt.ranks[j], tt.shape[j], new_rank), indices=tt.cores[f"core_{j}"].indices)
            dummy_core2 = TensorCore.dummy((new_rank, tt.shape[j+1], tt.ranks[j+2]), indices=tt.cores[f"core_{j+1}"].indices)

            # print(tt.cores[f"core_{j}"])
            # print(tt.cores[f"core_{j+1}"])
            # print(dummy_core1)
            # print(dummy_core2)
            # print(Q.shape, R.shape)

            Y = TensorCore.like(Q, dummy_core1)
            Z = TensorCore.like(R, dummy_core2)


            # Y = TensorCore.like(Q, tt.cores[f"core_{j}"])
            # Z = TensorCore.like(R, tt.cores[f"core_{j+1}"])

            tt.cores[f"core_{j}"] = Y
            tt.cores[f"core_{j+1}"] = Z
        # print("Rigth half sweep")
        # print(tt)
        # Right half sweep
        for j in range(tt.order - 2, 0, -1):
            # Micro optimization
            # print("k", j)
            W = micro_optimization(tt, j-1)
            # print(W.shape)

            U, S, V = np.linalg.svd(W.T)
            # print(U.shape, S.shape, V.shape)

            # Compute the new rank as the number of singular values that are greater than a threshold
            new_rank = np.sum(S / S[0] > threshold) | 1

            Q = U[:, :new_rank]
            R = np.diag(S[:new_rank]) @ V[:new_rank, :]

            # tt.ranks = tt.ranks[:j] + (new_rank,) + tt.ranks[j+1:]

            # print(R.T.shape, Q.T.shape)

            # print(tt.cores[f"core_{j-1}"])
            # print(tt.cores[f"core_{j}"])

            dummy_core1 = TensorCore.dummy((tt.ranks[j-1], tt.shape[j], new_rank), indices=tt.cores[f"core_{j-1}"].indices)
            dummy_core2 = TensorCore.dummy((new_rank, tt.shape[j], tt.ranks[j+1]), indices=tt.cores[f"core_{j}"].indices)

            # print(dummy_core1)
            # print(dummy_core2)

            Y = TensorCore.like(R.T, dummy_core1)
            Z = TensorCore.like(Q.T, dummy_core2)

            # Q, R = np.linalg.qr(W.T)

            # Q = Q[:, :tt.ranks[j]]
            # R = R[:tt.ranks[j], :]

            # Y = TensorCore.like(R.T, tt.cores[f"core_{j-1}"])
            # Z = TensorCore.like(Q.T, tt.cores[f"core_{j}"])

            # tt.cores[f"core_{j-1}"] = Y
            # tt.cores[f"core_{j}"] = Z

    return tt
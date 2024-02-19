from bsde_solver.core.tensor.tensor_train import BatchTensorTrain, TensorTrain, left_unfold, right_unfold
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
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
        # A += 0.1 * np.stack([
        #     np.random.rand(A.shape[1], A.shape[1])
        #     for _ in range(batch_size)
        # ])
        V *= result

        # X = np.linalg.solve(A.view(np.ndarray), V.view(np.ndarray))
        X = np.zeros_like(V.view(np.ndarray))
        for i in range(batch_size):
            X[i] = np.linalg.lstsq(A[i].view(np.ndarray), V[i].view(np.ndarray), rcond=None)[0]

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


def multi_als(phis: list[TensorCore], result: list[float], n_iter=10, ranks=None, eta=0.0001, init_tt=None):
    shape = tuple([phi.shape[1] for phi in phis])

    if init_tt:
        tt = init_tt
    else:
        tt = TensorTrain(shape, ranks).randomize()
    tt.orthonormalize(mode="right", start=1)

    phis = TensorNetwork(cores=phis)
    result = TensorCore(np.array(result), name="result", indices=("batch", ))

    def micro_optimization(tt, j):
        P = retraction_operator(tt, j)
        # print("P", P)
        # print("phis", phis)

        # import time

        # start = time.time()
        V = TensorNetwork(cores=[P, phis], names=["P", "phi"]).contract(batch=True, indices=(f"r_{j}", f"m_{j+1}", f"r_{j+1}"))
        # # print(time.time() - start)
        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")

        # unbatch_phis = []
        # for i in range(phis[0].shape_info[0]):
        #     cores = {}
        #     for k in range(len(phis.cores.keys())):
        #         indices = phis.cores[f"phi_{k+1}"].indices
        #         indices = [index for index in indices if index != "batch"]
        #         cores[f"core_{k+1}"] = TensorCore(phis.cores[f"phi_{k+1}"][i], name=f"phi_{k+1}", indices=indices)
        #     unphi = TensorNetwork(cores=cores)
        #     unbatch_phis.append(unphi)

        # # start = time.time()
        # contr = []
        # for i in range(phis[0].shape_info[0]):
        #     contr.append(TensorNetwork(cores=[P, unbatch_phis[i]], names=["P", "phi"]).contract(indices=(f"r_{j}", f"m_{j+1}", f"r_{j+1}")))
        # # print(time.time() - start)

        # V = TensorCore.concatenate(contr)
        # V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")
        # print(V)

        # import sys
        # sys.exit()

        # The operator A can be non invertible, so we need to use some regularization
        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract()
        A = A.unfold((f"s_{j}", f"n_{j+1}", f"s_{j+1}"), (f"r_{j}", f"m_{j+1}", f"r_{j+1}"))
        V = V.unfold("batch", (f"r_{j}", f"m_{j+1}", f"r_{j+1}"))

        Y = TensorNetwork(cores=[V, result], names=["V", "result"]).contract()

        # Add regularization term as eta * I

        A += eta * np.eye(A.shape[1])

        # X = np.linalg.solve(A.view(np.ndarray), Y.view(np.ndarray))
        X = np.linalg.lstsq(A.view(np.ndarray), Y.view(np.ndarray), rcond=None)[0]
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
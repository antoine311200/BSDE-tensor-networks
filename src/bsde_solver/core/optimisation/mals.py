from bsde_solver.core.optimisation.solve import solver
from bsde_solver.core.tensor.tensor_train import (
    TensorTrain,
    left_unfold,
    right_unfold,
)
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork

from bsde_solver import xp
from scipy import linalg


def second_retraction_operator(tt, phis, j):
    results = []
    for i, (core, phi) in enumerate(zip(tt.cores.values(), phis.cores.values())):
        if i == j or i == j + 1:
            result = phi.copy()
        else:
            result = TensorNetwork(
                cores=[core, phi], names=[f"core_{i}", f"phi_{i}"]
            ).contract(indices=("batch", f"r_{i}", f"r_{i+1}"))
        results.append(result)

    V = TensorNetwork(cores=results, names=[f"V_{i}" for i in range(len(phis.cores))])
    V = V.contract(batch=True, indices=(f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"))
    return V


def MALS(
    phis: list[TensorCore],
    result: list[float],
    init_tt=None,
    n_iter=10,
    ranks=None,
    threshold=1e-3,
    max_rank=8,
    optimizer="lstsq",
):
    shape = tuple([phi.shape[1] for phi in phis])
    tt = init_tt if init_tt else TensorTrain(shape, ranks)
    tt.randomize()
    tt.orthonormalize(mode="right", start=1)

    phis = TensorNetwork(cores=phis)
    result = TensorCore(xp.array(result), name="result", indices=("batch",))

    max_ranks = [
        xp.minimum(tt.ranks[i] * tt.shape[i], max_rank) for i in range(tt.order - 1)
    ]
    max_ranks[-1] = xp.minimum(tt.ranks[-1] * tt.shape[-1], max_rank)

    def micro_optimization(tt, j):
        V = second_retraction_operator(tt, phis, j)
        # V = V.unfold(f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}", "batch")
        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")

        # The operator A can be non invertible, so we need to use some regularization
        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract()
        A = A.unfold(
            (f"s_{j}", f"n_{j+1}", f"n_{j+2}", f"s_{j+2}"),
            (f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"),
        )
        V = V.unfold("batch", (f"r_{j}", f"m_{j+1}", f"m_{j+2}", f"r_{j+2}"))

        Y = TensorNetwork(cores=[V, result], names=["V", "result"]).contract()
        W = solver(A, Y, method=optimizer)
        dummy_core = TensorCore.dummy(
            (tt.ranks[j] * tt.shape[j], tt.shape[j + 1] * tt.ranks[j + 2]),
            indices=(f"r_{j}+m_{j+1}", f"m_{j+2}+r_{j+2}"),
        )
        W = TensorCore.like(W, dummy_core)
        return W

    for _ in range(n_iter):
        # print("Left sweep")
        # Left half sweep
        for j in range(tt.order - 2):
            # Micro optimization
            W = micro_optimization(tt, j)

            # Compute SVD
            U, S, V = xp.linalg.svd(W)

            # Compute the new rank as the number of singular values that are greater than a threshold
            new_rank = xp.minimum(xp.sum(S / S[0] > threshold), max_ranks[j])
            if new_rank != tt.ranks[j + 1]:
                print(j, f"Rank adaptation: {tt.ranks[j+1]} -> {new_rank}")
                tt.ranks = tt.ranks[: j + 1] + (new_rank,) + tt.ranks[j + 2 :]

            Q = U[:, :new_rank]
            R = xp.diag(S[:new_rank]) @ V[:new_rank, :]

            dummy_core1 = TensorCore.dummy(
                (tt.ranks[j], tt.shape[j], new_rank),
                indices=tt.cores[f"core_{j}"].indices,
            )
            dummy_core2 = TensorCore.dummy(
                (new_rank, tt.shape[j + 1], tt.ranks[j + 2]),
                indices=tt.cores[f"core_{j+1}"].indices,
            )

            Y = TensorCore.like(Q, dummy_core1)
            Z = TensorCore.like(R, dummy_core2)

            tt.cores[f"core_{j}"] = Y
            tt.cores[f"core_{j+1}"] = Z

            # Q, R = xp.linalg.qr(W)
            # Q = Q[:, : tt.ranks[j + 1]]
            # R = R[: tt.ranks[j + 1], :]
            # tt.cores[f"core_{j}"] = TensorCore.like(Q, tt.cores[f"core_{j}"])
            # tt.cores[f"core_{j+1}"] = TensorCore.like(R, tt.cores[f"core_{j+1}"])

        # print("Right sweep")
        # Right half sweep
        for j in range(tt.order - 1, 1, -1):
            # Micro optimization
            W = micro_optimization(tt, j - 1)

            U, S, V = xp.linalg.svd(W.T)

            # Compute the new rank as the number of singular values that are greater than a threshold
            new_rank = xp.minimum(xp.sum(S / S[0] > threshold), max_ranks[j - 1])

            if new_rank != tt.ranks[j]:
                print(j, f"Rank adaptation: {tt.ranks[j]} -> {new_rank}")
                tt.ranks = tt.ranks[:j] + (new_rank,) + tt.ranks[j + 1 :]

            Q = U[:, :new_rank]
            R = xp.diag(S[:new_rank]) @ V[:new_rank, :]

            dummy_core1 = TensorCore.dummy(
                (tt.ranks[j - 1], tt.shape[j], new_rank),
                indices=tt.cores[f"core_{j-1}"].indices,
            )
            dummy_core2 = TensorCore.dummy(
                (new_rank, tt.shape[j], tt.ranks[j + 1]),
                indices=tt.cores[f"core_{j}"].indices,
            )

            Y = TensorCore.like(R.T, dummy_core1)
            Z = TensorCore.like(Q.T, dummy_core2)

            tt.cores[f"core_{j-1}"] = Y
            tt.cores[f"core_{j}"] = Z

            # Q, R = xp.linalg.qr(W.T)
            # Q = Q[:, : tt.ranks[j]]
            # R = R[: tt.ranks[j], :]
            # tt.cores[f"core_{j-1}"] = TensorCore.like(R.T, tt.cores[f"core_{j-1}"])
            # tt.cores[f"core_{j}"] = TensorCore.like(Q.T, tt.cores[f"core_{j}"])

    return tt

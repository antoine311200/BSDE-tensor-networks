from bsde_solver.core.optimisation.solve import solver
from bsde_solver.core.tensor.tensor_train import TensorTrain, left_unfold, right_unfold
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork

import numpy as np
import time


def retraction(tt, j):
    return tt.extract([f"core_{i}" for i in range(tt.order) if i != j])


def retraction_operator(tt, phis, j):
    results = []
    for i, (core, phi) in enumerate(zip(tt.cores.values(), phis.cores.values())):
        if i == j:
            result = phi.copy()
        else:
            result = TensorNetwork(
                cores=[core, phi], names=[f"core_{i}", f"phi_{i}"]
            )
            # print("R", result)
            result = result.contract(indices=("batch", f"r_{i}", f"r_{i+1}"))#, f"m_{i+1}"
        results.append(result)

    V = TensorNetwork(cores=results, names=[f"V_{i}" for i in range(len(phis.cores))])
    V = V.contract(batch=True, indices=(f"r_{j}", f"m_{j+1}", f"r_{j+1}"))
    return V


def ALS(
    phis: list[TensorCore],
    result: list[float],
    n_iter=10,
    ranks=None,
    optimizer="lstsq",
    init_tt=None,
):
    shape = tuple([phi.shape[1] for phi in phis])

    tt = init_tt if init_tt else TensorTrain(shape, ranks)
    tt.randomize()
    tt.orthonormalize(mode="right", start=1)

    phis = TensorNetwork(cores=phis)
    result = TensorCore(np.array(result), name="result", indices=("batch",))

    def micro_optimization(tt, j):
        V = retraction_operator(tt, phis, j)
        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")

        # The operator A can be non invertible, so we need to use some regularization
        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract()
        A = A.unfold(
            (f"s_{j}", f"n_{j+1}", f"s_{j+1}"), (f"r_{j}", f"m_{j+1}", f"r_{j+1}")
        )
        V = V.unfold("batch", (f"r_{j}", f"m_{j+1}", f"r_{j+1}"))

        Y = TensorNetwork(cores=[V, result], names=["V", "result"]).contract()

        X = solver(A, Y, optimizer)
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


def SALSA(
    phis: list[TensorCore],
    result: list[float],
    n_iter=10,
    ranks=None,
    omega=1.0,
    freq_omega=1.05,
    min_sv=0.2,
    init_tt=None,
    max_rank=10,
    optimizer="lstsq",
    do_reg: bool = True
):
    shape = tuple([phi.shape[1] for phi in phis])

    tt = init_tt if init_tt else TensorTrain(shape, ranks)  # .randomize()
    tt.randomize()

    phis = TensorNetwork(cores=phis)
    result = TensorCore(np.array(result), name="result", indices=("batch",))

    eta = 1e-6
    max_ranks = [np.minimum(tt.ranks[i] * tt.shape[i], max_rank) for i in range(tt.order - 1)]
    max_ranks[-1] = np.minimum(tt.ranks[-1] * tt.shape[-1], max_rank)

    def micro_optimization(tt, phis, result, j, gamma, theta, eta, start=False):
        V = retraction_operator(tt, phis, j)
        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")
        R = V.copy()

        core_shape = [
            V.size(f"r_{j}"),
            V.size(f"m_{j+1}"),
            V.size(f"r_{j+1}"),
        ]
        axes = (f"r_{j}", f"m_{j+1}", f"r_{j+1}")
        axes_T = (f"s_{j}", f"n_{j+1}", f"s_{j+1}")

        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract()
        A = A.unfold(axes_T, axes)
        V = V.unfold("batch", axes)

        # The operator A can be non invertible, so we need to use some regularization
        reg_term = np.zeros_like(A.view(np.ndarray))
        if gamma is not None and theta is not None:
            if j != 0:
                # print("Gamma:", gamma.shape)
                # print(core_shape)
                reg_term += eta**2 * np.einsum(
                    "ab,bc,ij,kl->aikcjl",
                    gamma,
                    gamma,
                    np.eye(core_shape[1]),
                    np.eye(core_shape[2]),
                ).reshape(A.shape_info)
            if j != tt.order - 1:
                # print("Theta:", theta.shape)
                # print(core_shape)
                reg_term += eta**2 * np.einsum(
                    "ab,bc,ij,kl->aikcjl",
                    theta,
                    theta,
                    np.eye(core_shape[1]),
                    np.eye(core_shape[0]),
                ).reshape(A.shape_info)

        A += reg_term

        if j == 0 and start:
            axes = (f"r_{j}", f"m_{j+1}", f"r_{j+1}")
            last_core = tt.cores[f"core_{j}"].copy().unfold(axes)
            eval = TensorNetwork(cores=[V, last_core], names=["V", f"core_{j}"]).contract().view(np.ndarray)
            eta = min(1/V.size("batch") * np.linalg.norm(eval - result)**2, 10000)

        A += eta * np.eye(A.shape[0])
        Y = TensorNetwork(cores=[V, result], names=["V", "result"]).contract()

        X = solver(A, Y, optimizer)
        X = TensorCore.like(X, tt.cores[f"core_{j}"])
        return X, V, R, eta

    def rank_adaptation(U, S, V, min_sv, j):
        S = np.append(S, 0.01 * min_sv)

        U = TensorCore.like(U, tt.cores[f"core_{j-1}"]).view(np.ndarray)
        U_comp = np.ones((U.shape[0], U.shape[1]))
        U_comp -= np.einsum("abc,dec,de->ab", U, U, U_comp)
        U_comp -= np.einsum("abc,dec,de->ab", U, U, U_comp)
        U_norm = np.linalg.norm(U_comp)
        if U_norm != 0:
            U_comp /= U_norm
            U = np.concatenate((U, U_comp[:, :, None]), axis=2)

        V = TensorCore.like(V, tt.cores[f"core_{j}"]).view(np.ndarray)
        V_comp = np.ones((V.shape[1], V.shape[2]))
        V_comp -= np.einsum("abc,ade,de->bc", V, V, V_comp)
        V_comp -= np.einsum("abc,ade,de->bc", V, V, V_comp)
        V_norm = np.linalg.norm(V_comp)
        if V_norm != 0:
            V_comp /= V_norm
            V = np.concatenate((V, V_comp[None, :, :]), axis=0)

        core_prev = TensorCore.dummy(U.shape, indices=tt.cores[f"core_{j-1}"].indices)
        core_curr = TensorCore.dummy(V.shape, indices=tt.cores[f"core_{j}"].indices)

        W = np.diag(S) @ V.reshape((V.shape[0], -1))

        return U, S, W, core_prev, core_curr

    expand_rank = False
    has_adapted = False
    warmup = 2
    last_adapt = 0

    errors = []
    omegas = []
    min_svs = []

    for it in range(n_iter):
        # Rank adaptation strategy
        if it >= last_adapt + warmup: expand_rank = True
        if has_adapted: expand_rank, has_adapted, last_adapt = False, False, it

        # print("---------------")
        # print("Iteration:", it)
        # print("Expand rank:", expand_rank)
        # print("Has adapted:", has_adapted)
        # print("Next adapt:", last_adapt)

        # Left half sweep
        tt.orthonormalize(mode="right", start=1)
        for j in range(tt.order):
            gamma, theta = None, None

            # print("Core:", j, core_curr)

            if j != 0:
                core_curr = tt.cores[f"core_{j}"]
                core_prev = tt.cores[f"core_{j-1}"]

                L = left_unfold(core_prev).view(np.ndarray)
                R = right_unfold(core_curr).view(np.ndarray)

                U, S, V = np.linalg.svd(L, full_matrices=False)
                W = S * V @ R
                if expand_rank and S[-1] > min_sv and S.shape[0] < max_ranks[j-1]:
                    # print("Rank adaptation")
                    G = V @ R
                    U, S, W, core_prev, core_curr = rank_adaptation(U, S, G, min_sv, j)
                    has_adapted = True
                # else:
                    # print("No rank adaptation")

                tt.cores[f"core_{j-1}"] = TensorCore.like(U, core_prev)
                tt.cores[f"core_{j}"] = TensorCore.like(W, core_curr)

                if do_reg:
                    gamma = np.diag(np.nan_to_num(1 / np.maximum(S, min_sv), posinf=1e-6, neginf=1e-6))
                # print(np.diagonal(gamma))

                # print("Gamma:", 1 / np.maximum(S, min_sv), S)
            if j != tt.order - 1:
                core_curr = tt.cores[f"core_{j}"]
                core_next = tt.cores[f"core_{j+1}"]

                L = left_unfold(core_curr).view(np.ndarray)
                R = right_unfold(core_next).view(np.ndarray)

                U, S, V = np.linalg.svd(L, full_matrices=False)
                W = U * S
                Z = V @ R

                tt.cores[f"core_{j}"] = TensorCore.like(W, core_curr)
                tt.cores[f"core_{j+1}"] = TensorCore.like(Z, core_next)

                if do_reg:
                    theta = np.diag(np.nan_to_num(1 / np.maximum(S, min_sv), posinf=1e-6, neginf=1e-6))

                # print("Theta:", 1 / np.maximum(S, min_sv), S)

                # print("Theta:", 1 / np.maximum(S, min_sv), S)

                # print(np.diagonal(theta))

            X, V, R, eta = micro_optimization(
                tt, phis, result, j, gamma, theta, eta, start=(it == 0)
            )
            tt.cores[f"core_{j}"] = X

        axes = (f"r_{j}", f"m_{j+1}", f"r_{j+1}")
        last_core = tt.cores[f"core_{j}"].copy().unfold(axes)

        eval = TensorNetwork(cores=[V, last_core], names=["V", f"core_{j}"]).contract().view(np.ndarray)
        eta_tmp = eta
        eta = 1/V.size("batch") * np.linalg.norm(eval - result)**2
        rel_eta = abs(eta_tmp - eta) / eta_tmp
        # const = TensorNetwork(cores=[R, X], names=["R", "X"]).contract().view(np.ndarray)
        omega = min(omega / freq_omega, max(eta, np.sqrt(eta)))
        omega = max(omega, rel_eta)
        min_sv = max(min(0.2*omega, 0.2*rel_eta), 0)

        # print("~" * 50)
        # print("Error:", eta)
        # print("Relative error:", rel_eta)
        # print("Omega:", omega)
        # print("Min SV:", min_sv)

        errors.append(eta)
        omegas.append(omega)
        min_svs.append(min_sv)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 15))
    # plt.subplot(3, 1, 1)
    # plt.plot(errors)
    # plt.title("Error")
    # plt.subplot(3, 1, 2)
    # plt.plot(omegas)
    # plt.title("Omega")
    # plt.subplot(3, 1, 3)
    # plt.plot(min_svs)
    # plt.title("Min SV")
    # plt.show()

    return tt

from bsde_solver.core.optimisation.solve import solver
from bsde_solver.core.tensor.tensor_train import TensorTrain, left_unfold, right_unfold
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork

from bsde_solver import xp

def retraction_operator(tt, phis, j):
    results = []
    for i, (core, phi) in enumerate(zip(tt.cores.values(), phis.cores.values())):
        if i == j:
            result = phi.copy()
        else:
            result = TensorNetwork(
                cores=[core, phi], names=[f"core_{i}", f"phi_{i}"]
            )
            result = result.contract(indices=("batch", f"r_{i}", f"r_{i+1}"))
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
    result = TensorCore(xp.array(result), name="result", indices=("batch",))

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

            L = left_unfold(V).view(xp.ndarray)
            R = right_unfold(core_next).view(xp.ndarray)

            Q, S = xp.linalg.qr(L)
            W = S @ R

            tt.cores[f"core_{j}"] = TensorCore.like(Q, core_curr)
            tt.cores[f"core_{j+1}"] = TensorCore.like(W, core_next)

        # Right half sweep
        for j in range(tt.order - 1, 0, -1):
            # Micro optimization
            V = micro_optimization(tt, j)

            core_prev = tt.cores[f"core_{j-1}"]
            core_curr = tt.cores[f"core_{j}"]

            L = left_unfold(core_prev).view(xp.ndarray)
            R = right_unfold(V).view(xp.ndarray)

            Q, S = xp.linalg.qr(R.T)
            W = L @ S.T

            tt.cores[f"core_{j-1}"] = TensorCore.like(W, core_prev)
            tt.cores[f"core_{j}"] = TensorCore.like(Q.T, core_curr)

    return tt


def ALS_regularized(
    phis: list[TensorCore],
    result: list[float],
    n_iter=10,
    ranks=None,
    omega=1.0,
    freq_omega=1.05,
    min_sv=0.2,
    init_tt=None,
):
    raise NotImplementedError("ALS_regularized is not implemented yet")

    shape = tuple([phi.shape[1] for phi in phis])

    tt = init_tt if init_tt else TensorTrain(shape, ranks)  # .randomize()
    tt.orthonormalize(mode="right", start=1)

    phis = TensorNetwork(cores=phis)
    result = TensorCore(xp.array(result), name="result", indices=("batch",))

    eta = 1e-6

    def micro_optimization(tt, phis, result, j, gamma, theta):
        V = retraction_operator(tt, phis, j)
        V_T = V.copy().rename("r_*", "s_*").rename("m_*", "n_*")

        core_shape = [
            V.size_at(f"r_{j}"),
            V.size_at(f"m_{j+1}"),
            V.size_at(f"r_{j+1}"),
        ]
        axes = (f"r_{j}", f"m_{j+1}", f"r_{j+1}")
        axes_T = (f"s_{j}", f"n_{j+1}", f"s_{j+1}")

        A = TensorNetwork(cores=[V_T, V], names=["V_T", "V"]).contract()
        A = A.unfold(axes_T, axes)
        V = V.unfold("batch", axes)

        # The operator A can be non invertible, so we need to use some regularization
        reg_term = xp.zeros_like(A.view(xp.ndarray))
        if j != 0:
            # print("Gamma:", gamma.shape)
            # print(core_shape)
            reg_term += eta**2 * xp.einsum(
                "ab,bc,ij,kl->aikcjl",
                gamma,
                gamma,
                xp.eye(core_shape[1]),
                xp.eye(core_shape[2]),
            ).reshape(A.shape_info)
        if j != tt.order - 1:
            # print("Theta:", theta.shape)
            # print(core_shape)
            reg_term += eta**2 * xp.einsum(
                "ab,bc,ij,kl->aikcjl",
                theta,
                theta,
                xp.eye(core_shape[1]),
                xp.eye(core_shape[0]),
            ).reshape(A.shape_info)

        A += reg_term
        Y = TensorNetwork(cores=[V, result], names=["V", "result"]).contract()

        X = xp.linalg.lstsq(A.view(xp.ndarray), Y.view(xp.ndarray), rcond=None)[0]
        X = TensorCore.like(X, tt.cores[f"core_{j}"])
        return X, V

    for _ in range(n_iter):
        # Left half sweep
        for j in range(tt.order - 1):
            core_curr = tt.cores[f"core_{j}"]
            gamma, theta = None, None

            # print("Core:", j, core_curr)

            if j != 0:
                core_prev = tt.cores[f"core_{j-1}"]

                L = left_unfold(core_prev).view(xp.ndarray)
                R = right_unfold(core_curr).view(xp.ndarray)

                U, S, V = xp.linalg.svd(L, full_matrices=False)
                W = S * V @ R

                tt.cores[f"core_{j-1}"] = TensorCore.like(U, core_prev)
                tt.cores[f"core_{j}"] = TensorCore.like(W, core_curr)
                gamma = xp.diag(1 / xp.maximum(S, min_sv))

            if j != tt.order - 1:
                core_next = tt.cores[f"core_{j+1}"]

                L = left_unfold(core_curr).view(xp.ndarray)
                R = right_unfold(core_next).view(xp.ndarray)

                U, S, V = xp.linalg.svd(L, full_matrices=False)
                W = U * S
                Z = V @ R

                tt.cores[f"core_{j}"] = TensorCore.like(W, core_curr)
                tt.cores[f"core_{j+1}"] = TensorCore.like(Z.T, core_next)
                theta = xp.diag(1 / xp.maximum(S, min_sv))

            # print("Gamma:", gamma)
            # print("Theta:", theta)

            tt.cores[f"core_{j}"], V = micro_optimization(
                tt, phis, result, j, gamma, theta
            )

        # Right half sweep
        # for j in range(tt.order - 1, 0, -1):
        #     core_curr = tt.cores[f"core_{j}"]
        #     gamma, theta = None, None

        #     print("Core:", j, core_curr)

        #     if j != tt.order - 1:
        #         core_next = tt.cores[f"core_{j+1}"]

        #         L = left_unfold(core_curr).view(xp.ndarray)
        #         R = right_unfold(core_next).view(xp.ndarray)

        #         U, S, V = xp.linalg.svd(R.T, full_matrices=False)
        #         W = L @ U
        #         Z = S * V

        #         tt.cores[f"core_{j}"] = TensorCore.like(W, core_curr)
        #         tt.cores[f"core_{j+1}"] = TensorCore.like(Z.T, core_next)
        #         gamma = xp.diag(1 / xp.maximum(S, min_sv))

        #     if j != 0:
        #         core_prev = tt.cores[f"core_{j-1}"]

        #         L = left_unfold(core_prev).view(xp.ndarray)
        #         R = right_unfold(core_curr).view(xp.ndarray)

        #         U, S, V = xp.linalg.svd(R.T, full_matrices=False)
        #         W = L @ U
        #         Z = S * V

        #         tt.cores[f"core_{j-1}"] = TensorCore.like(W, core_prev)
        #         tt.cores[f"core_{j}"] = TensorCore.like(Z.T, core_curr)
        #         theta = xp.diag(1 / xp.maximum(S, min_sv))

        #     print("Gamma:", gamma)
        #     print("Theta:", theta)

        #     tt.cores[f"core_{j}"] = micro_optimization(
        #         tt, phis, result, j, gamma, theta
        #     )

        axes = (f"r_{j}", f"m_{j+1}", f"r_{j+1}")
        last_core = tt.cores[f"core_{j}"].copy().unfold(axes)

        eval = TensorNetwork(cores=[V, last_core], names=["V", f"core_{j}"]).contract().view(xp.ndarray)
        eta = 1/V.size("batch") * xp.linalg.norm(eval)**2
        omega = max(min(omega / freq_omega, max(eta, xp.sqrt(eta))), (omega - eta) / omega)
        min_sv = max(min(0.2*omega, 0.2*(omega - eta) / omega), 0)

        # print("Error:", eta, eval.shape)
        # print("Omega:", omega)
        # print("Min SV:", min_sv)

    return tt

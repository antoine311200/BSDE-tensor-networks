from bsde_solver import reload_backend
reload_backend("numpy")
from bsde_solver import xp

import matplotlib.pyplot as plt
import time

from bsde_solver.bsde import BackwardSDE, HJB, DoubleWellHJB, BlackScholes, AllenCahn
from bsde_solver.stochastic.path import generate_trajectories
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.calculus.derivative import multi_derivative
from bsde_solver.core.calculus.hessian import hessian
from bsde_solver.core.optimisation.als import SALSA
from bsde_solver.core.calculus.basis import PolynomialBasis
from bsde_solver.loss import PDELoss
from bsde_solver.utils import fast_contract, flatten

from dataclasses import dataclass

@dataclass
class Parameters:
    batch_size: int
    T: int
    N: int
    num_assets: int
    n_iter: int
    rank: int
    degree: int
    n_iter_implicit: int

    @property
    def dt(self):
        return self.T / self.N
    
    @property
    def ranks(self):
        return (1,) + (self.rank,) * (self.num_assets - 1) + (1,)

def initialize(params):
    xo = xp.array(flatten([(.5, 1.) for _ in range(params.num_assets//2)])) # Black-Scholes initial condition
    X0 = xp.tile(xo, (params.batch_size, 1))

    model = BlackScholes(X0, params.dt, params.T, 0.05, 0.4)

    # Compute trajectories
    X, noise = generate_trajectories(X0, params.N, model) # (batch_size, N + 1, dim), (batch_size, N + 1, dim) (xi[0] is not used)

    phi_X = []
    dphi_X = []

    basis = PolynomialBasis(params.degree)
    # Compute phi_X and dphi_X for each time step
    for n in range(params.N + 1):
        phi_X_n = [TensorCore(basis.eval(X[:, n, i]), name=f"phi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(params.num_assets)]
        dphi_X_n = [TensorCore(basis.grad(X[:, n, i]), name=f"dphi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(params.num_assets)]
        phi_X.append(phi_X_n)
        dphi_X.append(dphi_X_n)

    return X, xo, phi_X, dphi_X, noise, model

def run_explicit(X, xo, phi_X, dphi_X, noise, model, params):
    batch_size = params.batch_size
    N = params.N
    dt = params.dt
    n_iter = params.n_iter
    degree = params.degree
    ranks = params.ranks

    Y = xp.zeros((batch_size, N + 1))
    Y[:, -1] = model.g(X[:, -1])  # (batch_size, )

    start_time = time.perf_counter()
    V = [None for _ in range(N + 1)]
    V_N = SALSA(phi_X[-1], Y[:, -1], n_iter=n_iter, ranks=ranks, max_rank=degree)
    V[-1] = V_N

    check_V = fast_contract(V_N, phi_X[-1])
    error = check_V - Y[:, -1]

    step_times = []
    mean_relative_errors = []
    max_relative_errors = []

    mean_relative_errors.append(xp.mean(xp.abs(error / Y[:, -1])))
    max_relative_errors.append(xp.max(xp.abs(error / Y[:, -1])))

    for n in range(N - 1, -1, -1):
        step_start_time = time.perf_counter()
        V_n1 = V[n + 1]
        Y_n1 = Y[:, n + 1]  # (batch_size, )

        X_n1 = X[:, n + 1, :]  # (batch_size, num_assets)
        phi_X_n1 = phi_X[n+1]  # tensor core of shape (batch_size, degree) * num_assets
        dphi_X_n1 = dphi_X[n+1]  # tensor core of shape (batch_size, degree) * num_assets

        X_n = X[:, n, :]  # (batch_size, num_assets)
        phi_X_n = phi_X[n]  # tensor core of shape (batch_size, degree) * num_assets
        dphi_X_n = dphi_X[n]  # tensor core of shape (batch_size, degree) * num_assets

        grad_Vn1 = multi_derivative(V_n1, phi_X_n1, dphi_X_n1)  # (batch_size, num_assets)
        sigma_n1 = model.sigma(X_n1, (n+1) * dt)  # (batch_size, num_assets, num_assets)
        Z_n1 = xp.einsum('ijk, ik -> ij', sigma_n1, grad_Vn1)  # (batch_size, num_assets)
        h_n1 = model.h(X_n1, (n+1) * dt, Y_n1, Z_n1)  # (batch_size, )

        step_n1 = h_n1*dt + Y_n1 # (batch_size, )
        if n == 0:
            V_n = SALSA(phi_X_n, step_n1, n_iter=n_iter, ranks=ranks, init_tt=V_n1, max_rank=degree, do_reg=False)
        else:
            V_n = SALSA(phi_X_n, step_n1, n_iter=n_iter, ranks=ranks, init_tt=V_n1, max_rank=degree)
        V[n] = V_n
        Y_n = fast_contract(V_n, phi_X_n)
        Y[:, n] = Y_n

        step_times.append(time.perf_counter() - step_start_time)
        ground_prices = model.price(X_n, n*dt) # (batch_size, )
        relative_errors = xp.abs((Y_n - ground_prices) / ground_prices)
        mean_relative_errors.append(xp.mean(relative_errors))
        max_relative_errors.append(xp.max(relative_errors))
        print(f"Step: {N - n}/{N} : Mean relative error: {xp.mean(relative_errors):.2f}, Max relative error: {xp.max(relative_errors):.2f}")

    total_time = time.perf_counter() - start_time
    ground_truth = model.price(xo[None, :], 0, n_sims=50_000).item()
    error = xp.abs((Y[0, 0] - ground_truth) / ground_truth)

    return step_times, mean_relative_errors, max_relative_errors, error, total_time

def run_implicit(X, xo, phi_X, dphi_X, noise, model, params):
    batch_size = params.batch_size
    N = params.N
    dt = params.dt
    n_iter = params.n_iter
    degree = params.degree
    ranks = params.ranks
    n_iter_implicit = params.n_iter_implicit

    Y = xp.zeros((batch_size, N + 1))
    Y[:, -1] = model.g(X[:, -1])  # (batch_size, )

    start_time = time.perf_counter()
    V = [None for _ in range(N + 1)]
    V_N = SALSA(phi_X[-1], Y[:, -1], n_iter=n_iter, ranks=ranks, max_rank=degree)
    V[-1] = V_N

    check_V = fast_contract(V_N, phi_X[-1])
    error = check_V - Y[:, -1]

    step_times = []
    mean_relative_errors = []
    max_relative_errors = []

    mean_relative_errors.append(xp.mean(xp.abs(error / Y[:, -1])))
    max_relative_errors.append(xp.max(xp.abs(error / Y[:, -1])))

    for n in range(N - 1, -1, -1):
        step_start_time = time.perf_counter()
        V_n1 = V[n + 1]
        Y_n1 = Y[:, n + 1]  # (batch_size, )

        X_n = X[:, n, :]  # (batch_size, num_assets)
        phi_X_n = phi_X[n]  # tensor core of shape (batch_size, degree) * num_assets
        dphi_X_n = dphi_X[n]  # tensor core of shape (batch_size, degree) * num_assets

        noise_n1 = noise[:, n + 1, :]  # (batch_size, num_assets)
        
        V_nk = V_n1
        Y_nk = Y_n1
        for k in range(n_iter_implicit):
            grad_Vnk = multi_derivative(V_nk, phi_X_n, dphi_X_n) # (batch_size, num_assets)
            sigma_nk = model.sigma(X_n, n*dt) # (batch_size, num_assets, num_assets)
            Z_nk = xp.einsum('ijk, ik -> ij', sigma_nk, grad_Vnk) # (batch_size, num_assets)
            h_nk = model.h(X_n, n*dt, Y_nk, Z_nk)

            step_nk = h_nk*dt + Y_n1 - xp.sqrt(dt) * xp.einsum('ij,ij->i', Z_nk, noise_n1)

            if n == 0:
                V_nk = SALSA(phi_X_n, step_nk, n_iter=n_iter, ranks=ranks, init_tt=V_nk, max_rank=degree, do_reg=False)
            else:
                V_nk = SALSA(phi_X_n, step_nk, n_iter=n_iter, ranks=ranks, init_tt=V_nk, max_rank=degree)
            Y_nk = fast_contract(V_nk, phi_X_n)

        V[n] = V_nk
        Y[:, n] = Y_nk

        step_times.append(time.perf_counter() - step_start_time)

        ground_prices = model.price(X_n, n*dt) # (batch_size, )
        relative_errors = xp.abs((Y_nk - ground_prices) / ground_prices)
        mean_relative_errors.append(xp.mean(relative_errors))
        max_relative_errors.append(xp.max(relative_errors))
        print(f"Step: {N - n}/{N} : Mean relative error: {xp.mean(relative_errors):.2f}, Max relative error: {xp.max(relative_errors):.2f} | {time.perf_counter() - step_start_time:.2f}s")


    total_time = time.perf_counter() - start_time
    ground_truth = model.price(xo[None, :], 0, n_sims=50_000).item()
    error = xp.abs((Y[0, 0] - ground_truth) / ground_truth)

    return step_times, mean_relative_errors, max_relative_errors, error, total_time

if __name__ == "__main__":
    n_iters = [1, 2, 5, 10, 25, 50] 

    errors_explicit = []
    errors_implicit = []
    for n_iter in n_iters:
        params = Parameters(
            batch_size = 2000,
            T = 1,
            N = 100,
            num_assets = 4,

            n_iter=50,
            n_iter_implicit=n_iter,
            degree=3,
            rank=3,
        )

        X, xo, phi_X, dphi_X, noise, model = initialize(params)
        
        step_times_explicit, mean_relative_errors_explicit, max_relative_errors_explicit, error_explicit, total_time_explicit = run_explicit(X, xo, phi_X, dphi_X, noise, model, params)
        step_times_implicit, mean_relative_errors_implicit, max_relative_errors_implicit, error_implicit, total_time_implicit = run_implicit(X, xo, phi_X, dphi_X, noise, model, params)
        print(f"Number of implicit iterations: {n_iter}, Explicit time: {total_time_explicit:.2f}s, Implicit time: {total_time_implicit:.2f}s")
        print(f"Explicit error: {error_explicit:.2f}, Implicit error: {error_implicit:.2f}")

        errors_explicit.append(error_explicit)
        errors_implicit.append(error_implicit)

    plt.plot(n_iters, errors_explicit, label="Explicit")
    plt.plot(n_iters, errors_implicit, label="Implicit")
    plt.xlabel("Number of implicit iterations")
    plt.ylabel("Error")
    plt.title("Error comparison between explicit and implicit schemes for different number of implicit iterations")
    plt.legend()
    plt.savefig("n_iter_error_comparison.png")
    plt.show()

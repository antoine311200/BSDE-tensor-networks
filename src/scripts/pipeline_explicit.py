import numpy as np

import matplotlib.pyplot as plt

from bsde_solver.bsde import BackwardSDE, HJB, DoubleWellHJB, BlackScholes, AllenCahn
from bsde_solver.stochastic.path import generate_trajectories
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.tensor.tensor_train import BatchTensorTrain
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.calculus.derivative import batch_derivative, multi_derivative
from bsde_solver.core.calculus.hessian import hessian
from bsde_solver.core.optimisation.als import ALS
from bsde_solver.core.calculus.basis import LegendreBasis, PolynomialBasis
from bsde_solver.loss import PDELoss
from bsde_solver.utils import flatten, fast_contract

import time

batch_size = 2000
T = 1
N = 10
num_assets = 10
dt = T / N

n_iter = 10
n_iter_implicit = 10
rank = 2
degree = 3
shape = tuple([degree for _ in range(num_assets)])
ranks = (1,) + (rank,) * (num_assets - 1) + (1,)

basis = PolynomialBasis(degree)

xo = np.zeros(num_assets)
X0 = np.tile(xo, (batch_size, 1))

model = HJB(X0=X0, delta_t=dt, T=T, sigma=np.sqrt(2))
configurations = f"{num_assets} assets | {N} steps | {batch_size} batch size | {n_iter} iterations | {degree} degree | {rank} rank"

# Compute trajectories
X, noise = generate_trajectories(X0, T, N, model) # (batch_size, N + 1, dim), (batch_size, N + 1, dim) (xi[0] is not used)

phi_X = []
dphi_X = []
ddphi_X = []

# Compute phi_X and dphi_X for each time step
for n in range(N + 1):
    phi_X_n = [TensorCore(basis.eval(X[:, n, i]), name=f"phi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]
    dphi_X_n = [TensorCore(basis.grad(X[:, n, i]), name=f"dphi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]
    ddphi_X_n = [TensorCore(basis.dgrad(X[:, n, i]), name=f"ddphi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]
    phi_X.append(phi_X_n)
    dphi_X.append(dphi_X_n)
    ddphi_X.append(ddphi_X_n)

Y = np.zeros((batch_size, N + 1))
Y[:, -1] = model.g(X[:, -1])  # (batch_size, )

start_time = time.perf_counter()
V = [None for _ in range(N + 1)]
V_N = ALS(phi_X[-1], Y[:, -1], n_iter=n_iter, ranks=ranks)
V[-1] = V_N
print("Time to compute V_N:", f"{time.perf_counter() - start_time:.2f}s")

check_V = fast_contract(V_N, phi_X[-1])
error = check_V - Y[:, -1]
print(f"Mean reconstruction error at N: {np.mean(np.abs(error)):.2e}, Max reconstruction error at N: {np.max(np.abs(error)):.2e}\n")

start_time = time.perf_counter()

step_times = []
mean_relative_errors = []
max_relative_errors = []

mean_relative_errors.append(np.mean(np.abs(error)))
max_relative_errors.append(np.max(np.abs(error)))

for n in range(N - 1, -1, -1):
    print("Step:", n)
    step_start_time = time.perf_counter()
    # Compute Y = V_n+1(X_n+1)
    # Compute Z = grad_x V_n+1(X_n+1)
    # Compute h_n(X_n, t_n, Y, Z)
    # Compute dt * h_n(X_n, t_n, Y, Z) + V_n+1(X_n+1)
    V_n1 = V[n + 1]
    Y_n1 = Y[:, n + 1]  # (batch_size, )

    X_n1 = X[:, n + 1, :]  # (batch_size, num_assets)
    phi_X_n1 = phi_X[n+1]  # tensor core of shape (batch_size, degree) * num_assets
    dphi_X_n1 = dphi_X[n+1]  # tensor core of shape (batch_size, degree) * num_assets

    X_n = X[:, n, :]  # (batch_size, num_assets)
    phi_X_n = phi_X[n]  # tensor core of shape (batch_size, degree) * num_assets
    dphi_X_n = dphi_X[n]  # tensor core of shape (batch_size, degree) * num_assets

    noise_n1 = noise[:, n + 1, :]  # (batch_size, num_assets)
    
    V_nk = V_n1
    Y_nk = Y_n1
    for k in range(n_iter_implicit):
        grand_Vnk = multi_derivative(V_nk, phi_X_n, dphi_X_n) # (batch_size, num_assets)
        sigma_nk = model.sigma(X_n, n*dt) # (batch_size, num_assets, num_assets)
        Z_nk = np.einsum('ijk, ik -> ij', sigma_nk, grand_Vnk) # (batch_size, num_assets)
        h_nk = model.h(X_n, n*dt, Y_nk, Z_nk)

        # print(Z_nk.shape)
        # print(model.sigma(X_n, n*dt).shape)
        # print((np.sum(Z_nk * model.sigma(X_n, n*dt) * noise[:, n+1], axis=1)).shape)
        # print(V_n1)
        # print(h_nk)
        step_nk = h_nk*dt + Y_n1 - np.sqrt(dt) * np.einsum('ij,ij->i', Z_nk, noise_n1)

        V_nk = ALS(phi_X_n, step_nk, n_iter=n_iter, ranks=ranks, init_tt=V_nk)
        Y_nk = fast_contract(V_nk, phi_X_n)

    V[n] = V_nk
    Y[:, n] = Y_nk

    step_times.append(time.perf_counter() - step_start_time)

    ground_prices = model.price(X_n, n*dt) # (batch_size, )
    print("Mean reconstruction error at n:", f"{np.mean(np.abs(Y_nk - ground_prices)):.2e}, Max reconstruction error at n:", f"{np.max(np.abs(Y_nk - ground_prices)):.2e}")
    print("Step time:", f"{time.perf_counter() - step_start_time:.2f}s\n")

    relative_errors = np.abs(Y_nk - ground_prices) / ground_prices
    mean_relative_errors.append(np.mean(relative_errors))
    max_relative_errors.append(np.max(relative_errors))

end_time = time.perf_counter()

print(f"Total time: {end_time - start_time:.2f}s")
print(f"Mean step time: {np.mean(step_times):.2f}s\n")

ground_truth = model.price(xo[None, :], 0, n_sims=50_000).item()
print(f"Predicted price at 0: {Y[0, 0]:.4f} | Ground price at 0: {ground_truth:.4f}")

error = np.abs(Y[0, 0] - ground_truth)
print(f"Error at 0: {error:.2e} | Relative error at 0: {error / ground_truth:.2e}")

plt.figure(figsize=(10, 5))
n_simulations = 3
colormap = plt.cm.viridis

simulation_indices = np.random.choice(batch_size, n_simulations, replace=False)
for j in range(len(simulation_indices)):
    predicted_prices = [Y[simulation_indices[j], i] for i in range(N + 1)]
    ground_prices = [model.price(np.array([X[simulation_indices[j], i]]), i * dt, n_sims=50_000) for i in range(N + 1)]

    plt.plot(predicted_prices, label=f"Price #{j}", linestyle="--", color=colormap(j / n_simulations), lw=0.8)
    plt.plot(ground_prices, label=f"Ground Price #{j}", linestyle="-", color=colormap(j / n_simulations), lw=0.8)

plt.scatter([0], [np.mean(Y[:, 0])], color="red", label="Predicted Price at 0", marker="x")
plt.scatter([0], [ground_truth], color="red", label="Ground Price at 0", marker="o")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title(f"Evolutions of prices | {configurations}")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, T, N + 1), mean_relative_errors[::-1], label="Mean relative error")
plt.plot(np.linspace(0, T, N + 1), max_relative_errors[::-1], label="Max relative error")
plt.xlabel("Time")
plt.ylabel("Relative error")
plt.title(f"Relative errors | {configurations}")
plt.legend()
plt.show()

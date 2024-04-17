from bsde_solver import xp

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
sigma = 0.4
r = 0.05
N = 100
num_assets = 2
dt = T / N

n_iter = 5
n_iter_implicit = 10
rank = 2
degree = 2
shape = tuple([degree for _ in range(num_assets)])
ranks = (1,) + (rank,) * (num_assets - 1) + (1,)

# basis = LegendreBasis(degree)
basis = PolynomialBasis(degree)

# X0 = xp.zeros(num_assets) # Hamilton-Jacobi-Bellman (HJB) initial condition
# X0 = xp.zeros(num_assets) # Allen-Cahn initial condition
X0 = xp.array(flatten([(1, 0.5) for _ in range(num_assets//2)])) # Black-Scholes initial condition
# X0 = -xp.ones(num_assets) # Double-well HJB initial condition
X0_batch = xp.broadcast_to(X0, (batch_size, num_assets))

model = BlackScholes(X0, dt, T, r, sigma)
# model = AllenCahn(X0, dt, T)
# model = HJB(X0, dt, T, sigma=xp.sqrt(2))
# nu = xp.array([0.05 for _ in range(num_assets)])
#model =DoubleWellHJB(X0, dt, T, nu)
pde_loss = PDELoss(model)

configurations = f"{num_assets} assets | {N} steps | {batch_size} batch size | {n_iter} iterations | {degree} degree | {rank} rank"

# Compute trajectories
X, noise = generate_trajectories(X0_batch, N, model) # (batch_size, N + 1, num_assets)

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

Y = xp.zeros((batch_size, N + 1))
Y[:, -1] = model.g(X[:, -1])  # (batch_size, )

start_time = time.perf_counter()
V = [None for _ in range(N + 1)]
V_N = ALS(phi_X[-1], Y[:, -1], n_iter=n_iter, ranks=ranks)
V[-1] = V_N
print("Time to compute V_N:", f"{time.perf_counter() - start_time:.2f}s")

check_V = fast_contract(V_N, phi_X[-1])

print("Mean reconstruction error at N:", f"{xp.abs(xp.mean(check_V - Y[:, -1])):.2e}")
print("Prediction at N:", f"{xp.mean(Y[:, -1]):.4f} | Value at N:", f"{xp.mean(check_V):.4f}")

print("Start")
start_time = time.perf_counter()

step_times = []
relative_errors = []
errors = []

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

    V_nk = V_n1
    Y_nk = Y_n1
    for k in range(n_iter_implicit):
        Z_nk = multi_derivative(V_nk, phi_X_n, dphi_X_n)
        h_nk = model.h(X_n, n*dt, Y_nk, Z_nk)

        # print(Z_nk.shape)
        # print(model.sigma(X_n, n*dt).shape)
        # print((xp.sum(Z_nk * model.sigma(X_n, n*dt) * noise[:, n+1], axis=1)).shape)
        # print(V_n1)
        # print(h_nk)
        step_nk = h_nk*dt + Y_n1 - (xp.sum(Z_nk * model.sigma(X_n, n*dt) * noise[:, n+1], axis=1) * xp.sqrt(dt))

        V_nk = ALS(phi_X_n, step_nk, n_iter=n_iter, ranks=ranks, init_tt=V_nk)
        Y_nk = fast_contract(V_nk, phi_X_n)

    V[n] = V_nk
    Y[:, n] = Y_nk.view(xp.ndarray).squeeze()

    step_times.append(time.perf_counter() - step_start_time)

    price_n = model.price(X_n, n*dt)
    # print("Mean reconstruction error at n:", f"{xp.abs(xp.mean(price_n - Y[:, n])):.2e}")
    print("Mean reconstruction error at n:", f"{xp.mean(xp.abs(price_n - Y[:, n])):.2e}")
    print("Step time:", f"{time.perf_counter() - step_start_time:.2f}s")

    relative_errors.append(xp.abs(price_n - xp.mean(Y[:, n])) / price_n)
    errors.append(xp.abs(price_n - xp.mean(Y[:, n])))

    if num_assets < 10:
        vt = (Y[:, n + 1] - Y[:, n]) / dt
        vx = Z_nk
        vxx = hessian(V_nk, phi_X_n, dphi_X_n, ddphi_X[n], batch=True).transpose((2, 0, 1))
        loss = pde_loss(n*dt, X_n, Y_nk, vt, vx, vxx)
        print("Mean PDE loss", loss.mean())
        print("Mean abs PDE loss", xp.abs(loss).mean())

print("End")

end_time = time.perf_counter()

print(f"Time: {end_time - start_time:.2f}s")
print(f"Mean step time: {xp.mean(step_times):.2f}s")

print()
price = xp.mean(model.price(X0_batch, 0))
print("Price at 0", price)
print("Predicted Price:", xp.mean(Y[:, 0]))

print(Y[:, 0])

# Print relative error in percentage
print("Relative error:", f"{xp.abs(price - xp.mean(Y[:, 0])) / price * 100:.2f}%")

plt.figure(figsize=(10, 5))
n_simulations = 3
colormap = plt.cm.viridis

simulation_indices = xp.random.choice(batch_size, n_simulations, replace=False)
for j in range(len(simulation_indices)):
    predicted_prices = [Y[simulation_indices[j], i] for i in range(N + 1)]
    ground_prices = [model.price(xp.array([X[simulation_indices[j], i]]), i * dt) for i in range(N + 1)]

    plt.plot(predicted_prices, label=f"Price #{j}", linestyle="--", color=colormap(j / n_simulations), lw=0.8)
    plt.plot(ground_prices, label=f"Ground Price #{j}", linestyle="-", color=colormap(j / n_simulations), lw=0.8)

plt.scatter([0], [xp.mean(Y[:, 0])], color="red", label="Predicted Price at 0", marker="x")
plt.scatter([0], [xp.mean(model.price(X0_batch, 0))], color="red", label="Ground Price at 0", marker="o")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title(f"Evolutions of prices | {configurations}")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(xp.arange(N + 1), relative_errors[::-1])
plt.xlabel("Time")
plt.ylabel("Relative error")
plt.title(f"Relative error | {configurations}")
plt.show()

plt.figure(figsize=(10, 5))

plt.xlabel("Time")
plt.ylabel("Error")
plt.title(f"Error | {configurations}")
plt.show()
import numpy as np

from bsde_solver.bsde import BackwardSDE, HJB
from bsde_solver.stochastic.path import generate_trajectories
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.tensor.tensor_train import BatchTensorTrain
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.calculus.derivative import batch_derivative, multi_derivative
from bsde_solver.core.optimisation.scalar_als import multi_als


def poly(x, degree=10):
    return np.array([x**i for i in range(degree)]).T #+ [np.log(1/2+1/2*x**2)]


def poly_derivative(x, degree=10):
    return np.array([i * x ** (i - 1) for i in range(degree)]).T# + [2*x/(x**2+1)]).T


batch_size = 1000
N = 2
num_assets = 10
dt = 0.01
T = N * dt

rank = 2
degree = 2
shape = tuple([degree for _ in range(num_assets)])
ranks = (1,) + (rank,) * (num_assets - 1) + (1,)

X0 = np.zeros(num_assets) #np.random.rand(num_assets)

model = HJB(X0, dt, T)

# Compute trajectories
X = generate_trajectories(batch_size, N, num_assets, X0, model, dt)
print(X.shape)  # (batch_size, N + 1, num_assets)

phi_X = []
dphi_X = []

# Compute phi_X and dphi_X for each time step
for n in range(N + 1):
    phi_X_n = [TensorCore(poly(X[:, n, i], degree=degree),name=f"phi_{i+1}",indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]
    dphi_X_n = [TensorCore(poly_derivative(X[:, n, i], degree=degree),name=f"dphi_{i+1}",indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]
    phi_X.append(phi_X_n)
    dphi_X.append(dphi_X_n)

print(phi_X[-1])
print(dphi_X[-1])

Y = np.zeros((batch_size, N + 1))
Y[:, -1] = model.g(X[:, -1, :])  # (batch_size, )

print(Y[:10, -1])

V = [None for _ in range(N + 1)]
V_N = multi_als(phi_X[-1], Y[:, -1], n_iter=50, ranks=ranks)
print(V_N)
V[-1] = V_N

check_V = (
    TensorNetwork(
        cores=[V_N] + phi_X[-1], names=["V"] + [f"phi_{i+1}" for i in range(num_assets)]
    )
    .contract(batch=True)
    .view(np.ndarray)
    .squeeze()
)
print('Reconstruction error at N:', np.linalg.norm(check_V - Y[:, -1]))
print('Mean reconstruction error at N:', np.mean(np.abs(check_V - Y[:, -1])))
print('Max reconstruction error at N:', np.max(np.abs(check_V - Y[:, -1])))
print('Min reconstruction error at N:', np.min(np.abs(check_V - Y[:, -1])))

print("Top 10 largest errors at N:", np.sort(np.abs(check_V - Y[:, -1]))[-10:])

for n in range(N - 1, -1, -1):
    print("Step:", n)
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

    Z_n1 = multi_derivative(V_n1, phi_X_n1, dphi_X_n1)  # (batch_size, num_assets)
    h_n1 = model.h(X_n1, (n+1) * dt, Y_n1, Z_n1)  # (batch_size, )

    noise = np.random.randn(batch_size, num_assets)
    V_n = multi_als(phi_X_n, h_n1*dt + Y_n1 - np.sum(model.sigma(X_n1, (n+1) *dt) @ Z_n1 * noise * np.sqrt(dt), axis=1), n_iter=15, ranks=ranks)
    V[n] = V_n

    Y_n = TensorNetwork(
        cores=[V_n] + phi_X_n, names=["V"] + [f"phi_{i+1}" for i in range(num_assets)]
    ).contract(batch=True, indices=('batch',))

    Y[:, n] = Y_n.view(np.ndarray).squeeze()

    print(Y[:10, n])
    # break
print("End")

# Compute with Monte Carlo - log E[exp(-model.g(X0 + sqrt(T) * model.sigma(X0) * Z))]

# Z from N(0, I_dxd) with d = num_assets
size = 100000
Z = np.random.randn(size, num_assets)
print(model.sigma(X0, T).shape)
X_T = np.sqrt(T) * Z @ model.sigma(X0, T) + np.repeat(X0, size).reshape(num_assets, size).T
# X_T = np.sqrt(T) * model.sigma(X0, T) @ Z.T + np.repeat(X0, size).reshape(num_assets, size)
print(X_T.shape)
Y_T = model.g(X_T)
print(Y_T.shape)
price = -np.log(np.mean(np.exp(-Y_T)))

print("Ground Price:", price)

# Compute with Tensor Train
V_0 = V[0]
Y_0 = TensorNetwork(
    cores=[V_0] + phi_X[0], names=["V"] + [f"phi_{i+1}" for i in range(num_assets)]
).contract(batch=True, indices=('batch',))

print(np.mean(Y_0.view(np.ndarray).squeeze()))
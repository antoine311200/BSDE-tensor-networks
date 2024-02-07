import numpy as np

from bsde_solver.bsde import BackwardSDE, HJB
from bsde_solver.stochastic.path import generate_trajectories
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.tensor.tensor_train import BatchTensorTrain
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.calculus.derivative import batch_derivative
from bsde_solver.core.optimisation.scalar_als import batch_scalar_ALS


def poly(x, degree=10):
    return np.array([x**i for i in range(degree)]).T


def poly_derivative(x, degree=10):
    return np.array([i * x ** (i - 1) for i in range(degree)]).T


batch_size = 64
N = 100
num_assets = 5
dt = 0.01
T = N * dt

rank = 3
degree = 4
shape = tuple([degree for _ in range(num_assets)])
ranks = (1,) + (rank,) * (num_assets - 1) + (1,)

X0 = np.random.rand(num_assets)

model = HJB(X0, dt, T)

# Compute trajectories
X = generate_trajectories(batch_size, N, num_assets, X0, model, dt)
print(X.shape)  # (batch_size, N + 1, dim)

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

V = [None for _ in range(N + 1)]
V_N = batch_scalar_ALS(phi_X[-1], Y[:, -1], n_iter=5, ranks=ranks)
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

for n in range(N - 1, -1, -1):
    # Compute Y = V_n+1(X_n+1)
    # Compute Z = grad_x V_n+1(X_n+1)
    # Compute h_n(X_n, t_n, Y, Z)
    # Compute dt * h_n(X_n, t_n, Y, Z) + V_n+1(X_n+1)
    V_n = V[n + 1]
    Y_n = Y[:, n + 1]  # (batch_size, )
    X_n = X[:, n, :]  # (batch_size, num_assets)
    phi_X_n = phi_X[n]  # tensor core of shape (batch_size, degree) * num_assets
    dphi_X_n = dphi_X[n]  # tensor core of shape (batch_size, degree) * num_assets
    Z_n = batch_derivative(V_n, phi_X_n, dphi_X_n)  # (batch_size, num_assets)
    h_n = model.h(X_n, n * dt, Y_n, Z_n)  # (batch_size, )

    V_n = batch_scalar_ALS(phi_X_n, h_n, n_iter=5, ranks=ranks)
    V[n] = V_n
    print(V_n)
    Y_n = TensorNetwork(
        cores=[V_n] + phi_X_n, names=["V"] + [f"phi_{i+1}" for i in range(num_assets)]
    ).contract(batch=True, indices=('batch',))
    print(Y_n)

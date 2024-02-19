import numpy as np

import matplotlib.pyplot as plt

from bsde_solver.bsde import BackwardSDE, HJB, DoubleWellHJB
from bsde_solver.stochastic.path import generate_trajectories
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.tensor.tensor_train import BatchTensorTrain
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.calculus.derivative import batch_derivative, multi_derivative
from bsde_solver.core.calculus.hessian import hessian
from bsde_solver.core.optimisation.scalar_als import multi_als
from bsde_solver.core.calculus.basis import LegendreBasis, PolynomialBasis
from bsde_solver.loss import PDELoss

batch_size = 10000
T = 0.5
N = 50
num_assets = 5
dt = T / N


rank = 2
degree = 2
shape = tuple([degree for _ in range(num_assets)])
ranks = (1,) + (rank,) * (num_assets - 1) + (1,)
# ranks = (rank,) * (num_assets + 1)

basis = LegendreBasis(degree)
basis = PolynomialBasis(degree)

X0 = np.zeros(num_assets) #np.random.rand(num_assets)#-np.ones(num_assets) #
# X0 = np.arange(1, num_assets + 1)

# model = HJB(X0, dt, T)
nu = np.array([0.05 for _ in range(num_assets)])
model = HJB(X0, dt, T)
#DoubleWellHJB(X0, dt, T, nu)
pde_loss = PDELoss(model)

# Compute trajectories
X, noise = generate_trajectories(batch_size, N, num_assets, X0, model, dt) # (batch_size, N + 1, num_assets)

phi_X = []
dphi_X = []
ddphi_X = []

# Compute phi_X and dphi_X for each time step
for n in range(N + 1):
    phi_X_n = [TensorCore(basis.eval(X[:, n, i]),name=f"phi_{i+1}",indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]
    dphi_X_n = [TensorCore(basis.grad(X[:, n, i]),name=f"dphi_{i+1}",indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]
    ddphi_X_n = [TensorCore(basis.dgrad(X[:, n, i]),name=f"ddphi_{i+1}",indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]
    phi_X.append(phi_X_n)
    dphi_X.append(dphi_X_n)
    ddphi_X.append(ddphi_X_n)

Y = np.zeros((batch_size, N + 1))
Y[:, -1] = model.g(X[:, -1])  # (batch_size, )
#-np.log(np.mean(np.exp(-model.g(X[:, -1]))))
#model.g(X[:, -1])  # (batch_size, )

V = [None for _ in range(N + 1)]
V_N = multi_als(phi_X[-1], Y[:, -1], n_iter=1, ranks=ranks)
V[-1] = V_N

check_V = (
    TensorNetwork(
        cores=[V_N] + phi_X[-1], names=["V"] + [f"phi_{i+1}" for i in range(num_assets)]
    )
    .contract(batch=True)
    .view(np.ndarray)
    .squeeze()
)

print("Mean reconstruction error at N:", f"{np.abs(np.mean(check_V - Y[:, -1])):.2e}")
print("Value at N:", f"{np.mean(Y[:, -1]):.4f}")


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

    dV_n1 = multi_derivative(V_n1, phi_X_n1, dphi_X_n1)  # (batch_size, num_assets)
    Z_n1 = dV_n1 @ model.sigma(X_n1, (n+1) *dt) # (batch_size, num_assets)
    h_n1 = model.h(X_n1, (n+1) * dt, Y_n1, Z_n1)  # (batch_size, )

    # noise = np.random.randn(batch_size, num_assets)
    # print(noise.shape)
    # noise_n1 = noise[:, n+1, :]  # (batch_size, num_assets)

    step_n1 = h_n1*dt + Y_n1 #- np.einsum('ij,ij->i', Z_n1, noise) * np.sqrt(dt)
    #np.sum(Z_n1 @ model.sigma(X_n1, (n+1) *dt) * noise, axis=1) * np.sqrt(dt) + Y_n1
    V_n = multi_als(phi_X_n, step_n1, n_iter=1, ranks=ranks, init_tt=V_n1)

    V[n] = V_n
    Y_n = TensorNetwork(
        cores=[V_n] + phi_X_n, names=["V"] + [f"phi_{i+1}" for i in range(num_assets)]
    ).contract(batch=True, indices=('batch',))

    Y[:, n] = Y_n.view(np.ndarray).squeeze()

    # print(phi_X_n[0].view(np.ndarray)[:10])
    print(Y[:10, n])
    print(Y[:, n].mean())

    vt = (Y[:, n + 1] - Y[:, n]) / dt
    vx = dV_n1
    vxx = hessian(V_n, phi_X_n, dphi_X_n, ddphi_X[n], batch=True).transpose((2, 0, 1))
    loss = pde_loss(n*dt, X_n, Y_n, vt, vx, vxx)

    print("Mean PDE loss", loss.mean())
    print("Mean abs PDE loss", np.abs(loss).mean())

print("End")

# Compute with Monte Carlo - log E[exp(-model.g(X0 + sqrt(T) * model.sigma(X0) * Z))]

# Z from N(0, I_dxd) with d = num_assets
def compute_price(X, T, t, i):
    '''Compute HJB close form solution'''
    # X_T = X + np.sqrt(2 * (T - t)) * np.random.randn(X.shape[0], X.shape[1])
    X_T = np.sqrt(T - t) * np.random.randn(10*batch_size, num_assets) + X.repeat(10, axis=0)
    return -np.log(np.mean(2 / (1 + np.sum(X_T**2, 1))))

noise0 = np.random.randn(batch_size, num_assets)
X_T = np.sqrt(T) * noise0 @ model.sigma(X0, T) + np.broadcast_to(X0, (batch_size, num_assets))
print("Price at 0", -np.log(np.mean(np.exp(-model.g(X_T)))))

# Price at N
noiseN = np.random.randn(200*batch_size, num_assets)
X_T = np.sqrt(T-T) * noiseN @ model.sigma(X[:, -1], T) + X[:, -1].repeat(200, axis=0)
print("Price at N", -np.log(np.mean(np.exp(-model.g(X_T)))))
print(-np.log(np.mean(2/(1+np.sum(X_T**2, 1)))))

# price = compute_price(np.repeat(X0, batch_size).reshape(num_assets, batch_size).T, T)
# print("Ground Price:", price)
print("Predicted Price:", np.mean(Y[:, 0]))

# prices = [compute_price(X[:, i], T - i * dt) for i in range(N + 1)]
predicted_prices = [np.mean(Y[:, i]) for i in range(N + 1)]
print(predicted_prices)

ground_prices = [compute_price(X[:, i], T, i * dt, i) for i in range(N + 1)]
print(ground_prices)

plt.figure(figsize=(10, 5))
plt.plot(ground_prices, label="Ground Price")
plt.plot(predicted_prices, label="Predicted Price")
# # Plot the point Y[:, -1]
# plt.scatter([N], [np.mean(Y[:, -1])], color="red", label="Ground Price at T")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# # Compute with Tensor Train
# # V_0 = V[0]
# # Y_0 = TensorNetwork(
# #     cores=[V_0] + phi_X[0], names=["V"] + [f"phi_{i+1}" for i in range(num_assets)]
# # ).contract(batch=True, indices=('batch',))

# # print(np.mean(Y_0.view(np.ndarray).squeeze()))


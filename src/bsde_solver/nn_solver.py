import torch
from torch import nn

import numpy as np

from src.bsde_solver.bsde import BackwardSDE

class NeuralNetworkSolver(nn.Module):
    def __init__(self, X_init, bsde: BackwardSDE, method: str = "explicit"):
        super(NeuralNetworkSolver, self).__init__()

        self.X_init = X_init

        self.network = DeepDenseNet(bsde.dim + 1, bsde.dim, 50, 3)
        self.bsde = bsde

    def forward(self, t, X):
        X = torch.tensor(X, requires_grad=True)
        t = torch.ones(X.shape[0], 1) * t
        inp = torch.cat([t, X], 1)

        inp = inp.float()

        u = self.network(inp)
        Du = torch.autograd.grad(u.sum(), X, create_graph=True)[0]

        return u, Du

    def loss(self):
        X = torch.zeros((self.batch_size, self.bsde.N + 1, self.bsde.dim))
        xi = torch.randn((self.batch_size, self.bsde.N + 1, self.bsde.dim))

        Y = torch.zeros((self.batch_size, self.bsde.N + 1, 1), requires_grad=False)
        Z = torch.zeros((self.batch_size, self.bsde.N + 1, self.bsde.dim), requires_grad=False)

        Y0, Z0 = self.forward(0, X[:, 0])
        X[:, 0] = torch.tensor(self.X_init).repeat(self.batch_size, 1)
        Y[:, 0] = Y0.detach()
        Z[:, 0] = Z0.detach()

        loss = torch.zeros(1)
        dt = self.bsde.delta_t

        for n in range(self.bsde.N):
            X[:, n + 1] = X[:, n] + \
                self.bsde.b(X[:, n], n * dt) * dt + \
                self.bsde.sigma(X[:, n], n * dt) * np.sqrt(dt) * xi[:, n]

            # checkout for a - sign
            # print(Y[:, n].shape, Z[:, n].shape, self.bsde.sigma(X[:, n], n * dt).shape, xi[:, n].shape)
            Y[:, n + 1] = Y[:, n] - \
                self.bsde.h(X[:, n], n * dt, Y[:, n], Z[:, n]) * dt - \
                Z[:, n] @ self.bsde.sigma(X[:, n], n * dt).T @ (np.sqrt(dt) * xi[:, n])

            yn, zn = self.forward(n * self.bsde.delta_t, X[:, n])
            # print(yn.shape, zn.shape, Z[:, n].shape)
            Z[:, n + 1] = zn.detach()
            yn = yn.detach()

            loss += (Y[:, n] - yn).pow(2).sum()

        loss += (Y[:, self.bsde.N] - self.bsde.g(X[:, self.bsde.N])).pow(2).sum()

        return loss

    def train(self, num_iterations: int = 1000, batch_size: int = 64):
        self.batch_size = batch_size

        for it in range(num_iterations):
            self.network.optimizer.zero_grad()
            self.loss().backward()
            self.network.optimizer.step()

            if it % 10 == 0:
                print("Iteration: %d, Loss: %.4e" % (it, self.loss().item()))


class DeepDenseNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int = 3, learning_rate: float = 1e-3):
        super(DeepDenseNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
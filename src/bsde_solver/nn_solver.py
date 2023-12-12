import torch
from torch import nn

import numpy as np

from bsde_solver.bsde import BackwardSDE

class NeuralNetworkSolver(nn.Module):
    def __init__(self, bsde: BackwardSDE, network: nn.Module, method: str = "explicit"):
        super(NeuralNetworkSolver, self).__init__()

        self.network = network
        self.bsde = bsde

    def forward(self, t, X):
        inp = torch.cat([t, X], 1)

        u = self.network(inp)
        Du = torch.autograd.grad(u.sum(), X, create_graph=True)[0]

        return u, Du

    def loss(self):
        pass

    def train(self, num_iterations: int = 1000):
        for it in range(num_iterations):


            self.optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            self.optimizer.step()

    def fetch_batch(self):
        X = self.bsde.X(batch_size=self.batch_size)
        Y = np.zeros((self.batch_size, self.N + 1))

        for n in range(self.N):
            t = self.time[n]



            u, Du = self.forward(t, X[:, n])


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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.leanrning_rate)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
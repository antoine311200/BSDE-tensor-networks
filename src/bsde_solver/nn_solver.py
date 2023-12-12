import torch
from torch import nn

from bsde_solver.bsde import BackwardSDE

class NeuralNetworkSolver(nn.Module):
    def __init__(self, bsde: BackwardSDE, network: nn.Module, method: str = "explicit"):
        super(NeuralNetworkSolver, self).__init__()

        self.networks = [network for _ in range(bsde.N)]
        self.bsde = bsde


    def loss(self):
        pass

    def train(self):
        pass



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
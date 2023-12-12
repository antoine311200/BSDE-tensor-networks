from src.bsde_solver.nn_solver import NeuralNetworkSolver
from src.bsde_solver.bsde import BackwardSDE, BlackScholes

import torch
from torch import nn

import numpy as np

import matplotlib.pyplot as plt


def main():
    # BSDE parameters
    T = 1.0
    M = 100
    N = 100
    D = 1
    Xi = np.zeros((1, D))

    # Training parameters
    num_iterations = 10000
    batch_size = 64
    learning_rate = 1e-3

    # Create the BSDE
    #         def __init__(self, X0, delta_t, T, r, sigma, S0) -> None:
    bsde = BlackScholes(Xi, T / N, T, 0.05, 0.2, 100)

    # Create the solver
    solver = NeuralNetworkSolver(Xi, bsde)

    # Train the solver
    solver.train(num_iterations)

    # Generate the data
    # X = bsde.X(batch_size)
    # Y = bsde.Y(X)

    # # Predict the data
    # Y_pred = solver.predict(X)

    # # Plot the results
    # plt.figure()
    # plt.plot(X[0, :, 0], Y[0], label="True")
    # plt.plot(X[0, :, 0], Y_pred[0], label="Predicted")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
import numpy as np

from bsde_solver.tensor.tensor_train import TensorTrain
from bsde_solver.tensor.tensor_network import TensorNetwork
from bsde_solver.tensor.tensor_core import TensorCore


def derivative(tt: TensorTrain, phi, dphi):
    derivative = 0
    right_parts, left_parts = [], []
    for i in range(tt.order - 1, 0, -1):
        right_part = TensorNetwork(
            cores=[
                tt.cores[f"core_{i}"],
                phi[i],
                right_parts[-1] if len(right_parts) > 0 else None,
            ],
            names=[f"core_{i}", f"phi_{i}", f"right_part_{i}"],
        ).contract()
        right_parts.append(right_part)

    for i in range(tt.order):
        left_part = TensorNetwork(
            cores=[
                left_parts[-1] if len(left_parts) > 0 else None,
                phi[i],
                tt.cores[f"core_{i}"],
            ],
            names=[f"left_part_{i}", f"phi_{i}", f"core_{i}"],
        ).contract()

        partial_derivative = TensorNetwork(
            cores=[
                left_parts[-1] if len(left_parts) > 0 else None,
                right_parts[-(i+1)] if i < len(right_parts) else None,
                tt.cores[f"core_{i}"],
                dphi[i],
            ],
            names=[f"left_part_{i}", f"right_part_{i}", f"core_{i}", f"dphi_{i}"],
        )
        left_parts.append(left_part)
        derivative += float(partial_derivative.contract().squeeze())
    return derivative

def batch_derivative(tt: TensorTrain, phis, dphis):
    pass

if __name__ == "__main__":
    degree = 10
    num_assets = 5

    shape = [degree for _ in range(num_assets)]
    ranks = [1, 3, 3, 3, 3, 1]
    tt = TensorTrain(shape, ranks)
    tt.randomize()

    def poly(x, degree=10):
        return np.array([x**i for i in range(degree)]).T

    def poly_derivative(x, degree=10):
        return np.array([i*x**(i-1) for i in range(degree)]).T

    from time import perf_counter

    n_simulations = 1000
    phis, dphis = [], []
    for i in range(n_simulations):
        # x = [0.1, 0.4, 0.12, 0.9, 0.3]
        x = np.random.rand(num_assets)

        phi = [
            TensorCore(poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",))
            for i in range(tt.order)
        ]
        dphi = [
            TensorCore(poly_derivative(x[i], degree=degree), name=f"dphi_{i+1}", indices=(f"m_{i+1}",))
            for i in range(tt.order)
        ]
        phis.append(phi)
        dphis.append(dphi)

    start = perf_counter()
    derivatives = []
    for i in range(n_simulations):
        derivatives.append(derivative(tt, phis[i], dphis[i]))

    print("Time:", perf_counter() - start)

import numpy as np

from bsde_solver.tensor.tensor_train import TensorTrain, BatchTensorTrain
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
                right_parts[-(i + 1)] if i < len(right_parts) else None,
                tt.cores[f"core_{i}"],
                dphi[i],
            ],
            names=[f"left_part_{i}", f"right_part_{i}", f"core_{i}", f"dphi_{i}"],
        )
        left_parts.append(left_part)
        derivative += float(partial_derivative.contract().squeeze())
    return derivative


def batch_derivative(tt_batch: BatchTensorTrain, phis, dphis):
    phis_batch = TensorNetwork(cores=phis, names=[f"phi_{i}" for i in range(len(phis))])
    dphis_batch = TensorNetwork(
        cores=dphis, names=[f"dphi_{i}" for i in range(len(dphis))]
    )

    derivatives = np.zeros(tt_batch.batch_size)
    right_parts, left_parts = [], []

    for i in range(tt_batch.order - 1, 0, -1):
        right_part = TensorNetwork(
            cores=[
                tt_batch.cores[f"core_{i}"],
                phis_batch.cores[f"phi_{i}"],
                right_parts[-1] if len(right_parts) > 0 else None,
            ],
            names=[f"core_{i}", f"phi_{i}", f"right_part_{i}"],
        ).contract(batch=True)
        right_parts.append(right_part)

    for i in range(tt_batch.order):
        left_part = TensorNetwork(
            cores=[
                left_parts[-1] if len(left_parts) > 0 else None,
                phis_batch.cores[f"phi_{i}"],
                tt_batch.cores[f"core_{i}"],
            ],
            names=[f"left_part_{i}", f"phi_{i}", f"core_{i}"],
        ).contract(batch=True)

        partial_derivative = TensorNetwork(
            cores=[
                left_parts[-1] if len(left_parts) > 0 else None,
                right_parts[-(i + 1)] if i < len(right_parts) else None,
                tt_batch.cores[f"core_{i}"],
                dphis_batch.cores[f"dphi_{i}"],
            ],
            names=[f"left_part_{i}", f"right_part_{i}", f"core_{i}", f"dphi_{i}"],
        )
        left_parts.append(left_part)
        derivatives += partial_derivative.contract(batch=True).squeeze().view(np.ndarray)

    return derivatives

if __name__ == "__main__":
    from time import perf_counter

    degree = 10
    num_assets = 5

    shape = [degree for _ in range(num_assets)]
    ranks = [1, 3, 3, 3, 3, 1]
    tt = TensorTrain(shape, ranks)
    tt.randomize()

    def poly(x, degree=10):
        return np.array([x**i for i in range(degree)]).T

    def poly_derivative(x, degree=10):
        return np.array([i * x ** (i - 1) for i in range(degree)]).T

    n_simulations = 5000
    phis, dphis = [], []
    np.random.seed(0)
    for i in range(n_simulations):
        x = np.random.rand(num_assets)

        phi = [
            TensorCore(
                poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",)
            )
            for i in range(tt.order)
        ]
        dphi = [
            TensorCore(
                poly_derivative(x[i], degree=degree),
                name=f"dphi_{i+1}",
                indices=(f"m_{i+1}",),
            )
            for i in range(tt.order)
        ]
        phis.append(phi)
        dphis.append(dphi)

    #################### Single derivative ####################

    start_time = perf_counter()
    derivatives = np.zeros(n_simulations)
    for i in range(n_simulations):
        derivatives[i] = derivative(tt, phis[i], dphis[i])

    end_time = perf_counter() - start_time
    print("Time:", end_time)

    #################### Batch derivative ####################

    phis, dphis = [], []
    np.random.seed(0)
    for i in range(n_simulations):
        x = np.random.rand(num_assets)

        phi = [poly(x[i], degree=degree) for i in range(tt.order)]
        dphi = [poly_derivative(x[i], degree=degree) for i in range(tt.order)]
        phis.append(phi)
        dphis.append(dphi)
    phis = np.array(phis) # (n_simulations, tt.order, degree)
    dphis = np.array(dphis) # (n_simulations, tt.order, degree)

    phis = phis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    dphis = dphis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    phis = [
        TensorCore(
            phis[i], name=f"phi_{i+1}", indices=("batch", f"m_{i+1}")
        )
        for i in range(tt.order)
    ]
    dphis = [
        TensorCore(
            dphis[i], name=f"dphi_{i+1}", indices=("batch", f"m_{i+1}")
        )
        for i in range(tt.order)
    ]

    batch_tt = BatchTensorTrain.dupplicate(n_simulations, tt)

    start_time_batch = perf_counter()
    batch_derivatives = batch_derivative(batch_tt, phis, dphis)
    end_time_batch = perf_counter() - start_time_batch
    print("Time:", end_time_batch)

    print("Speed up factor:", end_time / end_time_batch)

    # Check if the results are the same
    print(np.allclose(derivatives, batch_derivatives))
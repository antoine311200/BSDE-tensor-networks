import numpy as np

from bsde_solver.tensor.tensor_train import TensorTrain, BatchTensorTrain
from bsde_solver.tensor.tensor_network import TensorNetwork

def derivative(tt: TensorTrain, phi, dphi):
    derivative = np.zeros(tt.order)
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
        derivative[i] = float(partial_derivative.contract().squeeze())
    return derivative


def batch_derivative(tt_batch: BatchTensorTrain, phis, dphis):
    phis_batch = TensorNetwork(cores=phis, names=[f"phi_{i}" for i in range(len(phis))])
    dphis_batch = TensorNetwork(
        cores=dphis, names=[f"dphi_{i}" for i in range(len(dphis))]
    )

    derivatives = np.zeros((tt_batch.batch_size, tt_batch.order))
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
        derivatives[:, i] = partial_derivative.contract(batch=True).squeeze().view(np.ndarray)

    return derivatives
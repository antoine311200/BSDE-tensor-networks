from bsde_solver.tensor_train import TensorTrain
from bsde_solver.utils import matricized

import numpy as np

if __name__ == "__main__":

    tt = TensorTrain([10, 10, 10], [1, 4, 4, 1])
    tt.random()

    print(tt.cores[0].shape)
    print(tt.cores[1].shape)
    print(tt.cores[2].shape)

    R = matricized(tt.cores[1], mode="right")
    print(R @ R.T)

    tt.orthonormalize(orthogonality="right")

    R = matricized(tt.cores[1], mode="right")
    print(R @ R.T)
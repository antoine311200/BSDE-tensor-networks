from bsde_solver.tensor_train import TensorTrain
from bsde_solver.utils import matricized

import numpy as np

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

if __name__ == "__main__":

    tt = TensorTrain([10, 10, 10], [1, 4, 4, 1])
    tt.random()

    tt2 = tt.copy()

    print(tt.cores[0].shape, tt.cores[1].shape, tt.cores[2].shape)

    # Right orthonormalization
    print("\nRight orthonormalization")

    R = matricized(tt.cores[1], mode="right")
    print(R @ R.T)

    tt.orthonormalize(start=0, orthogonality="right")

    R0 = matricized(tt.cores[0], mode="left")
    print(R0.T @ R0)
    R1 = matricized(tt.cores[1], mode="left")
    print(R1.T @ R1)
    R2 = matricized(tt.cores[2], mode="left")
    print(R2.T @ R2)

    # Left orthonormalization
    print("\nLeft orthonormalization")

    L1 = matricized(tt2.cores[1], mode="left")
    print(L1.T @ L1)

    tt2.orthonormalize(start=1, orthogonality="left")

    L0 = matricized(tt2.cores[0], mode="left")
    print(L0.T @ L0)
    L1 = matricized(tt2.cores[1], mode="left")
    print(L1.T @ L1)
    L2 = matricized(tt2.cores[2], mode="left")
    print(L2.T @ L2)
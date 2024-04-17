from bsde_solver.core.tensor.tensor_train import TensorTrain
from bsde_solver.utils import matricized

from bsde_solver import xp

xp.set_printoptions(precision=2)
xp.set_printoptions(suppress=True)

if __name__ == "__main__":

    tt = TensorTrain([3, 3, 3], [1, 3, 3, 1])
    tt.random()

    print(tt.cores[0])
    print(tt.cores[1])
    print(tt.cores[2])

    tt2 = tt.copy()

    print(tt.cores[0].shape, tt.cores[1].shape, tt.cores[2].shape)

    # Right orthonormalization
    print("\nRight orthonormalization")

    # R = matricized(tt.cores[1], mode="right")
    # print(R @ R.T)

    tt.orthonormalize(start=0, orthogonality="right")

    R0 = matricized(tt.cores[0], mode="right")
    print(R0 @ R0.T)
    # print(R0.T @ R0)
    R1 = matricized(tt.cores[1], mode="right")
    # print(R1.T @ R1)
    print(R1 @ R1.T)
    R2 = matricized(tt.cores[2], mode="right")
    # print(R2.T @ R2)
    print(R2 @ R2.T)

    # Left orthonormalization
    print("\nLeft orthonormalization")

    # L1 = matricized(tt2.cores[1], mode="left")
    # print(L1.T @ L1)

    tt2.orthonormalize(start=0, orthogonality="left")

    L0 = matricized(tt2.cores[0], mode="left")
    print(L0.T @ L0)
    L1 = matricized(tt2.cores[1], mode="left")
    print(L1.T @ L1)
    L2 = matricized(tt2.cores[2], mode="left")
    print(L2.T @ L2)
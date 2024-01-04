import numpy as np

from bsde_solver.utils import matricized, tensorized

class TensorTrain:

    def __init__(self, shape, rank):
        self.shape = shape
        self.rank = rank

        self.cores = []

    def fill(self, tensor, orthogonality="left"):
        T = tensor

        if orthogonality == "left":
            pass
        elif orthogonality == "right":
            for k in range(len(self.shape) - 1, 0, -1):
                L = matricized(T, shape=self.shape[k], mode="right").T
                Q, R = np.linalg.qr(L, mode="complete")

                Q = Q[:, :self.rank[k]].T
                R = R[:self.rank[k], :].T

                self.cores[k] = tensorized(Q, shape=self.shape[k])
                T = R

            self.cores[0] = tensorized(T, shape=self.shape[0])
        else:
            raise ValueError("orthogonality must be either 'left' or 'right'")

    def orthonormalize(self, orthogonality="right"):
        if orthogonality == "left":
            for k in range(len(self.shape) - 1):
                L = matricized(self.cores[k], mode="left")
                R = matricized(self.cores[k+1], mode="right")

                V, U = np.linalg.qr(L)
                W = U @ R

                self.cores[k] = tensorized(V, shape=self.cores[k].shape)
                self.cores[k+1] = tensorized(W, shape=self.cores[k+1].shape)
        elif orthogonality == "right":
            for k in range(len(self.shape) - 1, 0, -1):
                L = matricized(self.cores[k-1], mode="left")
                R = matricized(self.cores[k], mode="right")

                V, U = np.linalg.qr(R.T)
                W = L @ U.T

                self.cores[k-1] = tensorized(W, shape=self.cores[k-1].shape)
                self.cores[k] = tensorized(V.T, shape=self.cores[k].shape)
        else:
            raise ValueError("orthogonality must be either 'left' or 'right'")

    def random(self):
        for i in range(len(self.shape)):
            self.cores.append(np.random.normal(size=(self.rank[i], self.shape[i], self.rank[i + 1])))
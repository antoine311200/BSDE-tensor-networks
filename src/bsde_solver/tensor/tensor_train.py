import numpy as np

from copy import deepcopy

from bsde_solver.tensor.tensor_core import TensorCore
from bsde_solver.tensor.tensor_network import TensorNetwork


def left_unfold(core: TensorCore):
    return core.unfold((0, 1), 2)

def right_unfold(core: TensorCore):
    return core.unfold(0, (1, 2))

class TensorTrain(TensorNetwork):

    def __init__(self, shape: list[int], ranks: list[int]):
        """Initialize a matrix product state with a given shape and ranks."""
        cores = []
        for i in range(len(shape)):
            core = TensorCore(
                np.zeros((ranks[i], shape[i], ranks[i+1])),
                indices=(f"r_{i}", f"m_{i+1}", f"r_{i+1}")
            )
            cores.append(core)

        super().__init__(cores)

        self.shape = shape
        self.ranks = ranks
        self.order = len(shape)

    def orthonormalize(self, mode="right", start=0, stop=-1):
        """Orthonormalize the tensor train using a series of QR decompositions.

        This method is similar to the .fill method but the cores are used directly.

        Args:
            start (int, optional): Index of the first core to orthonormalize. Defaults to 0.
            stop (int, optional): Index of the last core to orthonormalize. Defaults to -1.
            orthogonality (str, optional): Orthogonality of the tensor train. Defaults to "right".

        Raises:
            ValueError: If the orthogonality is not either "left" or "right".
        """
        if stop == -1: stop = len(self.shape) - 1
        if mode == "left":
            for k in range(start, stop):
                core_curr = self.cores[f"core_{k}"]
                core_next = self.cores[f"core_{k+1}"]

                L = left_unfold(core_curr).view(np.ndarray)
                R = right_unfold(core_next).view(np.ndarray)

                V, U = np.linalg.qr(L)
                W = U @ R

                self.cores[f"core_{k}"] = TensorCore(
                    V.reshape(core_curr.shape),
                    indices=core_curr.indices, name=core_curr.name
                )
                self.cores[f"core_{k+1}"] = TensorCore(
                    W.reshape(core_next.shape),
                    indices=core_next.indices, name=core_next.name
                )
        elif mode == "right":
            for k in range(stop, start, -1):
                core_prev = self.cores[f"core_{k-1}"]
                core_curr = self.cores[f"core_{k}"]

                L = left_unfold(core_prev)
                R = right_unfold(core_curr)

                V, U = np.linalg.qr(R.T)
                W = L @ U.T

                self.cores[f"core_{k-1}"] = TensorCore(
                    W.reshape(core_prev.shape),
                    indices=core_prev.indices, name=core_prev.name
                )
                self.cores[f"core_{k}"] = TensorCore(
                    V.T.reshape(core_curr.shape),
                    indices=core_curr.indices, name=core_curr.name
                )
        else:
            raise ValueError("orthogonality must be either 'left' or 'right'")

    def randomize(self):
        for core in self.cores.values():
            core.randomize()

    def copy(self):
        tt = deepcopy(self)
        return tt

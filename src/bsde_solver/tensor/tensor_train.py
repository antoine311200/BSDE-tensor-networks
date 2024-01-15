import numpy as np

from copy import deepcopy

from bsde_solver.tensor.tensor_core import TensorCore
from bsde_solver.tensor.tensor_network import TensorNetwork


# class TensorTrain:
#     """Class dedicated to the tensor train format of a tensor also referred to as Matrix Product State (MPS).

#     The goal of this class is to provide a simple interface to manipulate tensors in the tensor train format.
#     That is, creating a random tensor train or decomposing a given tensor into a tensor train.
#     Some basic operations are also implemented such as the orthonormalization of a tensor train.
#     """

#     def __init__(self, shape, rank):
#         """Initialize a tensor train with a given shape and rank.

#         Given a shape of indices and the TT ranks, a tensor train is abstractly initialized.
#         Its diagram is given by:

#                    (s1)     (s2)     (s3)     (s4)
#                     |        |        |        |
#             --(r1)--O--(r2)--O--(r3)--O--(r4)--O--(r5)--

#         for shape = [s1, s2, s3, s4] and rank = [r1, r2, r3, r4, r5].
#         By convention, the first and last ranks are equal to 1 in most cases.

#         Args:
#             shape (list): Shape of the tensor train.
#             rank (list): Rank of the tensor train.
#         """
#         self.shape = shape
#         self.rank = rank
#         self.order = len(shape)

#         self.cores = []
#         self.indices = [
#             (f"r{i}", f"s{i+1}", f"r{i+1}")
#             for i in range(self.order)
#         ]

#     def __getitem__(self, key):
#         """Get an element of the tensor train.

#         Args:
#             key (int): Index of the element to get.

#         Returns:
#             np.ndarray: Element of the tensor train.
#         """
#         return self.cores[key]

#     def fill(self, tensor, orthogonality="left"):
#         """Fill the tensor train with a given tensor by performing a TT-SVD algorithm.

#         The principle of the TT-SVD algorithm is to decompose a tensor into a tensor train using
#         a series of SVD decompositions starting from the left or right side of the tensor.
#         This will give a canonical representation of the TT with left or right orthogonality.

#         For optimization purposes, the TT-SVD algorithm is implemented using the QR decomposition instead of the SVD.
#         This allows to reduce the complexity of the algorithm by not computing the singular values.

#         Args:
#             tensor (np.ndarray): Tensor to decompose into a tensor train.
#             orthogonality (str, optional): Orthogonality of the tensor train. Defaults to "left".

#         Raises:
#             ValueError: If the orthogonality is not either "left" or "right".
#         """
#         T = tensor

#         if orthogonality == "left":
#             pass
#         elif orthogonality == "right":
#             for k in range(len(self.shape) - 1, 0, -1):
#                 L = matricized(T, shape=self.shape[k], mode="right").T
#                 Q, R = np.linalg.qr(L, mode="complete")

#                 Q = Q[:, :self.rank[k]].T
#                 R = R[:self.rank[k], :].T

#                 self.cores[k] = tensorized(Q, shape=self.shape[k])
#                 T = R

#             self.cores[0] = tensorized(T, shape=self.shape[0])
#         else:
#             raise ValueError("orthogonality must be either 'left' or 'right'")

#     def orthonormalize(self, start=0, stop=-1, orthogonality="right", norm=True):
#         """Orthonormalize the tensor train using a series of QR decompositions.

#         This method is similar to the .fill method but the cores are used directly.

#         Args:
#             start (int, optional): Index of the first core to orthonormalize. Defaults to 0.
#             stop (int, optional): Index of the last core to orthonormalize. Defaults to -1.
#             orthogonality (str, optional): Orthogonality of the tensor train. Defaults to "right".

#         Raises:
#             ValueError: If the orthogonality is not either "left" or "right".
#         """
#         if stop == -1: stop = len(self.shape) - 1
#         if orthogonality == "left":
#             for k in range(start, stop):
#                 L = matricized(self.cores[k], mode="left")
#                 R = matricized(self.cores[k+1], mode="right")

#                 print(L.shape, R.shape)

#                 V, U = np.linalg.qr(L)
#                 W = U @ R

#                 self.cores[k] = tensorized(V, shape=self.cores[k].shape)
#                 self.cores[k+1] = tensorized(W, shape=self.cores[k+1].shape)

#             if norm:
#                 self.cores[stop] /= np.linalg.norm(self.cores[stop])
#         elif orthogonality == "right":
#             for k in range(stop, start-1, -1):
#                 L = matricized(self.cores[k-1], mode="left")
#                 R = matricized(self.cores[k], mode="right")

#                 V, U = np.linalg.qr(R.T)
#                 W = L @ U.T

#                 self.cores[k-1] = tensorized(W, shape=self.cores[k-1].shape)
#                 self.cores[k] = tensorized(V.T, shape=self.cores[k].shape)

#                 # print("v", V.T @ V)
#                 # print(matricized(self.cores[k], mode="left").T @ matricized(self.cores[k], mode="left"))

#             if norm:
#             #     # print(start, np.linalg.norm(self.cores[start]))
#                 self.cores[-1] /= np.linalg.norm(self.cores[-1]) / np.sqrt(self.cores[-1].shape[0])
#         else:
#             raise ValueError("orthogonality must be either 'left' or 'right'")

#     def random(self):
#         """Fill the tensor train with random values."""
#         for i in range(len(self.shape)):
#             self.cores.append(np.random.normal(size=(self.rank[i], self.shape[i], self.rank[i + 1])))

#     def copy(self):
#         """Copy the tensor train."""
#         tt = TensorTrain(self.shape, self.rank)
#         tt.cores = self.cores.copy()

#         return tt


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

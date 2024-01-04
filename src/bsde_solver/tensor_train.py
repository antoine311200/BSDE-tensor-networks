import numpy as np

from bsde_solver.utils import matricized, tensorized

class TensorTrain:
    """Class dedicated to the tensor train format of a tensor also referred to as Matrix Product State (MPS).

    The goal of this class is to provide a simple interface to manipulate tensors in the tensor train format.
    That is, creating a random tensor train or decomposing a given tensor into a tensor train.
    Some basic operations are also implemented such as the orthonormalization of a tensor train.
    """

    def __init__(self, shape, rank):
        """Initialize a tensor train with a given shape and rank.

        Given a shape of indices and the TT ranks, a tensor train is abstractly initialized.
        Its diagram is given by:

                   (s1)     (s2)     (s3)     (s4)
                    |        |        |        |
            --(r1)--O--(r2)--O--(r3)--O--(r4)--O--(r5)--

        for shape = [s1, s2, s3, s4] and rank = [r1, r2, r3, r4, r5].
        By convention, the first and last ranks are equal to 1 in most cases.

        Args:
            shape (list): Shape of the tensor train.
            rank (list): Rank of the tensor train.
        """
        self.shape = shape
        self.rank = rank

        self.cores = []

    def fill(self, tensor, orthogonality="left"):
        """Fill the tensor train with a given tensor by performing a TT-SVD algorithm.

        The principle of the TT-SVD algorithm is to decompose a tensor into a tensor train using
        a series of SVD decompositions starting from the left or right side of the tensor.
        This will give a canonical representation of the TT with left or right orthogonality.

        For optimization purposes, the TT-SVD algorithm is implemented using the QR decomposition instead of the SVD.
        This allows to reduce the complexity of the algorithm by not computing the singular values.

        Args:
            tensor (np.ndarray): Tensor to decompose into a tensor train.
            orthogonality (str, optional): Orthogonality of the tensor train. Defaults to "left".

        Raises:
            ValueError: If the orthogonality is not either "left" or "right".
        """
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

    def orthonormalize(self, start=0, stop=-1, orthogonality="right"):
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
        if orthogonality == "left":
            for k in range(start, stop):
                L = matricized(self.cores[k], mode="left")
                R = matricized(self.cores[k+1], mode="right")

                V, U = np.linalg.qr(L)
                W = U @ R

                self.cores[k] = tensorized(V, shape=self.cores[k].shape)
                self.cores[k+1] = tensorized(W, shape=self.cores[k+1].shape)
        elif orthogonality == "right":
            for k in range(stop, start-1, -1):
                L = matricized(self.cores[k-1], mode="left")
                R = matricized(self.cores[k], mode="right")

                V, U = np.linalg.qr(R.T)
                W = L @ U.T

                self.cores[k-1] = tensorized(W, shape=self.cores[k-1].shape)
                self.cores[k] = tensorized(V.T, shape=self.cores[k].shape)
        else:
            raise ValueError("orthogonality must be either 'left' or 'right'")

    def random(self):
        """Fill the tensor train with random values."""
        for i in range(len(self.shape)):
            self.cores.append(np.random.normal(size=(self.rank[i], self.shape[i], self.rank[i + 1])))

    def copy(self):
        """Copy the tensor train."""
        tt = TensorTrain(self.shape, self.rank)
        tt.cores = self.cores.copy()

        return tt
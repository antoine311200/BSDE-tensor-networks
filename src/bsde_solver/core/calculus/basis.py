import numpy as np
from scipy.special import legendre

class Basis:

    def eval(self, x):
        raise NotImplementedError("eval")

    def grad(self, x):
        raise NotImplementedError("grad")


class PolynomialBasis(Basis):

    def __init__(self, degree: int):
        self.degree = degree

    def eval(self, x):
        return np.array([x**i for i in range(self.degree)]).T

    def grad(self, x):
        return np.array([np.zeros(x.shape)] + [i * x ** (i - 1) for i in range(1, self.degree)]).T

class LegendreBasis(Basis):

    def __init__(self, degree: int):
        self.degree = degree

        self.coefs = [legendre(i) for i in range(self.degree)]
        self.grad_coefs = [legendre(i).deriv() for i in range(self.degree)]

    def eval(self, x):
        return np.array([np.polyval(self.coefs[i], x) for i in range(self.degree)]).T

    def grad(self, x):
        return np.array([np.polyval(self.grad_coefs[i], x) for i in range(self.degree)]).T

class WaveletBasis(Basis):

    def __init__(self, n, j) -> None:
        super().__init__()

        self.n = n
        self.j = j

    def base_func(x):
        return 2*np.sinc(2*x) - np.sinc(x)

    def grad_base_func(x):
        pass

    def eval(self, x):
        return np.array([
            1/np.power(2, j/2) * WaveletBasis.base_func((x - np.pow(2, j)*n) / np.pow(2, j))
            for j in range(-self.j, self.j) for n in range(-self.n, self.n)
        ])


if __name__ == "__main__":

    basis = LegendreBasis(4)

    x = np.arange(5)
    print(basis.eval(x))
    print(basis.grad(x))
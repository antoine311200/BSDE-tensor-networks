from bsde_solver import xp
from scipy.special import legendre

class Basis:

    def eval(self, x):
        raise NotImplementedError("eval")

    def grad(self, x):
        raise NotImplementedError("grad")

    def dgrad(self, x):
        raise NotImplementedError("dgrad")

class PolynomialBasis(Basis):

    def __init__(self, degree: int):
        self.degree = degree

    def eval(self, x):
        return xp.array([x**i for i in range(self.degree)]).T

    def grad(self, x):
        return xp.array([xp.zeros(x.shape)] + [i * x ** (i - 1) for i in range(1, self.degree)]).T

    def dgrad(self, x):
        return xp.array([xp.zeros(x.shape), xp.zeros(x.shape)] + [i * (i - 1) * x ** (i - 2) for i in range(2, self.degree)]).T

class LegendreBasis(Basis):

    def __init__(self, degree: int):
        self.degree = degree

        self.coefs = [legendre(i) for i in range(self.degree)]
        self.grad_coefs = [legendre(i).deriv() for i in range(self.degree)]

    def eval(self, x):
        return xp.array([xp.polyval(self.coefs[i], x) for i in range(self.degree)]).T

    def grad(self, x):
        return xp.array([xp.polyval(self.grad_coefs[i], x) for i in range(self.degree)]).T

    def dgrad(self, x):
        return xp.array([xp.polyval(self.grad_coefs[i].deriv(), x) for i in range(self.degree)]).T

class WaveletBasis(Basis):

    def __init__(self, n, j) -> None:
        super().__init__()

        self.n = n
        self.j = j

    def base_func(x):
        return 2*xp.sinc(2*x) - xp.sinc(x)

    def grad_base_func(x):
        pass

    def eval(self, x):
        return xp.array([
            1/xp.power(2, j/2) * WaveletBasis.base_func((x - xp.pow(2, j)*n) / xp.pow(2, j))
            for j in range(-self.j, self.j) for n in range(-self.n, self.n)
        ])


if __name__ == "__main__":

    basis = LegendreBasis(4)

    x = xp.arange(5)
    print(basis.eval(x))
    print(basis.grad(x))
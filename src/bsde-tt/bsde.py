

class BackwardSDE:
    """This class is used to represent a Partial Derivative Equation as a Backward Stochastic Differential Equation (BSDE).

    That is for a given parabolic PDE of the form:
    $$(\partial_t + L) V(x, t) + h(x, t, V(x, t), (\sigma^\intercal \Nabla V)(x, t)) = 0$$
    with
    $$L = \frac{1}{2} \sum_{i, j = 1}^d a_{i, j}(x, t) \partial_{x_i} \partial_{x_j} + \sum_{i = 1}^d b_i(x, t) \partial_{x_i}$$
    and $a_{i, j} = ((\sigma \sigma^\intercal)_{i, j})$.

    The terminal condition is given by $V(x, T) = g(x)$.
    """

    def __init__(self) -> None:
        pass

    def b(self, x, t):
        pass

    def sigma(self, x, t):
        pass

    def h(self, x, t, y, z):
        pass

    def g(self, x):
        pass
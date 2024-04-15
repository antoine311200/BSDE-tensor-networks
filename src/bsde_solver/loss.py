from bsde_solver import xp

from bsde_solver.bsde import BackwardSDE

class PDELoss:

    def __init__(self, model: BackwardSDE):
        self.model = model

    def __call__(self, t, x, v, vt, vx, vxx):
        """Compute the loss for the PDE.

        d_t v + 1/2 Tr(sigma sigma^T D^2 v) + b^T Dv + h = 0

        Args:
            t (float): Time.
            x (xp.ndarray): Space.
            v (xp.ndarray): Value function.
            vt (xp.ndarray): Time derivative of the value function.
            vx (xp.ndarray): Space derivative of the value function.
            vxx (xp.ndarray): Second space derivative of the value function.

        Returns:
            xp.ndarray: Loss.
        """
        sigma = self.model.sigma(x, t)
        print(sigma.shape)
        sigma2 = sigma.T @ sigma
        z = (vx @ sigma) if len(sigma.shape) != 1 else sigma * vx
        loss = vt + xp.sum(self.model.b(x, t) * z, axis=1)
        loss += 1/2 * xp.sum(sigma2 * vxx, axis=(1, 2))
        loss += self.model.h(x, t, v, vx)
        return loss

class ReferenceLoss:
    pass
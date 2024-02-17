import numpy as np

from bsde_solver.bsde import BackwardSDE

class PDELoss:

    def __init__(self, model: BackwardSDE):
        self.model = model

    def __call__(self, t, x, v, vt, vx, vxx):
        """Compute the loss for the PDE.

        d_t v + 1/2 Tr(sigma sigma^T D^2 v) + b^T Dv + h = 0

        Args:
            t (float): Time.
            x (np.ndarray): Space.
            v (np.ndarray): Value function.
            vt (np.ndarray): Time derivative of the value function.
            vx (np.ndarray): Space derivative of the value function.
            vxx (np.ndarray): Second space derivative of the value function.

        Returns:
            np.ndarray: Loss.
        """
        sigma = self.model.sigma(x, t)
        sigma2 = sigma.T @ sigma
        loss = vt + 1/2 * np.sum(sigma2 * vxx, axis=1)
        loss += self.model.b(x, t) @ vx + self.model.h(x, t, v, (sigma.T @ vx.T).T)
        return loss

class ReferenceLoss:
    pass
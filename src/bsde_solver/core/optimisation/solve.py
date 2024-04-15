from scipy import linalg
from bsde_solver import xp

from src.bsde_solver.core.tensor.tensor_network import TensorNetwork

def solver(A: TensorNetwork, b: TensorNetwork, method='lu'):
    if method == 'lu':
        lu_factor = linalg.lu_factor(A.view(xp.ndarray))
        x = linalg.lu_solve(lu_factor, b.view(xp.ndarray))
    elif method == 'lstsq':
        x = xp.linalg.lstsq(A.view(xp.ndarray), b.view(xp.ndarray), rcond=None)[0]
    elif method == 'qr':
        q, r = xp.linalg.qr(A.view(xp.ndarray))
        x = xp.linalg.solve(r, xp.dot(q.T, b.view(xp.ndarray)))
    elif method == 'solve':
        x = xp.linalg.solve(A.view(xp.ndarray), b.view(xp.ndarray))
    else:
        raise ValueError(f"Unknown method {method}")
    return x

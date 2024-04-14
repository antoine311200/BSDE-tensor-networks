from scipy import linalg
import numpy as np

from src.bsde_solver.core.tensor.tensor_network import TensorNetwork

def solver(A: TensorNetwork, b: TensorNetwork, method='lu'):
    if method == 'lu':
        # print(A)
        # print(A.view(np.ndarray))
        lu_factor = linalg.lu_factor(A.view(np.ndarray))
        x = linalg.lu_solve(lu_factor, b.view(np.ndarray))
    elif method == 'lstsq':
        x = np.linalg.lstsq(A.view(np.ndarray), b.view(np.ndarray), rcond=None)[0]
    elif method == 'qr':
        q, r = np.linalg.qr(A.view(np.ndarray))
        x = np.linalg.solve(r, np.dot(q.T, b.view(np.ndarray)))
    elif method == 'solve':
        x = np.linalg.solve(A.view(np.ndarray), b.view(np.ndarray))
    else:
        raise ValueError(f"Unknown method {method}")
    return x
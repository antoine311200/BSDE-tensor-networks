from bsde_solver import xp

from bsde_solver.core.tensor.tensor_train import TensorTrain
from bsde_solver.core.tensor.tensor_network import TensorNetwork


def hessian(tt: TensorTrain, x, dx, ddx, batch=False):
    if not batch: hessian = xp.zeros((tt.order, tt.order))
    else: hessian = xp.zeros((tt.order, tt.order, dx[0].shape[dx[0].indices.index("batch")], ))

    for i in range(tt.order):
        for j in range(i, tt.order):
            cores = [tt] + [x[k] for k in range(tt.order) if k != i and k != j]
            if i == j:
                cores.append(ddx[i])
            elif i < j:
                cores += [dx[i], dx[j]]
            names = [
                core.name if not isinstance(core, TensorTrain) else "tt"
                for core in cores
            ]

            term = TensorNetwork(cores=cores, names=names)
            hessian[i, j] = term.contract(batch=batch).view(xp.ndarray).squeeze()
            hessian[j, i] = hessian[i, j]

    return hessian

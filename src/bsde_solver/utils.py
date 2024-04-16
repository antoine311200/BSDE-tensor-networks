import numpy as np

def matricized(core, mode="left"):
    if mode == "right":
        return core.reshape((-1, np.prod(core.shape[1:])))
    elif mode == "left":
        return core.reshape((np.prod(core.shape[:-1]), -1))
    else:
        raise ValueError("mode must be either 'left' or 'right'")

def tensorized(core, shape):
    return core.reshape(shape)

def flatten(lst):
    return [item for sublist in lst for item in sublist]

from bsde_solver.core.tensor.tensor_network import TensorNetwork

def fast_contract(tt, x):
    results = []
    if not isinstance(x, TensorNetwork):
        x = TensorNetwork(cores=x, names=[f"x_{i}" for i in range(len(x))])
    for i, (core, phi) in enumerate(zip(tt.cores.values(), x.cores.values())):
        result = TensorNetwork(cores=[core, phi], names=[f"V_{i}", f"x_{i}"]).contract(indices=("batch", f"r_{i}", f"r_{i+1}"))
        results.append(result)
    V = TensorNetwork(cores=results, names=[f"V_{i}" for i in range(len(x.cores))]).contract(batch=True, indices=('batch',))
    return V


batch_qr = np.vectorize(np.linalg.qr, signature='(m,n)->(m,p),(p,n)')

from bsde_solver.core.tensor.tensor_core import TensorCore

def compute_solution(X, V0, basis): # (batch_size, num_assets), (TT, ), (Basis, )
    batch_size, num_assets = X.shape
    phi_X = [TensorCore(basis.eval(X[:, i]), name=f"phi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]

    print(f"phi_X: {phi_X}")
    print(f"V0: {V0}")

    print([np.array(core) for core in phi_X])

    # phi = np.stack([basis.eval(X[:, i]) for i in range(num_assets)], axis=0)
    # # print(phi.shape)

    # for k in range(batch_size):
    #     print("k:", k)
    #     print(X[k, :])
    #     print(phi[:, k])
    #     print()


    Ys = np.array(fast_contract_2(V0, phi_X))
    return Ys

from functools import partial
from typing import Callable, Any

def callable_name(any_callable: Callable[..., Any]) -> str:
    if isinstance(any_callable, partial):
        return any_callable.func.__name__

    try:
        return any_callable.__name__
    except AttributeError:
        return str(any_callable)
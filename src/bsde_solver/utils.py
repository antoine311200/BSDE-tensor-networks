from bsde_solver import xp

def matricized(core, mode="left"):
    if mode == "right":
        return core.reshape((-1, xp.prod(core.shape[1:])))
    elif mode == "left":
        return core.reshape((xp.prod(core.shape[:-1]), -1))
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

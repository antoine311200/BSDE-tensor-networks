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


batch_qr = np.vectorize(np.linalg.qr, signature='(m,n)->(m,p),(p,n)')

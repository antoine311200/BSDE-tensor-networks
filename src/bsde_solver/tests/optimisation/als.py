from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.optimisation.scalar_als import multi_als
import matplotlib.pyplot as plt
import numpy as np
import opt_einsum
import time  


def test_als_simple(batch_size, num_assets, basis_size, ranks):

    m = basis_size
    b = batch_size
    d = num_assets
    ranks = ranks

    phi_x = [2+np.random.randn(b, m) for _ in range(d)]

    struct = []
    for i in range(d):
        struct.append(phi_x[i])
        struct.append([f"batch", f"m_{i + 1}"])

    phi = opt_einsum.contract(*struct, ["batch"] + [f"m_{i + 1}" for i in range(d)]).reshape(b, m ** d)

    c = np.random.randn(m ** d)
    # c = np.arange(m ** d)
    v = np.dot(phi, c)

    

    phi_xs = [TensorCore(phi_x[i], name=f"phi_{i + 1}", indices=["batch", f"m_{i + 1}"]) for i in range(d)]

    # Single ALS
    start = time.time()
    min_tt = multi_als(phi_xs, result=v, n_iter=25, ranks=ranks)
    elaps = time.time() - start
    result = TensorNetwork(cores=[min_tt] + phi_xs, names=["tt"] + [f"phi_{i + 1}" for i in range(d)]).contract(batch=True, indices=["batch"]).view(np.ndarray).squeeze()

    # print diff nb parameters c vs min_tt
    print("Number of parameters c:", c.size)
    print("Number of parameters min_tt:", min_tt.size)

    return np.abs((result - v) / v).mean(), elaps


if __name__ == "__main__":

    batch_size = 100
    num_assets = 5
    basis_size = 4
    ranks = (1, basis_size,) + (basis_size + 6, ) * (num_assets - 3) + (basis_size, 1)

    # Influence of the batch size
    bs = [10, 100, 1000, 10000]
    errors = []
    times = []
    for b in bs:
        err, elaps = test_als_simple(b, num_assets, basis_size, ranks)
        errors.append(err)
        times.append(elaps)

    print(errors)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(bs, errors, label="Relative error")
    plt.xlabel("Batch size")
    plt.ylabel("Relative error")
    plt.title("Influence of the batch size")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(bs, times, label="Time")
    plt.xlabel("Batch size")
    plt.ylabel("Time (s)")
    plt.title("Influence of the batch size")
    plt.legend()
    plt.show()



from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.tensor.tensor_network import TensorNetwork
from bsde_solver.core.optimisation.als import ALS, ALS_regularized
from bsde_solver.core.optimisation.mals import MALS
from bsde_solver.utils import fast_contract
from bsde_solver.core.calculus.basis import PolynomialBasis

from bsde_solver import xp
from time import perf_counter
from functools import partial
import opt_einsum

def run(phis, b, algo, title=""):
    start_time_batch = perf_counter()
    ALS_result = algo(phis, b)
    end_time_batch = perf_counter() - start_time_batch
    print(f"\n{title}")
    print("Time:", end_time_batch)

    batch_phis = TensorNetwork(cores=phis)
    result = fast_contract(ALS_result, batch_phis).view(xp.ndarray).squeeze()

    diff = result - b
    l2 = xp.linalg.norm(diff)

    l1 = xp.linalg.norm(diff, ord=1)
    decomposition_size = sum((core.size for core in ALS_result))
    linear_size = phis[0].shape[1] ** len(phis)
    print(f"Number of factors in the decomposition: {decomposition_size}, number of factors in the linear application: {linear_size}, usable size: {decomposition_size/linear_size:.2%}")
    print("Reconstruction error (total)   L2:", round(l2, 4), "   L1:", round(l1, 4))
    print("Mean reconstruction error ", round(xp.mean(xp.abs(diff)), 4))
    print("Maximum reconstruction error:", round(xp.max(xp.abs(diff)), 4))
    print("Ground truth samples:", [round(c, 3) for c in b[:10]])
    print("Reconstruction samples:", [round(c, 3) for c in result[:10]])

if __name__ == "__main__":
    print(f"Running tests for the optimisation module")

    seed = 216540
    degree = 5
    num_assets = 10
    n_simulations = 1
    ranks = (1,) + (5,) * (num_assets - 1) + (1,)

    # generate linear application from R^(degree^num_assets) to R
    real = xp.random.rand(degree**num_assets)

    # generate data points in R^num_assets
    xs = xp.random.rand(n_simulations, num_assets) * 2 - 1

    # compute the value of the polynomial at the data points
    basis = PolynomialBasis(degree)
    phis = [basis.eval(xs[:, i]) for i in range(num_assets)]

    # construct the data points as a vector in R^(degree^num_assets)
    struct = []
    for i in range(num_assets):
        struct.append(phis[i])
        struct.append([f"batch", f"m_{i + 1}"])

    phi = opt_einsum.contract(*struct, ["batch"] + [f"m_{i + 1}" for i in range(num_assets)]).reshape(n_simulations, degree**num_assets)

    # apply the linear application to the data points
    b = xp.dot(phi, real)

    # convert the data points to TensorCore objects
    phis = [TensorCore(phis[i], name=f"phi_{i+1}", indices=("batch", f"m_{i+1}",)) for i in range(num_assets)]

    # run the optimisation algorithm
    run(phis, b, partial(ALS, n_iter=50, ranks=ranks), title="ALS")

from bsde_solver import reload_backend
reload_backend("cupy")
from bsde_solver import xp

from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.optimisation.als import ALS
from bsde_solver.core.calculus.basis import PolynomialBasis
from bsde_solver.utils import fast_contract

import time
from functools import partial

def generate_test_data():
    xp.random.seed(42)

    basis = PolynomialBasis(degree)

    x = xp.random.rand(n_simulations, num_assets) * 2 - 1
    phis = [TensorCore(basis.eval(x[:, i]), name=f"phi_{i+1}", indices=("batch", f"m_{i+1}"),) for i in range(num_assets)]

    payoffs = [xp.linalg.norm(x, axis=1) ** 2, xp.log(0.5 + 0.5 * xp.linalg.norm(x, axis=1) ** 2)]
    payoffs_names = ["L2 norm", "Log L2 norm"]
    return phis, payoffs, payoffs_names

if __name__ == "__main__":
    n_performance_tests = 5

    n_simulations = 2000
    degree = 5
    num_assets = 10
    dim = 5

    n_iter = 20
    ranks = (1, ) + (dim,) * (num_assets - 1) + (1, )

    als = partial(ALS, n_iter=20, ranks=(1,) + (dim,) * (num_assets - 1) + (1, ))
    optimizer = als

    print(f"Benchmarking Numpy against CuPy for {n_simulations} simulations with {num_assets} assets and degree {degree} and a rank of {dim}")
    phis, payoffs, payoffs_names = generate_test_data()

    print(f"Testing for {xp.__name__} backend")
    total_time = time.perf_counter()
    for payoff, payoff_name in zip(payoffs, payoffs_names):
        print(f"Payoff: {payoff_name}")
        top = time.perf_counter()
        for k in range(n_performance_tests):
            b = payoff

            start_time = time.perf_counter()
            result = optimizer(phis, b)
            time_opt = time.perf_counter() - start_time
            b_pred = fast_contract(result, phis)
            time_pred = time.perf_counter() - start_time - time_opt

        #     print(f"Time for optimization: {time_opt:.2f}s")
        #     print(f"Time for prediction: {time_pred:.2f}s")
        #     print(f"Error: {xp.linalg.norm((b - b_pred) / b)}")
        # print(f"Total time on {payoff_name}: {time.perf_counter() - top:.2f}s\n")

    print(f"Total time: {time.perf_counter() - total_time:.4f}s")

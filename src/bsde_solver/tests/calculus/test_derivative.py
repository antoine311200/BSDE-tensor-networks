from time import perf_counter

from bsde_solver import xp

from bsde_solver.core.tensor.tensor_train import TensorTrain, BatchTensorTrain
from bsde_solver.core.tensor.tensor_core import TensorCore
from bsde_solver.core.calculus.derivative import derivative, batch_derivative


if __name__ == "__main__":
    degree = 5
    num_assets = 5

    shape = [degree for _ in range(num_assets)]
    ranks = [1, 3, 3, 3, 3, 1]
    tt = TensorTrain(shape, ranks)
    tt.randomize()

    def poly(x, degree=10):
        return xp.array([x**i for i in range(degree)]).T

    def poly_derivative(x, degree=10):
        return xp.array([i * x ** (i - 1) for i in range(degree)]).T

    n_simulations = 1000
    phis, dphis = [], []
    xp.random.seed(0)
    for i in range(n_simulations):
        x = xp.random.rand(num_assets)

        phi = [
            TensorCore(
                poly(x[i], degree=degree), name=f"phi_{i+1}", indices=(f"m_{i+1}",)
            )
            for i in range(tt.order)
        ]
        dphi = [
            TensorCore(
                poly_derivative(x[i], degree=degree),
                name=f"dphi_{i+1}",
                indices=(f"m_{i+1}",),
            )
            for i in range(tt.order)
        ]
        phis.append(phi)
        dphis.append(dphi)

    #################### Single derivative ####################

    start_time = perf_counter()
    derivatives = [None for _ in range(n_simulations)]
    for i in range(n_simulations):
        derivatives[i] = derivative(tt, phis[i], dphis[i])

    end_time = perf_counter() - start_time
    print("Time:", end_time)

    #################### Batch derivative ####################

    phis, dphis = [], []
    xp.random.seed(0)
    for i in range(n_simulations):
        x = xp.random.rand(num_assets)

        phi = [poly(x[i], degree=degree) for i in range(tt.order)]
        dphi = [poly_derivative(x[i], degree=degree) for i in range(tt.order)]
        phis.append(phi)
        dphis.append(dphi)
    phis = xp.array(phis) # (n_simulations, tt.order, degree)
    dphis = xp.array(dphis) # (n_simulations, tt.order, degree)

    phis = phis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    dphis = dphis.transpose((1, 0, 2)) # (tt.order, n_simulations, degree)
    tphis = [TensorCore(phis[i], name=f"phi_{i+1}", indices=("batch", f"m_{i+1}")) for i in range(tt.order)]
    dphis = [TensorCore(dphis[i], name=f"dphi_{i+1}", indices=("batch", f"m_{i+1}")) for i in range(tt.order)]

    batch_tt = BatchTensorTrain.dupplicate(n_simulations, tt)

    start_time_batch = perf_counter()
    batch_derivatives = batch_derivative(batch_tt, tphis, dphis)
    end_time_batch = perf_counter() - start_time_batch
    print("Time:", end_time_batch)

    print("Speed up factor:", end_time / end_time_batch)

    print(batch_derivatives)

    # from torch import autograd, tensor, einsum

    # tsr = batch_tt.contract(batch=True).view(xp.ndarray).squeeze()
    # tsr = autograd.Variable(tensor(tsr).float(), requires_grad=True).flatten()


    # print(phis.shape)

    # prodphis = xp.einsum('a,b,c,d,e->abcde', phis[0].squeeze(), phis[1].squeeze(), phis[2].squeeze(), phis[3].squeeze(), phis[4].squeeze())
    # prodphis = autograd.Variable(tensor(prodphis).float(), requires_grad=True).flatten()

    # dot = einsum('a,a->', prodphis, tsr).squeeze().sum()
    # print(dot)
    # dot.backward(retain_graph=True)

    # grad = autograd.grad(dot, prodphis, create_graph=True)
    # print(len(grad))
    # print(grad[0].shape)
    # Check if the results are the same
    # print(xp.allclose(derivatives, batch_derivatives))
import numpy as np
from time import time

from bsde_solver.core.tensor.tensor_network import TensorCore, TensorNetwork
from bsde_solver.core.tensor.tensor_network import MatrixProductState, left_unfold, right_unfold

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

if __name__ == "__main__":

    tensor = TensorCore(np.random.rand(2, 3, 4), indices=('axis_0', 'axis_1', 'axis_2'))
    print("Original Tensor:")
    print("Data:\n", tensor)
    print("Shape:", tensor.shape_info)
    print("Indices:", tensor.indices)

    # = ((axis_a, axis_b), axis_c, (axis_d, axis_e)))
    tc = TensorCore(np.random.rand(2, 3, 4, 5, 6), indices=('axis_a', 'axis_b', 'axis_c', 'axis_d', 'axis_e'))
    print(tc)

    def deep_idx(axes):
        if isinstance(axes, tuple):
            return tuple(deep_idx(axis) for axis in axes)
        else:
            return tc.indices.index(axes)

    print(deep_idx((('axis_e', 'axis_a'), 'axis_c', ('axis_d', 'axis_b'))))

    reshaped_tensor = tensor.unfold(('axis_0', 'axis_2'), 'axis_1')
    print("\nReshaped Tensor:")
    print("Data:\n", reshaped_tensor)
    print("Shape:", reshaped_tensor.shape_info)
    print("Indices:", reshaped_tensor.indices)

    reshaped_tensor = tensor.unfold(('axis_0', 'axis_1'), 'axis_2')
    print("\nReshaped Tensor:")
    print("Data:\n", reshaped_tensor)
    print("Shape:", reshaped_tensor.shape_info)
    print("Indices:", reshaped_tensor.indices)

    reshaped_tensor = tensor.unfold((0, 1), 2)
    print("\nReshaped Tensor:")
    print("Data:\n", reshaped_tensor)
    print("Shape:", reshaped_tensor.shape_info)
    print("Indices:", reshaped_tensor.indices)

    reshaped_tensor = tensor.unfold((1, 2), 0)
    print("\nReshaped Tensor:")
    print("Data:\n", reshaped_tensor)
    print("Shape:", reshaped_tensor.shape_info)
    print("Indices:", reshaped_tensor.indices)

    print(reshaped_tensor.__dict__)

    # base_tensor = reshaped_tensor.unfold(0, 1, 2)
    # print("\nBase Tensor:")
    # print("Data:\n", base_tensor)
    # print("Shape:", base_tensor.shape_info)
    # print("Indices:", base_tensor.indices)

    print("\nRepresentation:")
    print(tensor)

    print("\n\n\n")

    tn = TensorNetwork([
        TensorCore(np.random.rand(2, 3, 4), indices=('axis_0', 'axis_1', 'axis_2')),
        TensorCore(np.random.rand(4, 5, 6), indices=('axis_2', 'axis_3', 'axis_4'))
    ])
    print(tn)

    tn.add_core(TensorCore(np.random.rand(2, 3, 4), indices=('axis_4', 'axis_5', 'axis_6')))
    tn.add_core(TensorCore(np.random.rand(2, 3, 4), indices=('axis_6', 'axis_7', 'axis_8')))
    tn.add_core(TensorCore(np.random.rand(2, 3, 4), indices=('axis_8', 'axis_9', 'axis_10')))
    print(tn)

    print(tn.cores)

    tn2 = tn.extract(['core_0', 'core_1', 'core_4'])
    print(tn2)

    print("\n\n\n")

    mps = MatrixProductState(shape=(4, 4, 4, 4, 4, 4), ranks=(1, 3, 3, 3, 3, 3, 1))
    print(mps)

    def retraction_operator(mps, i):
        return mps.extract([f'core_{j}' for j in range(mps.order) if j != i])

    def second_retraction_operator(mps, i):
        return mps.extract([f'core_{j}' for j in range(mps.order) if j != i and j != i+1])

    print("\nRetraction operators (1st, 2nd, 5th):")
    print(retraction_operator(mps, 0))
    print(retraction_operator(mps, 1))
    print(retraction_operator(mps, 4))

    print("\nSecond order retraction operators (1st, 2nd, 5th):")
    print(second_retraction_operator(mps, 0))
    print(second_retraction_operator(mps, 1))
    print(second_retraction_operator(mps, 4))

    P2 = retraction_operator(mps, 1)
    P22 = P2.copy().rename('r_*', 's_*')
    print("P2", P2)
    print("P22", P22)
    PP = P2.contract(P22)
    print("PP", PP)

    print("\n\n\nLeft and right unfoldings:")

    core = mps.cores[f'core_{0}']
    print(f"\nLeft unfolding ({core}):")
    print(left_unfold(core))
    print(f"Right unfolding ({core}):")
    print(right_unfold(core))

    core = mps.cores[f'core_{2}']
    print(f"\nLeft unfolding ({core}):")
    print(left_unfold(core))
    print(f"Right unfolding ({core}):")
    print(right_unfold(core))

    left_unfolding = left_unfold(core)
    print(left_unfolding.shape_info, left_unfolding.shape)

    print("\n\nOrthonormalization:")
    mps.randomize()

    start_time = time()
    mps.orthonormalize(mode="left")

    np.set_printoptions(precision=10)
    print("Left orthonormalization:", time() - start_time)
    np.set_printoptions(precision=2)

    print(mps)

    core = mps.cores[f'core_{1}']
    left_unfolding = left_unfold(core).view(np.ndarray)
    print(f"\nLeft unfolding ({core}):")
    print(left_unfold(core))
    identity = left_unfolding.T @ left_unfolding
    print(identity)

    core = mps.cores[f'core_{0}']
    left_unfolding = left_unfold(core).view(np.ndarray)
    print(f"\nLeft unfolding ({core}):")
    print(left_unfold(core))
    identity = left_unfolding.T @ left_unfolding
    print(identity)

    core = mps.cores[f'core_{5}']
    left_unfolding = left_unfold(core).view(np.ndarray)
    print(f"\nLeft unfolding ({core}):")
    print(left_unfold(core))
    identity = left_unfolding.T @ left_unfolding
    print(identity)

    start_time = time()
    mps.orthonormalize(mode="right")

    np.set_printoptions(precision=10)
    print("Left orthonormalization:", time() - start_time)
    np.set_printoptions(precision=2)

    core = mps.cores[f'core_{1}']
    right_unfolding = right_unfold(core).view(np.ndarray)
    print(f"\nLeft unfolding ({core}):")
    print(right_unfold(core))
    identity = right_unfolding @ right_unfolding.T
    print(identity)

    core = mps.cores[f'core_{0}']
    right_unfolding = right_unfold(core).view(np.ndarray)
    print(f"\nLeft unfolding ({core}):")
    print(right_unfold(core))
    identity = right_unfolding @ right_unfolding.T
    print(identity)

    core = mps.cores[f'core_{5}']
    right_unfolding = right_unfold(core).view(np.ndarray)
    print(f"\nLeft unfolding ({core}):")
    print(right_unfold(core))
    identity = right_unfolding @ right_unfolding.T
    print(identity)

    # print("\n\nStress test:")
    # ext = 100
    # int = 10
    # ncore = 5#000
    # huge_mps = MatrixProductState(
    #     shape=[ext for _ in range(ncore)],
    #     ranks=[1] + [int for _ in range(ncore)] + [1]
    # )

    # start_time = time()
    # huge_mps.randomize()

    # np.set_printoptions(precision=10)
    # print("Randomize time:", time() - start_time)
    # np.set_printoptions(precision=2)

    # start_time = time()
    # huge_mps.orthonormalize(mode="left")

    # np.set_printoptions(precision=10)
    # print("Left orthonormalization time:", time() - start_time)
    # np.set_printoptions(precision=2)
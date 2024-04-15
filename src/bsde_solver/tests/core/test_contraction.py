from opt_einsum import contract, contract_path
from bsde_solver import xp
from time import perf_counter

import sys

batch_size = 1000
degree = 5
num_assets = 10
rank = 4

phis = [xp.random.rand(batch_size, degree) for _ in range(num_assets)]
cores = [xp.random.rand(1, degree, rank)] + [xp.random.rand(rank, degree, rank) for _ in range(num_assets - 2)] + [xp.random.rand(rank, degree, 1)]

phis_indices = [("batch", f"m_{i+1}") for i in range(num_assets)]
cores_indices = [(f"r_{i}", f"m_{i+1}", f"r_{i+1}") for i in range(num_assets)]

skip_core_idx = num_assets // 2

# Optimized contraction in batch mode using opt_einsum
struct = []
for i in range(num_assets):
    struct.append(phis[i])
    struct.append(phis_indices[i])

    if i != skip_core_idx:
        struct.append(cores[i])
        struct.append(cores_indices[i])

start_time = perf_counter()
# contract_path(*struct)
print(struct[1::2])
print(["batch", f"r_{skip_core_idx}", f"r_{skip_core_idx+1}"])
print(contract_path(*struct))
# struct += ["batch", ...]#f"r_{skip_core_idx}", f"r_{skip_core_idx+1}"]
result = contract(*struct, optimize="auto")
end_time = perf_counter() - start_time
print("Time:", end_time)

print(result.shape)

# Optimized contraction in single mode using opt_einsum x batch_size

results = []
structs = []
for j in range(batch_size):
    struct = []
    for i in range(num_assets):
        struct.append(phis[i][j])
        struct.append((phis_indices[i][0], ))

        if i != skip_core_idx:
            struct.append(cores[i])
            struct.append(cores_indices[i])
    structs.append(struct)

start_time = perf_counter()
for struct in structs:
    # contract_path(*struct)
    struct += [f"r_{skip_core_idx}", f"r_{skip_core_idx+1}"]
    result = contract(*struct, optimize="auto")
end_time = perf_counter() - start_time
print("Time:", end_time)

results.append(result)
results = xp.stack(results)
print(results.shape)
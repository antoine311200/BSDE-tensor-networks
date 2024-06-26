from __future__ import annotations

from bsde_solver import xp
from copy import deepcopy
from opt_einsum import contract

from bsde_solver.core.tensor.tensor_core import TensorCore

class TensorNetwork:

    def __init__(self, cores: list[TensorCore | TensorNetwork] | dict[str, TensorCore], names: list[str] = None):
        """Initialize a tensor network with a given list of tensor cores."""
        if isinstance(cores, dict):
            self.cores = cores
        else:
            self.cores = {}
            for i, core in enumerate(cores):
                if core is None: continue
                if names: name = names[i]
                elif core.name: name = core.name
                else: name = f"core_{i}"

                if hasattr(core, 'cores'):
                    for subname, core in core.cores.items():
                        core.name = name+'_'+subname
                        self.add_core(core, name+'_'+subname)
                else:
                    core.name = name
                    self.add_core(core, name)

    def add_core(self, core: TensorCore, name: str = None):
        if name in self.cores:
            raise ValueError(f"Tensor core with name {name} already exists.")
        core.name = name
        self.cores[name] = core

    def extract(self, core_names: list[str]):
        new_cores = {
            name: self.cores[name]
            for name in core_names
        }
        return TensorNetwork(new_cores)

    def contract(self, tn:  TensorNetwork | TensorCore = None, scheme = None, indices: list[str] = None, batch: bool = False, optimize: str = 'auto') -> TensorCore:
        struct = []
        all_indices = []
        for core in self.cores.values():
            struct.append(core)
            struct.append(core.indices)
            all_indices += core.indices

        if isinstance(tn, TensorNetwork): tn_cores = tn.cores.values()
        elif isinstance(tn, TensorCore): tn_cores = [tn]
        else: tn_cores = []
        for core in tn_cores:
            struct.append(core)
            struct.append(core.indices)

        if not indices:
            indices_1 = [idx for core in self.cores.values() for idx in core.indices]
            indices_2 = [idx for core in tn_cores for idx in core.indices]
            indices = indices_1 + indices_2

            unique_indices = [e for e in indices if indices.count(e) == 1]
        else:
            unique_indices = indices
            # Check if there are indices that does not appear in the network
            # if not, add them to the unique_indices
            new_indices = [idx for idx in indices if idx not in all_indices]
            if len(new_indices) > 0:
                # Add new axis to the last core
                if new_indices:
                    for idx in new_indices:
                        struct[-2] = struct[-2].expand_dims(-1, name=idx, inplace=False)
                    struct[-1] = struct[-2].indices

        if batch:
            unique_indices = list(set(unique_indices))
            if 'batch' not in unique_indices: unique_indices.append('batch')

        struct.append(unique_indices)
        result = contract(*struct, optimize=optimize)

        core = TensorCore(result, indices=unique_indices)
        return core

    def rename(self, old_name: str, new_name: str, inplace: bool = True):
        cores = self.cores if inplace else deepcopy(self.cores)

        for _, core in cores.items():
            core.rename(old_name, new_name, inplace=True)

        return self if inplace else TensorNetwork(cores)

    def copy(self):
        return TensorNetwork(deepcopy(self.cores))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.cores[key]
        else:
            key = list(self.cores.keys())[key]
            return self.cores[key]

    def __repr__(self):
        info = ",\n    ".join([str(core) for core in self.cores.values()])
        return f"TensorNetwork(\n    {info}\n)"

    def sample(self, idx: int) -> TensorCore:
        batch = all(['batch' in core.indices for core in self.cores.values()])
        if batch:
            sampled_cores = []
            for core in self.cores.values():
                batch_index = core.indices.index('batch')
                permutation = list(range(len(core.indices)))
                permutation[0], permutation[batch_index] = permutation[batch_index], permutation[0]
                transposed_core = core.view(type=xp.ndarray).transpose(permutation)
                subtensor = transposed_core[idx, ...]
                sampled_cores.append(TensorCore(subtensor, indices=core.indices[1:]))
            return TensorNetwork(sampled_cores, names=list(self.cores.keys()))
        else:
            return self

from __future__ import annotations
from typing import Any

import re
import numpy as np
from copy import deepcopy
from opt_einsum import contract

from bsde_solver.tensor.tensor_core import TensorCore

class TensorNetwork:

    def __init__(self, cores: list[TensorCore] | dict[str, TensorCore]):
        """Initialize a tensor network with a given list of tensor cores."""
        if isinstance(cores, dict):
            self.cores = cores
        else:
            self.cores = {}
            for i, core in enumerate(cores):
                self.add_core(core)

    def add_core(self, core: TensorCore):
        name = core.name if core.name is not None else f"core_{len(self.cores)}"
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

    def contract(self, tn: TensorNetwork = None, indices: list[str] = None) -> TensorCore:
        struct = []
        for core in self.cores.values():
            struct.append(core)
            struct.append(core.indices)

        tn_cores = tn.cores.values() if tn is not None else []
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

        struct.append(unique_indices)
        result = contract(*struct)

        core = TensorCore(result, indices=unique_indices)
        return core

    def rename(self, old_name: str, new_name: str):
        if "*" in old_name: old_name = old_name.replace("*", "(\d+)")
        for _, core in self.cores.items():
            new_indices = []
            for idx in core.indices:
                new_indices.append(re.sub(
                    old_name.replace('*', '(.*)'),
                    new_name.replace('*', r'\1'),
                    idx
                ))
            core.indices = tuple(new_indices)

        return self


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
import re
import numpy as np

from copy import deepcopy

class TensorCore(np.ndarray):
    """Create a tensor core object as a subclass of np.ndarray.

    A tensor core is represented as a numpy array with an additional attribute "shape" which is a tuple of the shape of the tensor core,
    as well as an additional attribute 'legs' which is a tuple of strings representing the indices of the tensor core.
    """

    def __new__(cls, input_array, indices=None, name=None):
        obj = np.asarray(input_array).view(cls)
        obj.name = name
        obj.shape_info = tuple(obj.shape)
        obj.indices = indices if indices is not None else cls._generate_indices(obj.shape_info)

        obj.base_shape = obj.shape
        obj.base_indices = obj.indices
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.shape_info = getattr(obj, 'shape_info', None)
        self.indices = getattr(obj, 'indices', None)
        self.name = getattr(obj, 'name', None)

        self.base_shape = getattr(obj, 'base_shape', None)
        self.base_indices = getattr(obj, 'base_indices', None)

    def __str__(self):
        prefix = self.name+": " if self.name is not None else ""
        info = ", ".join([f"{self.indices[i]} {{{self.shape_info[i]}}}" for i in range(len(self.shape_info))])
        return prefix+f"TensorCore({info})"

    def __repr__(self):
        info = ", ".join([f"{self.indices[i]} {{{self.shape_info[i]}}}" for i in range(len(self.shape_info))])
        return f"TensorCore({info})"

    @staticmethod
    def _generate_indices(shape_info):
        return tuple(f"axis_{i}" for i in range(len(shape_info)))

    def like(input_array, like_core: 'TensorCore' = None, name=None):
        return TensorCore(input_array.reshape(like_core.shape),
            indices=like_core.indices, name=like_core.name if name is None else name
        )

    def unfold(self, *args, **kwargs):

        def merge_axes(axes):
            indices, shapes = [], []

            visited = []
            idx = None
            for i, item in enumerate(axes):
                if isinstance(item, tuple):
                    item = [
                        self.indices[idx]
                        if isinstance(idx, int) else idx
                        for idx in item
                    ]
                    visited += item
                    indices.append("+".join(item))
                    shapes.append(np.prod([
                        self.shape_info[self.indices.index(idx)]
                        for idx in item
                    ]))
                elif item != -1:
                    item = self.indices[item] if isinstance(item, int) else item
                    visited.append(item)
                    indices.append(item)
                    shapes.append(self.shape_info[self.indices.index(item)])
                else:
                    indices.append(-1)
                    shapes.append(-1)
                    idx = i

            # Replace result[idx] by the missing indices
            if idx:
                unseen_indices = [item for item in self.indices if item not in visited]
                indices[idx] = '+'.join(unseen_indices)
                shapes[idx] = np.prod([
                    self.shape_info[idx]
                    for idx in [
                        self.indices.index(idx)
                        for idx in unseen_indices
                    ]
                ])

            return tuple(indices), tuple(shapes)

        new_indices, reshape_axes = merge_axes(args)

        # Transpose axis to match the new indices
        flatten_indices = [idx.split('+') for idx in new_indices]
        flatten_indices = [ idx for sublist in flatten_indices for idx in sublist]
        transpose_axes = [
            self.indices.index(idx)
            for idx in flatten_indices
        ]

        new_array = super().transpose(*transpose_axes)
        new_array = new_array.reshape(*reshape_axes)
        new_array.indices = new_indices
        new_array.shape_info = reshape_axes

        return new_array

    def randomize(self):
        self[:] = (np.random.rand(*self.shape_info) - 0.5) * 2

    def rename(self, old_name: str, new_name: str, inplace: bool = True):
        if "*" in old_name: old_name = old_name.replace("*", "(\d+)")
        new_indices = []
        for idx in self.indices:
            new_indices.append(re.sub(
                old_name.replace('*', '(.*)'),
                new_name.replace('*', r'\1'),
                idx
            ))

        if inplace:
            self.indices = tuple(new_indices)
        else:
            return TensorCore(deepcopy(self), indices=tuple(new_indices), name=self.name)

        return self
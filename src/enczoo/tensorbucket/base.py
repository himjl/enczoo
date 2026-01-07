from pathlib import Path
from typing import List, Union, Dict, Tuple

import h5py
import numpy as np


# %%
class TensorBucket:

    def __init__(
            self,
            loc: Path,
            shape: Union[int, Tuple[int, ...]],  # The shape of a single entry
    ):
        if isinstance(shape, int):
            shape = shape,
        self.shape: Tuple[int, ...] = shape

        if not isinstance(loc, Path):
            raise ValueError(f'loc must be a Path, not {type(loc)}')

        if not loc.suffix == '.h5':
            raise ValueError(f'loc must be an .h5 file, not {loc.suffix}')

        self._filepath = loc

        # Initialize the key cache
        self._key_to_i: Dict[str, int] = {key: i for (i, key) in enumerate(self.list_keys())}

    @property
    def filepath(self) -> Path:
        if not self._filepath.exists():
            self._filepath.parent.mkdir(parents=True, exist_ok=True)

            with h5py.File(self._filepath, 'w') as f:
                # Initialize the file
                f.create_dataset(
                    name='tensors',
                    shape=(0, *self.shape),
                    maxshape=(None, *self.shape),
                    chunks=(1, *self.shape),
                )

                f.create_dataset(
                    name='keys',
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(1000,),
                    dtype=h5py.string_dtype(),
                )
                pass
            print('Started new tensor cache at', self.filepath)
        return self._filepath

    def check_keys_exist(self, keys: List[str]) -> List[bool]:
        key_set = self._key_to_i.keys()
        return [key in key_set for key in keys]

    def list_keys(self) -> List[str]:
        with h5py.File(self.filepath, 'r') as f:
            keys = f['keys'][()]

        # Decode the keys from bytes to str:
        keys = keys.astype('U')
        return keys

    def store_tensors(
            self,
            key_to_tensor: Dict[str, np.ndarray],
            overwrite_if_exists: bool = False
    ):

        if len(key_to_tensor) == 0:
            return

        with h5py.File(self.filepath, 'a') as f:

            key_group = f['keys']
            tensor_group = f['tensors']
            for key, tensor in key_to_tensor.items():
                if tensor.shape != self.shape:
                    raise ValueError(f'tensor shape {tensor.shape} does not match expected shape {self.shape}')

                index = self._get_key_index_or_none(key)

                if index is not None:
                    if not overwrite_if_exists:
                        raise ValueError(f'Tensor with key "{key}" already exists. Set overwrite_if_exists=True to overwrite.')

                if not isinstance(tensor, np.ndarray):
                    raise ValueError(f'tensor must be a numpy.ndarray, not {type(tensor)}')

                if index is None:
                    # Append the tensor to the end of the dataset
                    key_group.resize((key_group.shape[0] + 1,))
                    tensor_group.resize((tensor_group.shape[0] + 1, *self.shape))

                    # Assign the tensor to the last index
                    key_group[-1] = key
                    tensor_group[-1] = tensor

                    # Update the key cache
                    self._key_to_i[key] = key_group.shape[0] - 1
                else:
                    # Overwrite the tensor at the index
                    assert key_group[index] == key.encode('utf-8'), f'Key mismatch: {key_group[index]} != {key}? Bug.'
                    tensor_group[index] = tensor

    def retrieve_tensors(self, keys: List[str]) -> List[np.ndarray]:
        if len(keys) == 0:
            return []

        # Get unique keys:
        key_to_index: Dict[str, int] = {}
        for key in set(keys):
            index = self._get_key_index_or_none(key)

            if index is None:
                raise KeyError(f'{key} not found in TensorBucket.')

            key_to_index[key] = index

        # Load from H5PY:
        key_to_array = {}
        with h5py.File(self.filepath, 'r') as f:
            group = f['tensors']
            for key, index in key_to_index.items():
                key_to_array[key] = group[index][()]

        # Assemble return
        tensors = [key_to_array[key] for key in keys]
        return tensors

    def delete_tensors(self, keys: List[str]):
        if len(keys) == 0:
            return

        raise NotImplementedError('Deleting tensors is not yet implemented. This is a non-trivial operation because it requires shifting all tensors after the deleted tensor to the left. This is not currently supported by h5py.')

    def __len__(self) -> int:
        return len(self._key_to_i)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(nkeys={len(self)})"

    def __repr__(self) -> str:
        return self.__str__()

    def _get_key_index_or_none(self, key: str) -> Union[int, None]:
        """
        Returns the linear index of the key. If it does not exist, return None.
        :param keys:
        :return:
        """

        if key not in self._key_to_i:
            return None
        return self._key_to_i[key]

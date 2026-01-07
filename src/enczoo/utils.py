import hashlib
import itertools
from typing import Iterable, Iterator, List, TypeVar

import torch

T = TypeVar("T")


def iterate_batches(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1!")

    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if len(batch) == 0:
            return
        yield batch


def hash_torch_module(module: torch.nn.Module) -> str:
    """
    Returns a hash of a torch.nn.Module. The hash function depends on:
    - The model's state_dict
    - The model's string representation

    :param module:
    :return:
    """

    sha256_hash = hashlib.sha256()

    # Hash the model's state_dict:
    state_dict = module.state_dict()
    for key in sorted(state_dict.keys()):
        tensor_value = state_dict[key].detach().cpu().numpy()
        sha256_hash.update(tensor_value.tobytes())

    # Hash the module's string representation:
    module_string = str(module)
    module_string = module_string.encode("utf-8")
    sha256_hash.update(module_string)

    # Return the combined hash:
    return sha256_hash.hexdigest()

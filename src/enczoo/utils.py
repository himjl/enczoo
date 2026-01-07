import hashlib
import itertools
from typing import Iterable, Iterator, List, TypeVar

import torch

T = TypeVar("T")


def iterate_batches(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Yield items from an iterable in fixed-size batches.

    Args:
        iterable: Source iterable to batch.
        batch_size: Number of items per batch.

    Yields:
        Lists of up to batch_size elements.

    Raises:
        ValueError: If batch_size is less than 1.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1!")

    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if len(batch) == 0:
            return
        yield batch


def hash_torch_module(module: torch.nn.Module) -> str:
    """Return a hash for a torch.nn.Module.

    The hash depends on the module's state_dict keys, shapes, dtypes, and values.

    Args:
        module: Module to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """

    sha256_hash = hashlib.sha256()

    # Hash the model's state_dict, including key names for structural stability.
    state_dict = module.state_dict()
    for key in sorted(state_dict.keys()):
        sha256_hash.update(key.encode("utf-8"))
        tensor = state_dict[key].detach().cpu()
        sha256_hash.update(str(tensor.dtype).encode("utf-8"))
        sha256_hash.update(str(tuple(tensor.shape)).encode("utf-8"))
        tensor_value = tensor.numpy()
        sha256_hash.update(tensor_value.tobytes())

    # Return the combined hash:
    return sha256_hash.hexdigest()

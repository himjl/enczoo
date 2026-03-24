import itertools
from typing import Iterable, Iterator, List, TypeVar

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

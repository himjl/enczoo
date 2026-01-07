import numpy as np
from pathlib import Path

from enczoo.tensorbucket.base import TensorBucket


def test_tensor_bucket(tmpdir):
    shape = (3, 3)
    test_cache = TensorBucket(loc=Path(tmpdir) / "test.h5", shape=shape)

    assert len(test_cache.list_keys()) == 0

    t = np.random.rand(*shape)
    test_cache.store_tensors({"test": t}, overwrite_if_exists=True)
    assert len(test_cache.list_keys()) == 1

    data = np.array(test_cache.retrieve_tensors(["test"]))
    assert np.allclose(data, t)

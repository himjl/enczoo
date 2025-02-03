import functools
from typing import Tuple, Union

import torch
import torch.nn as nn
import numpy as np

class RandomProjection(nn.Module):
    """
    A torch.nn.Module which efficiently performs a random projection on an incoming
    tensor layer.
    """

    def __init__(
            self,
            output_shape: Union[int, Tuple[int, ...]],
            seed: int,
            max_projection_floats: int = 128000000,
    ):
        """

        :param output_shape: The shape of the output tensor.
        :param seed: The seed for the random projection.
        :param max_projection_floats: The maximum number of projection weights which may be held in memory at once.
        """
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        super().__init__()
        self.register_buffer(
            name='seed',
            tensor=torch.tensor(seed, dtype=torch.int64),
            persistent=True,
        )

        self.register_buffer(
            name='output_shape',
            tensor=torch.tensor(output_shape, dtype=torch.int64),
            persistent=True,
        )

        # Get the total number of output features
        self._num_out_features = functools.reduce(lambda x, y: x * y, output_shape)
        self.max_projection_floats = max_projection_floats

        # Register a parameter for hashing purposes
        self._representative_weight = torch.nn.Parameter(
            self._get_projection_submatrix(
                p_lb=0,
                p_ub=1,
                p_total=3,
                d=4,
                seed=0,
                device=torch.device('cpu'),
                dtype=torch.float32
            ),
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape the input tensor
        batch_size = x.shape[0]
        A = x.reshape(batch_size, -1)  # [b, d]
        num_in_features = A.shape[1]

        # Preallocate the output tensor of shape [b, p]
        C = torch.zeros(
            (batch_size, self._num_out_features),
            dtype=A.dtype,
            device=A.device
        )

        # Perform the projection in chunks by iterating over the columns of C
        noutput_dims_per_multiply = max(min(self.max_projection_floats // num_in_features, self._num_out_features), 1)

        # Iterate over the columns of the output
        for p_lb in range(0, self._num_out_features, noutput_dims_per_multiply):
            p_ub = min(p_lb + noutput_dims_per_multiply, self._num_out_features)

            # Allocate the Rchunk matrix – todo, most of the slowdown occurs here
            Rchunk = self._get_projection_submatrix(
                p_lb=p_lb,
                p_ub=p_ub,
                p_total=self._num_out_features,
                seed=self.seed,
                d=num_in_features,
                device=A.device,
                dtype=A.dtype,
            )

            # Execute the chunk matrix multiplication
            C[:, p_lb:p_ub] += A @ Rchunk

        # Reshape the output tensor
        return C.view(batch_size, *self.output_shape)

    @staticmethod
    def _get_projection_submatrix(
            p_lb: int,
            p_ub: int,
            p_total: int,
            d: int,
            seed: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Get the p_lb to p_ub columns of the (d, p) random projection matrix given by seed.
        :param seed:
        :return:
        """

        Rchunk = torch.zeros((d, p_ub - p_lb), dtype=dtype, device=device)

        for j, p in enumerate(range(p_lb, p_ub)):
            # Combine p and seed to get a unique seed for this column
            column_seed = int(seed + p)
            with torch.random.fork_rng():
                torch.manual_seed(column_seed)

                # These calls incur all the cost:
                Rchunk[:, j] = torch.randn(size=(d,)) / np.sqrt(d * p_total)

        return Rchunk

    def __repr__(self):
        return f"RandomProjection(output_shape={self.output_shape.item()}, seed={self.seed})"


# %%
if __name__ == '__main__':
    rp = RandomProjection(output_shape=10, seed=0, max_projection_floats=4096)
    import numpy as np

    np.random.seed(0)
    x = torch.tensor(np.random.rand(2, 3, 4), dtype=torch.float32)
    y = rp(x)
    print(list(rp.parameters()))

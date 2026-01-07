import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomProjection(nn.Module):
    """
    A torch.nn.Module which efficiently performs a random projection on an incoming
    tensor layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int,
    ):
        """
        :param input_shape: The shape of the input tensor, not including the batch dimension.
        :param output_shape: The shape of the output tensor, not including the batch dimension.
        :param seed: The seed for the random projection.
        :param max_projection_floats: The maximum number of projection weights which may be held in memory at once.
        """

        super().__init__()
        self.train(mode=False)

        # Register inputs as buffers; these will constitute the module's hash.
        self.register_buffer(
            "seed", torch.tensor(seed, dtype=torch.int64, requires_grad=False)
        )
        self.register_buffer(
            "in_features",
            torch.tensor(in_features, dtype=torch.int64, requires_grad=False),
        )
        self.register_buffer(
            "out_features",
            torch.tensor(out_features, dtype=torch.int64, requires_grad=False),
        )

        with torch.random.fork_rng():
            # Set the weights from a standard normal distribution:
            torch.manual_seed(seed)
            weights = torch.randn(
                size=(out_features, in_features),
                dtype=torch.float32,
                requires_grad=False,
            ) / math.sqrt(in_features * out_features)

        self.register_buffer("projection_weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.projection_weights)

    def __repr__(self):
        return f"RandomProjection(in_features={self.in_features}, out_features={self.out_features}, seed={self.seed})"

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.projection_weights.shape[0],)

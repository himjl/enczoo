import math
from typing import Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class RandomProjection(nn.Module):
    """Apply a fixed random projection to an input tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int,
    ):
        """Initialize the random projection.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            seed: Seed for the random projection weights.
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
        """Project input features with a fixed random matrix.

        Args:
            x: Input tensor of shape [B, in_features].

        Returns:
            Projected tensor of shape [B, out_features].
        """
        weights = cast(torch.Tensor, self.projection_weights)
        return F.linear(x, weights)

    def __repr__(self):
        """Return a concise representation for debugging."""
        return f"RandomProjection(in_features={self.in_features}, out_features={self.out_features}, seed={self.seed})"

    @property
    def output_shape(self) -> Tuple[int]:
        """Return the output feature shape (excluding batch dimension)."""
        weights = cast(torch.Tensor, self.projection_weights)
        return (weights.shape[0],)

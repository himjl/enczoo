import torch
import torch.nn as nn
from typing import Tuple
import math


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

        # Register seed as buffer
        self.register_buffer('seed', torch.tensor(seed, dtype=torch.int64, requires_grad=False))

        # Initialize the module
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            dtype=torch.float32,
        )

        # Register scale parameter
        invscale_value = torch.tensor(math.sqrt(in_features * out_features), requires_grad=False, dtype=torch.float32)
        self.invscale = nn.Parameter(data=invscale_value, requires_grad=False)
        torch.round(self.invscale, out=self.invscale, decimals=5)  # Round to address non-deterministic floating point runoff across platforms

        # Turn off gradient tracking
        self.linear.requires_grad_(requires_grad=False)

        with torch.random.fork_rng():
            # Set the weights from a standard normal distribution:
            torch.manual_seed(seed)
            self.linear.weight[:] = torch.randn(
                size=(out_features, in_features),
                requires_grad=False
            )

            # Round the weights to address non-deterministic floating point runoff across platforms:
            torch.round(self.linear.weight, out=self.linear.weight, decimals=1) # todo revert decimals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) / self.invscale

    def __repr__(self):
        return f"RandomProjection(in_features={self.linear.weight.shape[1]}, out_features={self.linear.weight.shape[0]}, seed={self.seed})"

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.linear.weight.shape[0],)

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

        # Register inputs as buffers; these will constitute the module's hash.
        self.register_buffer('seed', torch.tensor(seed, dtype=torch.int64, requires_grad=False))
        self.register_buffer('in_features', torch.tensor(in_features, dtype=torch.int64, requires_grad=False))
        self.register_buffer('out_features', torch.tensor(out_features, dtype=torch.int64, requires_grad=False))

        # Initialize the linear map. This has proven impossible so far to hash consistently, due to floating point runoff, so it is not registered.
        # See: https://discuss.pytorch.org/t/saving-nn-module-to-parent-nn-module-without-registering-paremeters/132082/6
        self.linear_wrapper = [nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            dtype=torch.float32,
        )]

        # Turn off gradient tracking
        self.linear_wrapper[0].requires_grad_(requires_grad=False)

        with torch.random.fork_rng():
            # Set the weights from a standard normal distribution:
            torch.manual_seed(seed)
            self.linear_wrapper[0].weight[:] = torch.randn(
                size=(out_features, in_features),
                requires_grad=False
            ) / math.sqrt(in_features * out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_wrapper[0](x)

    def __repr__(self):
        return f"RandomProjection(in_features={self.in_features}, out_features={self.out_features}, seed={self.seed})"

    @property
    def output_shape(self) -> Tuple[int]:
        return (self.linear.weight.shape[0],)

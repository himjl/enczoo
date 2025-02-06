import torch
import torch.nn as nn
from typing import Tuple


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
        self.register_buffer(
            name='seed',
            tensor=torch.tensor(seed, dtype=torch.int64),
            persistent=True,
        )

        # Record the output features
        self._out_features = out_features

        # Initialize the module
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False
        )
        # Turn off gradient tracking
        self.linear.requires_grad_(requires_grad=False)

        # Set the weights from a standard normal distribution
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.linear.weight[:] = torch.randn(
            size=(out_features, in_features),
            generator=gen,
            requires_grad=False
        )

        with torch.no_grad():
            # Scale the weights
            self.linear.weight /= np.sqrt(in_features * out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def __repr__(self):
        return f"RandomProjection(in_features={self.linear.weight.shape[1]}, out_features={self.linear.weight.shape[0]}, seed={self.seed})"

    @property
    def output_shape(self) -> Tuple[int]:
        return (self._out_features,)


# %%
if __name__ == '__main__':
    rp = RandomProjection(out_features=10, in_features=100, seed=0)
    import numpy as np

    np.random.seed(0)
    x = torch.tensor(np.random.rand(1, 100), dtype=torch.float32)
    y = rp(x)
    print(list(rp.parameters()))

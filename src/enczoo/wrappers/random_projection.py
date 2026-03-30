import math
from typing import cast

import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from enczoo.base import DeviceType, ImageEncoding, TorchImageEncoding


class RandomProjectionLayer(nn.Module):
    """Apply a fixed random projection to a 2D input tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: int,
    ):
        """Initialize the random projection."""
        super().__init__()
        self.train(mode=False)

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
            torch.manual_seed(seed)
            weights = torch.randn(
                size=(out_features, in_features),
                dtype=torch.float32,
                requires_grad=False,
            ) / math.sqrt(out_features)

        self.register_buffer("projection_weights", weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features with a fixed random matrix."""
        weights = cast(torch.Tensor, self.projection_weights)
        return F.linear(x, weights)

    def __repr__(self):
        """Return a concise representation for debugging."""
        return f"RandomProjectionLayer(in_features={self.in_features}, out_features={self.out_features}, seed={self.seed})"


class RandomProjection(TorchImageEncoding):
    """Wrap an image encoder with a fixed random projection."""

    def __init__(
        self,
        encoder: ImageEncoding,
        out_features: int,
        seed: int,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize the projection wrapper."""
        super().__init__(device=device, device_index=device_index)
        self.encoder = encoder
        self.out_features = out_features
        self.seed = seed

        in_features = int(np.prod(self.encoder.output_shape, dtype=np.int64))
        self.layer = RandomProjectionLayer(
            in_features=in_features,
            out_features=out_features,
            seed=seed,
        )
        self.layer = self.layer.to(self.torch_device)

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the projected feature shape."""
        return (self.out_features,)

    def _images_to_features(
        self,
        images: list[PIL.Image.Image],
        seed: int | None = None,
    ) -> torch.Tensor:
        """Project the wrapped encoder's flattened features."""
        features = self.encoder.compute_features(images=images, seed=seed)
        features = features.reshape(features.shape[0], -1)

        return self.layer(
            torch.from_numpy(features).to(
                device=self.torch_device,
                dtype=torch.float32,
            )
        )

from abc import ABC, abstractmethod

import PIL.Image
import numpy as np
import torch

from enczoo.base import DeviceType, ImageEncoding


class TorchImageEncoding(ImageEncoding, ABC):
    """Torch-backed image encoder with shared device and execution handling."""

    def __init__(
        self,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize backend-neutral state and resolve the torch device."""
        super().__init__(device=device, device_index=device_index)
        self._torch_device = self._resolve_torch_device()

    @property
    def torch_device(self) -> torch.device:
        """Return the resolved torch device for this encoder."""
        return self._torch_device

    @property
    def training(self) -> bool:
        """Expose a module-like training flag for compatibility."""
        modules = [
            value
            for value in self.__dict__.values()
            if isinstance(value, torch.nn.Module)
        ]
        if not modules:
            return False
        return any(module.training for module in modules)

    def compute_features(
        self,
        images: list[PIL.Image.Image],
        flatten: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        """Compute torch features and return them as a NumPy array."""
        self.validate_images(images)

        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                torch_features = self.forward(images=images, flatten=flatten, seed=seed)
                return torch_features.detach().cpu().numpy()

    def forward(
        self,
        images: list[PIL.Image.Image],
        flatten: bool = False,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Compute torch features for a batch of images."""
        self.validate_images(images)
        features = self._images_to_features(images=images, seed=seed)
        if flatten:
            features = features.reshape(features.shape[0], -1)
        return features

    @abstractmethod
    def _images_to_features(
        self,
        images: list[PIL.Image.Image],
        seed: int | None = None,
    ) -> torch.Tensor:
        """Convert images to torch features."""
        raise NotImplementedError

    def _resolve_torch_device(self) -> torch.device:
        """Resolve the backend-neutral device to a concrete torch device."""
        if self.device == "cpu":
            return torch.device("cpu")

        if torch.cuda.is_available():
            gpu_index = 0 if self.device_index is None else self.device_index
            device_count = torch.cuda.device_count()
            if gpu_index >= device_count:
                raise ValueError(
                    f"Requested PyTorch GPU index {gpu_index}, but only {device_count} CUDA device(s) are available."
                )
            return torch.device(f"cuda:{gpu_index}")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            gpu_index = 0 if self.device_index is None else self.device_index
            if gpu_index != 0:
                raise ValueError(
                    "PyTorch MPS exposes a single GPU and only supports device_index=0."
                )
            return torch.device("mps")

        raise ValueError(
            "device='gpu' was requested, but PyTorch could not find an available GPU backend."
        )

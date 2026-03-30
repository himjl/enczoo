from abc import ABC, abstractmethod
from typing import Any, Literal

import PIL.Image
import numpy as np
import torch

DeviceType = Literal["cpu", "gpu"]


class ImageEncoding(ABC):
    """Framework-agnostic interface for mapping PIL images to NumPy features."""

    def __init__(
        self,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize cached metadata and execution-device preferences.

        Args:
            device: Whether computations should run on the CPU or a GPU.
            device_index: Optional zero-based GPU index. This is only valid when
                device="gpu".
        """
        self._output_shape: tuple[int, ...] | None = None
        self._device = device
        self._device_index = device_index
        self._validate_device_configuration()

    @property
    def device(self) -> DeviceType:
        """Return the requested backend-neutral device kind."""
        return self._device

    @property
    def device_index(self) -> int | None:
        """Return the requested zero-based GPU index, if any."""
        return self._device_index

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the output feature shape (excluding batch dimension)."""
        if self._output_shape is None:
            test_image = PIL.Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            test_result = self.compute_features(images=[test_image])
            if not isinstance(test_result, np.ndarray):
                raise ValueError(
                    f"Expected a np.ndarray from compute_features, but got {type(test_result)}"
                )
            if not test_result.shape[0] == 1:
                raise ValueError(
                    f"Expected a batch size of 1, but got {test_result.shape}"
                )
            if len(test_result.shape) == 1:
                output_shape = tuple()
            else:
                output_shape = test_result.shape[1:]

            self._output_shape = tuple(output_shape)

        return self._output_shape

    @staticmethod
    def validate_images(images: list[PIL.Image.Image]) -> None:
        """Validate that the input is a non-empty image list."""
        if not isinstance(images, list):
            raise ValueError(
                f"Expected a list of PIL.Image.Images, but got {type(images)}"
            )
        if len(images) == 0:
            raise ValueError("Expected a non-empty list of PIL.Image.Images.")
        if not all(isinstance(image, PIL.Image.Image) for image in images):
            first_non_image = next(
                type(image)
                for image in images
                if not isinstance(image, PIL.Image.Image)
            )
            raise ValueError(
                "Expected a list of PIL.Image.Images, "
                f"but found an element of type {first_non_image}"
            )

    @abstractmethod
    def compute_features(
        self,
        images: list[PIL.Image.Image],
        seed: int | None = None,
    ) -> np.ndarray:
        """Compute features and return them as a NumPy array.

        Args:
            images: A B-length list of PIL.Image.Image.
            seed: Optional RNG seed for deterministic results.

        Returns:
            A NumPy array of shape [B, *].

        Raises:
            ValueError: If the input images are invalid.
        """
        raise NotImplementedError

    def _validate_device_configuration(self) -> None:
        """Validate the backend-neutral device configuration."""
        if self._device not in {"cpu", "gpu"}:
            raise ValueError(
                f"Unknown device: {self._device}. Expected 'cpu' or 'gpu'."
            )

        if self._device_index is not None:
            if not isinstance(self._device_index, int):
                raise ValueError(
                    "device_index must be an int or None, "
                    f"but got {type(self._device_index)}."
                )
            if self._device_index < 0:
                raise ValueError(
                    f"device_index must be non-negative, but got {self._device_index}."
                )

        if self._device == "cpu" and self._device_index is not None:
            raise ValueError("device_index can only be set when device='gpu'.")


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
                torch_features = self.forward(images=images, seed=seed)
                return torch_features.detach().cpu().numpy()

    def forward(
        self,
        images: list[PIL.Image.Image],
        seed: int | None = None,
    ) -> torch.Tensor:
        """Compute torch features for a batch of images."""
        self.validate_images(images)
        return self._images_to_features(images=images, seed=seed)

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


class TensorflowImageEncoding(ImageEncoding, ABC):
    """TensorFlow-backed image encoder with shared device resolution."""

    def __init__(
        self,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize backend-neutral state and resolve the TensorFlow device."""
        super().__init__(device=device, device_index=device_index)
        import tensorflow as tf

        self._tensorflow_device_name = self._resolve_tensorflow_device_name(tf)

    @property
    def tensorflow_device_name(self) -> str:
        """Return the resolved TensorFlow device name for this encoder."""
        return self._tensorflow_device_name

    def _resolve_tensorflow_device_name(self, tf: Any) -> str:
        """Resolve the backend-neutral device to a concrete TensorFlow device."""
        if self.device == "cpu":
            logical_cpus = tf.config.list_logical_devices("CPU")
            if logical_cpus:
                return logical_cpus[0].name
            return "/CPU:0"

        logical_gpus = tf.config.list_logical_devices("GPU")
        if not logical_gpus:
            raise ValueError(
                "device='gpu' was requested, but TensorFlow could not find an available GPU."
            )

        gpu_index = 0 if self.device_index is None else self.device_index
        if gpu_index >= len(logical_gpus):
            raise ValueError(
                f"Requested TensorFlow GPU index {gpu_index}, but only {len(logical_gpus)} logical GPU device(s) are available."
            )
        return logical_gpus[gpu_index].name

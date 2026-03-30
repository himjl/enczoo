from abc import ABC, abstractmethod
from typing import Literal

import PIL.Image
import numpy as np

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
            test_result = self.compute_features(images=[test_image], flatten=False)
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
        flatten: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        """Compute features and return them as a NumPy array.

        Args:
            images: A B-length list of PIL.Image.Image.
            flatten: If True, flatten the output to [B, d].
            seed: Optional RNG seed for deterministic results.

        Returns:
            A NumPy array of shape [B, *], or [B, d] if flatten=True.

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

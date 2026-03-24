from abc import ABC, abstractmethod

import PIL.Image
import numpy as np
import torch


class ImageEncoding(ABC):
    """Framework-agnostic interface for mapping PIL images to NumPy features."""

    def __init__(self):
        """Initialize cached metadata."""
        self._output_shape = None

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


class TorchImageEncoding(torch.nn.Module, ImageEncoding, ABC):
    """Torch-backed image encoder that implements NumPy conversion."""

    def __init__(self):
        """Initialize the torch module and shared metadata."""
        torch.nn.Module.__init__(self)
        ImageEncoding.__init__(self)

    @property
    def device(self) -> torch.device:
        """Infer the device from the first parameter or buffer."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    def compute_features(
        self,
        images: list[PIL.Image.Image],
        flatten: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        """Compute features and return them as a NumPy array."""
        self.validate_images(images)

        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                torch_features = self(images=images, flatten=flatten)
                numpy_features = torch_features.detach().cpu().numpy()
        return numpy_features

    def forward(
        self,
        images: list[PIL.Image.Image],
        flatten: bool = False,
    ) -> torch.Tensor:
        """Compute torch features for a batch of images."""
        self.validate_images(images)
        feats = self._images_to_features(images=images)
        if flatten:
            feats = feats.reshape(feats.shape[0], -1)
        return feats

    @abstractmethod
    def _images_to_features(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        """Convert images to torch features."""
        raise NotImplementedError

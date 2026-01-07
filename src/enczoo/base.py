from abc import ABC, abstractmethod
from typing import List, Tuple

import PIL.Image
import numpy as np
import torch

import enczoo.utils as utils


# %%
class ImageEncoding(
    torch.nn.Module,
    ABC,
):
    """Map PIL images to batched float tensors.

    This module maps B-length lists of PIL.Image.Image to float tensors shaped
    [B, *]. Parameters do not aggregate gradients by default.
    """

    def __init__(
        self,
        trainable: bool = False,
    ):
        """Initialize the encoder.

        Args:
            trainable: If True, the module is put in train mode.
        """
        super().__init__()
        self.trainable = trainable
        self._module_hash = None
        self._tensor_bucket = None
        self._output_shape = None

        # Set the module's mode
        self.train(mode=trainable)

    @property
    def device(self) -> torch.device:
        """Infer the device from the first parameter or buffer.

        Returns:
            The device of the first parameter or buffer, or CPU if none exist.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Return the output feature shape (excluding batch dimension)."""
        if self._output_shape is None:
            test_image = PIL.Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            test_result = self.compute_features(images=[test_image], flatten=False)
            if not isinstance(test_result, np.ndarray):
                raise ValueError(
                    f"Expected a np.ndarray from self.forward, but got {type(test_result)}"
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

    @property
    def module_hash(self) -> str:
        """Return a stable hash for the module's parameters and structure."""
        if self.trainable:
            raise ValueError("Cannot hash a trainable model.")

        # Turn off gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Hash self if unhashed
        if self._module_hash is None:
            self._module_hash = utils.hash_torch_module(module=self)
        return self._module_hash

    def compute_features(
        self,
        images: List[PIL.Image.Image],
        flatten: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        """Compute features and return them as a NumPy array.

        This is an alias for __call__ to help IDEs that do not recognize the
        torch.nn.Module __call__ signature.

        Args:
            images: A B-length list of PIL.Image.Image.
            flatten: If True, flatten the output to [B, d].
            seed: Optional RNG seed for deterministic results.

        Returns:
            A NumPy array of shape [B, *], or [B, d] if flatten=True.

        Raises:
            ValueError: If the input images are invalid.
        """

        with torch.random.fork_rng():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                torch_features: torch.Tensor = self(images=images, flatten=flatten)
                numpy_features = torch_features.detach().cpu().numpy()
        return numpy_features

    def forward(
        self,
        images: List[PIL.Image.Image],
        flatten: bool = False,
    ) -> torch.Tensor:
        """Compute features for a batch of images.

        Args:
            images: A B-length list of PIL.Image.Image.
            flatten: If True, flatten the output to [B, d].

        Returns:
            A torch.Tensor of shape [B, *].

        Raises:
            ValueError: If the images list is empty or not image objects.
        """
        if not isinstance(images, list):
            raise ValueError(
                f"Expected a list of PIL.Image.Images, but got {type(images)}"
            )
        if len(images) == 0:
            raise ValueError("Expected a non-empty list of PIL.Image.Images.")
        if not isinstance(images[0], PIL.Image.Image):
            raise ValueError(
                f"Expected a list of PIL.Image.Images, but element 0 is a {type(images[0])}"
            )

        # Call the subclass implementation
        feats = self._images_to_features(images=images)
        if flatten:
            # Flatten the features
            feats = feats.reshape(feats.shape[0], -1)

        return feats

    @abstractmethod
    def _images_to_features(self, images: List[PIL.Image.Image]) -> torch.Tensor:
        """Convert images to features.

        Args:
            images: A list of PIL.Image.Image.

        Returns:
            A torch.Tensor of shape [B, *].
        """
        raise NotImplementedError

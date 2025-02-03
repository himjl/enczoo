from abc import ABC, abstractmethod
from typing import List

import PIL.Image
import torch

# %%
class ImageEncoding(
    torch.nn.Module,
    ABC
):
    """
    torch.nn.Module which represents a fixed mapping from B-length lists of PIL.Images to [B, *] float tensors.
    """

    def compute_features(self, images: List[PIL.Image]) -> torch.Tensor:
        """
        Just an alias for __call__, to allow for type hinting by IDEs
        (PyCharm does not recognize __call__ signature of torch.nn.Modules, for some reason).
        :param images: a B length list of PIL.Images.
        :return: a torch.Tensor of shape [B, *]
        """
        return self(images=images)

    def forward(self, images: List[PIL.Image]) -> torch.Tensor:
        """
        :param images: a B length list of PIL.Images.
        :return: a torch.Tensor of shape [B, *]
        """
        if not isinstance(images, list):
            raise ValueError(f'Expected a list of PIL.Images, but got {type(images)}')
        if not isinstance(images[0], PIL.Image.Image):
            raise ValueError(f'Expected a list of PIL.Images, but element 0 is a {type(images[0])}')
        if len(images) == 0:
            raise ValueError('Expected a non-empty list of PIL.Images.')

        # Call the subclass implementation
        return self._images_to_features(images=images)

    @abstractmethod
    def _images_to_features(self, images: List[PIL.Image]) -> torch.Tensor:
        """
        :param images: a B-length list of PIL.Images
        :return: a torch.Tensor of shape [B, *]
        """
        raise NotImplementedError("Subclasses must implement this method.")

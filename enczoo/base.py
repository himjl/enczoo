from abc import ABC, abstractmethod
from typing import List, Tuple

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

    def compute_features(
            self,
            images: List[PIL.Image],
            flatten: bool = True
    ) -> torch.Tensor:
        """
        Just an alias for __call__, to allow for type hinting by IDEs
        (PyCharm does not recognize __call__ signature of torch.nn.Modules, for some reason).
        :param images: a B length list of PIL.Images.
        :param flatten: if True, flattens the output tensor to [B, d].
        :return: a torch.Tensor of shape [B, *]. If flatten=True, returns [B, d] instead.
        """
        return self(images=images, flatten=flatten)

    def forward(
            self,
            images: List[PIL.Image],
            flatten: bool = True
    ) -> torch.Tensor:
        """
        :param images: a B length list of PIL.Images.
        :param flatten: if True, flattens the output tensor to [B, d].
        :return: a torch.Tensor of shape [B, *]
        """
        if not isinstance(images, list):
            raise ValueError(f'Expected a list of PIL.Images, but got {type(images)}')
        if not isinstance(images[0], PIL.Image.Image):
            raise ValueError(f'Expected a list of PIL.Images, but element 0 is a {type(images[0])}')
        if len(images) == 0:
            raise ValueError('Expected a non-empty list of PIL.Images.')

        # Call the subclass implementation
        feat = self._images_to_features(images=images)

        if flatten:
            # Flatten the features
            feat = feat.reshape(feat.shape[0], -1)

        return feat

    @abstractmethod
    def _images_to_features(self, images: List[PIL.Image]) -> torch.Tensor:
        """
        :param images: a B-length list of PIL.Images
        :return: a torch.Tensor of shape [B, *]
        """
        raise NotImplementedError("Subclasses must implement this method.")

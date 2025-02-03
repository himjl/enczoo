from typing import List

import PIL.Image
import torch
import torchvision.transforms.v2 as v2

from enczoo.base import ImageEncoding


# %%
class Pixels(ImageEncoding):
    """
    This ImageEncoding simply takes the center crop of an image and resizes it to size x size.
    """

    def __init__(self, size: int = 16):
        super().__init__()

        self.size = torch.tensor(size, dtype=torch.int16, requires_grad=False)
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.RGB(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=None, max_size=size, antialias=False),
                v2.CenterCrop(size=size),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def _images_to_features(self, images: List[PIL.Image]) -> torch.Tensor:
        # Apply the transformations to each image
        transformed_images = [self.transforms(image) for image in images]

        # Stack the transformed images into a single tensor
        images_tensor = torch.stack(transformed_images)

        # Rearrange from BCHW to BHWC order
        images_tensor = images_tensor.permute(0, 2, 3, 1)

        return images_tensor


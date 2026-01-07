import PIL.Image
import torch
import torchvision.transforms.v2 as v2
from typing import List

from enczoo.base import ImageEncoding, ImageEncodingConfig
from enczoo.transforms.random_projection import RandomProjection


# %%
class Pixels(ImageEncoding):
    """
    This ImageEncoding simply takes the center crop of an image and resizes it to (size x size x 3).
    """

    def __init__(
        self,
        size: int = 16,
        random_projection_dim: int | None = None,
        random_projection_seed: int | None = None,
        config: ImageEncodingConfig | None = None,
    ):
        super().__init__(config=config)

        # Register size tensor as buffer
        self.register_buffer(
            "size", torch.tensor(size, dtype=torch.int16, requires_grad=False)
        )

        # Transform
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

        # Random projection
        if random_projection_dim is not None:
            if random_projection_dim > (size * size * 3):
                raise ValueError(
                    f"random_projection_dim must be less than or equal to size * size * 3={size * size * 3}!"
                )

            if random_projection_seed is None:
                raise ValueError(
                    "random_projection_seed must be provided if random_projection_dim is not None!"
                )
            self.random_projection = RandomProjection(
                seed=random_projection_seed,
                in_features=int(size * size * 3),
                out_features=random_projection_dim,
            )
        else:
            self.random_projection = None

    def _images_to_features(self, images: List[PIL.Image.Image]) -> torch.Tensor:
        # Apply the transformations to each image
        transformed_images = [self.transforms(image.convert("RGB")) for image in images]

        # Stack the transformed images into a single tensor
        images_tensor = torch.stack(transformed_images)

        # If random projection is enabled, apply it
        if self.random_projection is not None:
            return self.random_projection(
                images_tensor.reshape(images_tensor.shape[0], -1)
            )
        else:
            # Rearrange from BCHW to BHWC order
            images_tensor = images_tensor.permute(0, 2, 3, 1)
            return images_tensor

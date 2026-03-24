import PIL.Image
import torch
import torchvision.transforms.v2 as v2

from enczoo.base import TorchImageEncoding


# %%
class Pixels(TorchImageEncoding):
    """Encode images by their resized center-crop pixels."""

    def __init__(
        self,
        size: int = 16,
    ):
        """Initialize the pixel encoder.

        Args:
            size: Output side length in pixels.
        """
        super().__init__()

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

    def _images_to_features(self, images: list[PIL.Image.Image]) -> torch.Tensor:
        """Convert images to pixel features.

        Args:
            images: A list of PIL.Image.Image.

        Returns:
            A torch.Tensor of shape [B, size, size, 3] or projected features.
        """
        # Apply the transformations to each image
        transformed_images = [self.transforms(image.convert("RGB")) for image in images]

        # Stack the transformed images into a single tensor
        images_tensor = torch.stack(transformed_images)

        # Rearrange from BCHW to BHWC order
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        return images_tensor

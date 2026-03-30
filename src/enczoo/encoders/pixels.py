import PIL.Image
import numpy as np

from enczoo.base import DeviceType, ImageEncoding

_MODEL_INPUT_SIZE = 224


class Pixels(ImageEncoding):
    """Encode images as float32 HWC pixels after 224x224 center-crop preprocessing."""

    def __init__(
        self,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize the pixel encoder."""
        super().__init__(device=device, device_index=device_index)
        if self.device != "cpu":
            raise ValueError("Pixels only supports device='cpu'.")

    def compute_features(
        self,
        images: list[PIL.Image.Image],
        seed: int | None = None,
    ) -> np.ndarray:
        """Return float32 pixels with shape [B, 224, 224, 3]."""
        del seed
        self.validate_images(images)
        return np.stack(
            [self._preprocess_image(image) for image in images],
            axis=0,
        )

    @staticmethod
    def _preprocess_image(image: PIL.Image.Image) -> np.ndarray:
        """Convert an image to float32 HWC pixels with 224x224 center-crop preprocessing."""
        image = image.convert("RGB")
        width, height = image.size
        scale = _MODEL_INPUT_SIZE / min(width, height)
        resized = image.resize(
            size=(round(width * scale), round(height * scale)),
            resample=PIL.Image.Resampling.BILINEAR,
        )

        left = (resized.width - _MODEL_INPUT_SIZE) // 2
        top = (resized.height - _MODEL_INPUT_SIZE) // 2
        cropped = resized.crop(
            (
                left,
                top,
                left + _MODEL_INPUT_SIZE,
                top + _MODEL_INPUT_SIZE,
            )
        )
        return np.asarray(cropped, dtype=np.float32) / 255.0

from typing import List

import PIL.Image
import torch
import transformers

from enczoo.base import ImageEncoding


class CLIPViTB16(ImageEncoding):
    """CLIP ViT-B/16 encoder returning the pooled image feature."""

    model_id = "openai/clip-vit-base-patch16"
    output_dim = 768

    def __init__(self):
        """Initialize the CLIP ViT-B/16 image encoder."""
        super().__init__(trainable=False)
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(
            self.model_id
        )
        self.model = transformers.CLIPVisionModel.from_pretrained(self.model_id)
        self.model.train(mode=False)

    def _images_to_features(self, images: List[PIL.Image.Image]) -> torch.Tensor:
        """Convert images to pooled CLIP vision features."""
        inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.model(pixel_values=pixel_values)
        pooled_features = outputs.pooler_output

        if pooled_features.ndim != 2 or pooled_features.shape[1] != self.output_dim:
            raise ValueError(
                "Expected CLIP ViT-B/16 pooled features with shape [B, 768], "
                f"but got {tuple(pooled_features.shape)}"
            )

        return pooled_features

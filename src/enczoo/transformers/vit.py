from abc import ABC, abstractmethod

import PIL.Image
import torch
import transformers

from enczoo.base import DeviceType
from enczoo.torch_base import TorchImageEncoding


class _HuggingFaceViT(TorchImageEncoding, ABC):
    """Base class for Hugging Face vision-transformer encoders."""

    model_id: str
    output_dim: int
    use_fast_processor: bool = False
    suppress_transformers_load_logging: bool = False

    def __init__(
        self,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize the image processor and model."""
        super().__init__(device=device, device_index=device_index)

        if self.suppress_transformers_load_logging:
            # CLIP checkpoints on HF include text-side weights we intentionally ignore.
            transformers.logging.set_verbosity_error()

        self.image_processor = transformers.AutoImageProcessor.from_pretrained(
            self.model_id,
            use_fast=self.use_fast_processor,
        )
        model = self._load_model()
        model.train(mode=False)
        self.model = model.to(self.torch_device)

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """Load the underlying Hugging Face model."""
        raise NotImplementedError

    @abstractmethod
    def _select_features(self, outputs) -> torch.Tensor:
        """Select the desired feature tensor from the model outputs."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _feature_name(self) -> str:
        """Return a human-readable feature name for error messages."""
        raise NotImplementedError

    def _images_to_features(
        self,
        images: list[PIL.Image.Image],
        seed: int | None = None,
    ) -> torch.Tensor:
        """Convert images to pooled transformer features."""
        del seed
        inputs = self.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.torch_device)

        outputs = self.model(pixel_values=pixel_values)
        features = self._select_features(outputs=outputs)

        if features.ndim != 2 or features.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected {self._feature_name} with shape [B, {self.output_dim}], "
                f"but got {tuple(features.shape)}"
            )

        return features


class CLIPViTB16(_HuggingFaceViT):
    """CLIP ViT-B/16 encoder returning the pooled image feature."""

    model_id = "openai/clip-vit-base-patch16"
    output_dim = 768
    suppress_transformers_load_logging = True

    @property
    def _feature_name(self) -> str:
        return "CLIP ViT-B/16 pooled features"

    def _load_model(self) -> torch.nn.Module:
        return transformers.CLIPVisionModel.from_pretrained(self.model_id)

    def _select_features(self, outputs) -> torch.Tensor:
        return outputs.pooler_output


class DINOv2ViTB14(_HuggingFaceViT):
    """DINOv2 ViT-B/14 encoder returning the pooled image feature."""

    model_id = "facebook/dinov2-base"
    output_dim = 768

    @property
    def _feature_name(self) -> str:
        return "DINOv2 ViT-B/14 pooled features"

    def _load_model(self) -> torch.nn.Module:
        return transformers.AutoModel.from_pretrained(self.model_id)

    def _select_features(self, outputs) -> torch.Tensor:
        return outputs.pooler_output

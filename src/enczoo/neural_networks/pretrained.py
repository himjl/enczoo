from abc import ABC, abstractmethod
from typing import List, Tuple

import PIL.Image
import torch
import torchvision.models
import torchvision.transforms.functional as F

from enczoo.neural_networks.base import ImageNeuralNetwork


class StandardImageLoader(torch.nn.Module):
    """Load and normalize images for standard torchvision models.

    This loader resizes directly to 224 before center-crop, preserving the
    largest square sub-image. It also converts inputs to RGB.

    Example:
        - Original loader: resize to 256, then center-crop 224.
        - This loader: resize to 224, then center-crop 224.
    """

    def forward(self, img: PIL.Image.Image) -> torch.Tensor:
        """Convert a PIL image into a normalized tensor.

        Args:
            img: Input image.

        Returns:
            Normalized image tensor suitable for torchvision models.
        """
        img = img.convert("RGB")

        img_tensor = F.pil_to_tensor(pic=img)
        img_tensor = F.resize(
            img=img_tensor, size=[224], interpolation=F.InterpolationMode.BILINEAR
        )
        img_tensor = F.center_crop(img=img_tensor, output_size=[224])
        img_tensor = F.convert_image_dtype(image=img_tensor, dtype=torch.float)
        img_tensor = F.normalize(
            tensor=img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return img_tensor


class _PretrainedNN(ImageNeuralNetwork, ABC):
    """Base class for pretrained torchvision encoders."""

    layer_names: List[str]

    def __init__(
        self,
        layer_name: str,
        random_projection_dim: int | None = None,
        random_projection_seed: int | None = None,
    ):
        """Initialize a pretrained encoder.

        Args:
            layer_name: Name of the layer whose activations are returned.
            random_projection_dim: Optional output dimension for projection.
            random_projection_seed: Seed for projection weights.

        Raises:
            ValueError: If the layer name is invalid.
        """
        if layer_name not in self.layer_names:
            raise ValueError(
                f"Unknown layer_name: {layer_name}. Available:\n{self.layer_names}"
            )

        image_loader, model = self._load_modules()

        # Ensure modules are in evaluation mode by default
        image_loader.train(mode=False)
        model.train(mode=False)

        super().__init__(
            image_loader=image_loader,
            model=model,
            layer_name=layer_name,
            random_projection_dim=random_projection_dim,
            random_projection_seed=random_projection_seed,
        )

    @abstractmethod
    def _load_modules(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Load the image loader and model for this network.

        Returns:
            A tuple of (image_loader, model).
        """
        raise NotImplementedError


class AlexNet(_PretrainedNN):
    """AlexNet encoder with named layer outputs."""

    # A subset of all layers (each separated by one nonlinearity):
    layer_names = [
        "features.1",
        "features.4",
        "features.7",
        "features.9",
        "features.11",
        "classifier.2",
        "classifier.5",
        "classifier.6",
    ]

    def _load_modules(self):
        """Load the AlexNet image loader and model."""
        image_loader = StandardImageLoader()
        model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        )
        return image_loader, model


class ResNet50(_PretrainedNN):
    """ResNet-50 encoder with named layer outputs."""

    # A subset of layers (each separated by one nonlinearity, except layer4.2.relu, avgpool, and fc, which are connected by a linear layer):
    layer_names = [
        "relu",
        "layer1.0.relu",
        "layer1.1.relu",
        "layer1.2.relu",
        "layer2.0.relu",
        "layer2.1.relu",
        "layer2.2.relu",
        "layer2.3.relu",
        "layer3.0.relu",
        "layer3.1.relu",
        "layer3.2.relu",
        "layer3.3.relu",
        "layer3.4.relu",
        "layer3.5.relu",
        "layer4.0.relu",
        "layer4.1.relu",
        "layer4.2.relu",
        "avgpool",
        "fc",
    ]

    def _load_modules(self):
        """Load the ResNet-50 image loader and model."""
        image_loader = StandardImageLoader()
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        return image_loader, model


if __name__ == "__main__":
    resnet50 = ResNet50(
        layer_name="avgpool", random_projection_dim=None, random_projection_seed=0
    )
    print(resnet50.training)
    print(resnet50.model.training)
    print(getattr(resnet50.image_loader, "training", None))

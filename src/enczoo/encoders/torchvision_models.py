from abc import ABC, abstractmethod

import PIL.Image
import torch
import torchvision.models
import torchvision.transforms.functional as F

from enczoo.base import DeviceType
from enczoo.encoders.base import ImageNeuralNetwork


class StandardImageLoader(torch.nn.Module):
    """Load and normalize images for standard torchvision models."""

    def forward(self, img: PIL.Image.Image) -> torch.Tensor:
        """Convert a PIL image into a normalized tensor."""
        img = img.convert("RGB")

        img_tensor = F.pil_to_tensor(pic=img)
        img_tensor = F.resize(
            img=img_tensor,
            size=[224],
            interpolation=F.InterpolationMode.BILINEAR,
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

    layer_names: list[str]

    def __init__(
        self,
        layer_name: str,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize a pretrained encoder."""
        if layer_name not in self.layer_names:
            raise ValueError(
                f"Unknown layer_name: {layer_name}. Available:\n{self.layer_names}"
            )

        image_loader, model = self._load_modules()

        if isinstance(image_loader, torch.nn.Module):
            image_loader.train(mode=False)
        model.train(mode=False)

        super().__init__(
            image_loader=image_loader,
            model=model,
            layer_name=layer_name,
            device=device,
            device_index=device_index,
        )

    @abstractmethod
    def _load_modules(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Load the image loader and model for this network."""
        raise NotImplementedError


class AlexNet(_PretrainedNN):
    """AlexNet encoder with named layer outputs."""

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

    def __init__(
        self,
        layer_name: str = "classifier.5",
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize an AlexNet encoder."""
        super().__init__(
            layer_name=layer_name,
            device=device,
            device_index=device_index,
        )

    def _load_modules(self):
        """Load the AlexNet image loader and model."""
        image_loader = StandardImageLoader()
        model = torchvision.models.alexnet(
            weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        )
        return image_loader, model


class ResNet50(_PretrainedNN):
    """ResNet-50 encoder with named layer outputs."""

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

    def __init__(
        self,
        layer_name: str = "avgpool",
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize a ResNet-50 encoder."""
        super().__init__(
            layer_name=layer_name,
            device=device,
            device_index=device_index,
        )

    def _load_modules(self):
        """Load the ResNet-50 image loader and model."""
        image_loader = StandardImageLoader()
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        return image_loader, model


class ConvNeXtB(_PretrainedNN):
    """ConvNeXt-B encoder with named layer outputs."""

    layer_names = [
        "features.0",
        "features.1",
        "features.2",
        "features.3",
        "features.4",
        "features.5",
        "features.6",
        "features.7",
        "avgpool",
        "classifier",
    ]

    def __init__(
        self,
        layer_name: str = "avgpool",
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize a ConvNeXt-B encoder."""
        super().__init__(
            layer_name=layer_name,
            device=device,
            device_index=device_index,
        )

    def _load_modules(self):
        """Load the ConvNeXt-B image loader and model."""
        image_loader = StandardImageLoader()
        model = torchvision.models.convnext_base(
            weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        )
        return image_loader, model

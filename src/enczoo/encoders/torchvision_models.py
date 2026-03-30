from abc import ABC, abstractmethod

import PIL.Image
import numpy as np
import torch
import torchvision.models
import torchvision.transforms.functional as F

from enczoo.base import DeviceType, TorchImageEncoding


class _ImageNeuralNetwork(TorchImageEncoding, ABC):
    """Image encoding backed by a torch neural network."""

    def __init__(
        self,
        image_loader: torch.nn.Module | torchvision.transforms.Compose,
        model: torch.nn.Module,
        layer_name: str,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize the neural network encoder.

        Args:
            image_loader: Module that converts PIL images to model inputs.
            model: Torch model used to compute activations.
            layer_name: Name of the layer whose activations are returned.
            device: Whether computations should run on the CPU or a GPU.
            device_index: Optional zero-based GPU index used when device="gpu".

        Raises:
            ValueError: If the layer name is not found.
        """
        super().__init__(device=device, device_index=device_index)

        if isinstance(image_loader, torch.nn.Module):
            image_loader.train(mode=False)
        model.train(mode=False)

        self._layer_name = layer_name

        def register_hook(
            module: torch.nn.Module,
            root_name: str,
            activations_dict: dict[str, torch.Tensor],
        ) -> list[str]:
            """Recursively register forward hooks on named modules."""
            module_names = []

            if root_name != "":
                discovered_layer_name = root_name

                if discovered_layer_name in activations_dict:
                    raise Exception(
                        f"Layer name {discovered_layer_name} already exists in hidden activations! Existing keys: {self._hidden_activations.keys()}"
                    )

                def hook_function(module: torch.nn.Module, args, output):
                    del module, args
                    activations_dict[discovered_layer_name] = output

                module.register_forward_hook(hook_function)
                module_names.append(discovered_layer_name)

            for module_name, submodule in module.named_children():
                if module_name == "":
                    raise ValueError("Empty module name found in model!")
                next_root_name = (
                    root_name + "." + module_name if root_name != "" else module_name
                )
                submodule.train(mode=False)
                module_names.extend(
                    register_hook(
                        submodule,
                        root_name=next_root_name,
                        activations_dict=activations_dict,
                    )
                )
            return module_names

        self._hidden_activations: dict[str, torch.Tensor] = {}
        self._layer_names = register_hook(
            model,
            root_name="",
            activations_dict=self._hidden_activations,
        )

        if layer_name not in self._layer_names:
            raise ValueError(
                f"Layer name {layer_name} not found in model.\nAvailable layer names: {'\n'.join(self._layer_names)}"
            )

        with torch.no_grad():
            test_image = PIL.Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            test_image = image_loader(test_image)
            model(test_image.unsqueeze(0))

        self._layer_to_shape = {
            layer: tuple(self._hidden_activations[layer].shape[1:])
            for layer in self._hidden_activations
        }

        self.image_loader = image_loader
        self.model = model.to(self.torch_device)

    def _images_to_features(
        self,
        images: list[PIL.Image.Image],
    ) -> torch.Tensor:
        """Convert images to network activations."""
        preprocessed_images = torch.stack(
            [self.image_loader(image) for image in images],
            dim=0,
        )
        preprocessed_images = preprocessed_images.to(self.torch_device)
        self.model(preprocessed_images)
        return self._hidden_activations[self._layer_name]

    @property
    def layer_name_to_shape(self) -> dict[str, tuple[int, ...]]:
        """Return a mapping of layer names to activation shapes."""
        return self._layer_to_shape


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


class _PretrainedNN(_ImageNeuralNetwork, ABC):
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

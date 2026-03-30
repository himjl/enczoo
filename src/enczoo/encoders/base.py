from abc import ABC

import PIL.Image
import numpy as np
import torch
import torchvision

from enczoo.base import DeviceType, TorchImageEncoding


class ImageNeuralNetwork(TorchImageEncoding, ABC):
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

from abc import ABC

import PIL.Image
import numpy as np
import torch
import torchvision

from enczoo.base import DeviceType
from enczoo.torch_base import TorchImageEncoding


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

        # Ensure modules will be registered in evaluation mode
        if isinstance(image_loader, torch.nn.Module):
            image_loader.train(mode=False)
        model.train(mode=False)

        self._layer_name = layer_name  # Needed for the forward pass

        def register_hook(
            module: torch.nn.Module,
            root_name: str,
            activations_dict: dict[str, torch.Tensor],
        ) -> list[str]:
            """Recursively register forward hooks on named modules.

            Args:
                module: Module whose children are walked.
                root_name: Prefix for module names.
                activations_dict: Dict populated with layer activations.

            Returns:
                A list of layer names in discovery order.
            """
            module_names = []

            if root_name != "":
                layer_name = root_name

                if layer_name in activations_dict:
                    raise Exception(
                        f"Layer name {layer_name} already exists in hidden activations! Existing keys: {self._hidden_activations.keys()}"
                    )

                def hook_function(module: torch.nn.Module, args, output):
                    activations_dict[layer_name] = output

                module.register_forward_hook(hook_function)
                module_names.append(layer_name)

            for module_name, submodule in module.named_children():
                if module_name != "":
                    next_root_name = (
                        root_name + "." + module_name
                        if root_name != ""
                        else module_name
                    )
                else:
                    raise ValueError("Empty module name found in model!")
                # Ensure module is in evaluation mode
                submodule.train(mode=False)
                # Recursive call:
                submodule_names = register_hook(
                    submodule,
                    root_name=next_root_name,
                    activations_dict=activations_dict,
                )

                module_names.extend(submodule_names)
            return module_names

        # Register forward hooks that will populate this dictionary with hidden activations on the forward pass:
        self._hidden_activations: dict[str, torch.Tensor] = {}
        self._layer_names = register_hook(
            model, root_name="", activations_dict=self._hidden_activations
        )

        if layer_name not in self._layer_names:
            raise ValueError(
                f"Layer name {layer_name} not found in model.\nAvailable layer names: {'\n'.join(self._layer_names)}"
            )

        # Populate the sizes of the layers with a forward pass
        with torch.no_grad():
            test_image = PIL.Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            test_image = image_loader(test_image)
            model(test_image.unsqueeze(0))

        self._layer_to_shape = {
            layer: tuple(self._hidden_activations[layer].shape[1:])
            for layer in self._hidden_activations
        }

        # Register modules
        self.image_loader = image_loader
        self.model = model.to(self.torch_device)

    def _images_to_features(
        self,
        images: list[PIL.Image.Image],
        seed: int | None = None,
    ) -> torch.Tensor:
        """Convert images to network activations.

        Args:
            images: A list of PIL.Image.Image.
            seed: Unused backend seed forwarded for API consistency.

        Returns:
            A torch.Tensor of shape [B, *].
        """
        del seed

        # Preprocess the images
        preprocessed_images = torch.stack(
            [self.image_loader(image) for image in images], dim=0
        )

        # Transfer to the correct device
        preprocessed_images = preprocessed_images.to(self.torch_device)

        # Run the forward pass
        self.model(preprocessed_images)

        # Retrieve the activations for the given layer
        f = self._hidden_activations[self._layer_name]

        return f

    @property
    def layer_name_to_shape(self) -> dict[str, tuple[int, ...]]:
        """Return a mapping of layer names to activation shapes."""
        return self._layer_to_shape

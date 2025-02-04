from abc import ABC
from enczoo.base import ImageEncoding
import torch
from typing import Literal, Union, Dict, List, Tuple
import numpy as np
import PIL.Image
import torchvision
from enczoo.transforms.random_projection import RandomProjection


class ImageNeuralNetwork(
    ImageEncoding,
    ABC,
):
    def __init__(
            self,
            image_loader: Union[torch.nn.Module, torchvision.transforms.Compose],
            model: torch.nn.Module,
            layer_name: str,
            random_projection_dim: Union[int, None],
            random_projection_seed: Union[int, None],
    ):
        super().__init__()

        # Register buffers to ensure the model's hash is distinctive for each layer
        self.register_buffer('layer_name', torch.tensor([ord(c) for c in layer_name], dtype=torch.int16))
        self._layer_name = layer_name  # Needed for the forward pass

        def register_hook(
                module: torch.nn.Module,
                root_name: str,
                activations_dict: Dict[str, torch.Tensor]
        ) -> List[str]:
            """
            Recursively registers the forward hook on all children of a module.
            :param module:
            :param root_name:
            :return: a list of layer names, in the order they were discovered.
            """
            module_names = []

            nchildren = 0
            for module_name, submodule in module.named_children():
                if module_name != '':
                    next_root_name = root_name + '.' + module_name if root_name != '' else module_name
                else:
                    raise ValueError('Empty module name found in model!')

                # Recursive call:
                submodule_names = register_hook(submodule, root_name=next_root_name, activations_dict=activations_dict)

                # Update the number of children
                nchildren += 1
                module_names.extend(submodule_names)

            # Base case:
            if nchildren == 0:
                layer_name = root_name

                if layer_name in activations_dict:
                    # Don't think this should ever happen:
                    raise Exception(f'Layer name {layer_name} already exists in hidden activations! Existing keys: {self._hidden_activations.keys()}')

                def hook_function(module: torch.nn.Module, args, output):
                    # print(f'Hook called on {layer_name}')
                    activations_dict[layer_name] = output

                # Attach forward hook
                module.register_forward_hook(hook_function)

                # Base case: no children
                module_names.append(layer_name)
            return module_names

        # Register forward hooks that will populate this dictionary with hidden activations on the forward pass:
        self._hidden_activations: Dict[str, torch.Tensor] = {}
        self._layer_names = register_hook(model, root_name='', activations_dict=self._hidden_activations)

        if layer_name not in self._layer_names:
            raise ValueError(f'Layer name {layer_name} not found in model.\nAvailable layer names: {self._layer_names}')

        # Turn off training mode by default
        self.train(mode=False)

        # Populate the sizes of the layers with a forward pass
        with torch.no_grad():
            test_image = PIL.Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            test_image = image_loader(test_image)
            model(test_image.unsqueeze(0))

        self._layer_to_shape = {
            layer: tuple(self._hidden_activations[layer].shape[1:]) for layer in self._hidden_activations
        }

        # Register modules
        self.image_loader = image_loader
        self.model = model

        if random_projection_dim is not None:
            if random_projection_seed is None:
                raise ValueError('random_projection_seed must be provided if random_projection_dim is not None!')
            self.random_projection = RandomProjection(
                seed=random_projection_seed,
                in_features=int(np.prod(self._layer_to_shape[layer_name])),
                out_features=random_projection_dim
            )
        else:
            self.random_projection = None

    def _images_to_features(self, images: List[PIL.Image]) -> torch.Tensor:
        """
        :param images: a list of PIL.Images
        :return: a torch.Tensor of shape [B, *]
        """

        # Preprocess the images
        preprocessed_images = torch.stack([self.image_loader(image) for image in images], dim=0)

        # Run the forward pass
        self.model(preprocessed_images)

        # Retrieve the activations for the given layer
        f = self._hidden_activations[self._layer_name]

        # Perform random projection if requested
        if self.random_projection is not None:
            # Flatten the features
            f = f.reshape(f.shape[0], -1)

            # Run the random projection forward
            f = self.random_projection(f)

        return f

    @property
    def layer_name_to_shape(self) -> Dict[str, Tuple[int, ...]]:
        """
        :return: a dictionary mapping layer names to the number of features in the layer.
        """
        return self._layer_to_shape

    @property
    def output_shape(self) -> Tuple[int, ...]:
        if self.random_projection is None:
            return self._layer_to_shape[self._layer_name]
        else:
            return self.random_projection.output_shape

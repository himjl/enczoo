import PIL.Image
import torch
import torchvision.models
import torchvision.transforms.functional as F
from abc import ABC, abstractmethod
from typing import Union, List, Tuple

from enczoo.neural_networks.base import ImageNeuralNetwork, ImageEncodingConfig


class StandardImageLoader(torch.nn.Module):
    """
    A variant of the image loader used for many torchvision models. However, this
    variant does not necessarily "chop" the image margins. That is, this
    image loader results in the largest square sub-image (unlike the standard image loader, which returns a proportionally smaller sub-image).

    It also coerces the image to RGB mode, if it is not already in that mode.

    See the ImageClassification torchvision.transforms._presents.py module for the original image loader.

    Example:
        - Original image loader: resize to 256, then center-crop the 224x224 subimage.
        - This image loader: resize to 224, then center-crop the 224x224 subimage.
    :return:
    """

    def forward(self, img: PIL.Image.Image) -> torch.Tensor:
        img = img.convert('RGB')

        img = F.resize(
            img=img,
            size=[224],
            interpolation=F.InterpolationMode.BILINEAR
        )
        img = F.center_crop(
            img=img,
            output_size=[224]
        )
        img = F.pil_to_tensor(pic=img)
        img = F.convert_image_dtype(image=img, dtype=torch.float)
        img = F.normalize(
            tensor=img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return img


class _PretrainedNN(ImageNeuralNetwork, ABC):
    layer_names: List[str]

    def __init__(
            self,
            layer_name: str,
            random_projection_dim: Union[int, None] = None,
            random_projection_seed: Union[int, None] = None,
            config: ImageEncodingConfig = None,
    ):
        if layer_name not in self.layer_names:
            raise ValueError(f'Unknown layer_name: {layer_name}. Available:\n{self.layer_names}')

        image_loader, model = self._load_modules()

        # Ensure modules are in evaluation mode
        image_loader.train(mode=False)
        model.train(mode=False)

        super().__init__(
            image_loader=image_loader,
            model=model,
            layer_name=layer_name,
            random_projection_dim=random_projection_dim,
            random_projection_seed=random_projection_seed,
            config=config,
        )

    @abstractmethod
    def _load_modules(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Load the image loader and model for this neural network.
        :return:
        """
        raise NotImplementedError


class AlexNet(_PretrainedNN):
    # A subset of all layers (each separated by one nonlinearity):
    layer_names = [
        'features.1',
        'features.4',
        'features.7',
        'features.9',
        'features.11',
        'classifier.2',
        'classifier.5',
        'classifier.6'
    ]

    def _load_modules(self):
        image_loader = StandardImageLoader()
        model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        return image_loader, model


class ResNet50(_PretrainedNN):
    # A subset of layers (each separated by one nonlinearity, except layer4.2.relu, avgpool, and fc, which are connected by a linear layer):
    layer_names = [
        'relu',
        'layer1.0.relu',
        'layer1.1.relu',
        'layer1.2.relu',
        'layer2.0.relu',
        'layer2.1.relu',
        'layer2.2.relu',
        'layer2.3.relu',
        'layer3.0.relu',
        'layer3.1.relu',
        'layer3.2.relu',
        'layer3.3.relu',
        'layer3.4.relu',
        'layer3.5.relu',
        'layer4.0.relu',
        'layer4.1.relu',
        'layer4.2.relu',
        'avgpool',
        'fc',
    ]

    def _load_modules(self):
        image_loader = StandardImageLoader()
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        return image_loader, model


if __name__ == '__main__':
    resnet50 = ResNet50(
        layer_name='avgpool',
        random_projection_dim=None,
        random_projection_seed=0
    )
    print(resnet50.training)
    print(resnet50.model.training)
    print(resnet50.image_loader.training)

import PIL.Image
import numpy as np
import torch

from enczoo.neural_networks.base import ImageNeuralNetwork


class _ImageToTensor(torch.nn.Module):
    def forward(self, img: PIL.Image.Image) -> torch.Tensor:
        array = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=1),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.head(x)


def test_supports_non_leaf_module_hooks():
    torch.manual_seed(0)
    encoder = ImageNeuralNetwork(
        image_loader=_ImageToTensor(),
        model=_ToyModel(),
        layer_name="head",
        random_projection_dim=None,
        random_projection_seed=None,
    )

    assert "head" in encoder.layer_name_to_shape
    assert "stem" in encoder.layer_name_to_shape
    assert encoder.layer_name_to_shape["head"] == (2,)

    image = PIL.Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    features = encoder.compute_features(images=[image], flatten=False)
    assert features.shape == (1, 2)

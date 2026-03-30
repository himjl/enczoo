import torch

from enczoo.base import DeviceType
from enczoo.encoders.torchvision_models import StandardImageLoader, _ImageNeuralNetwork

_WEIGHTS_URLS = {
    "ImageNetL2Epsilon3.0": "https://www.dropbox.com/s/knf4uimlqsi1yz8/imagenet_l2_3_0.pt?dl=1",
}


def _load_model(weights_name: str) -> torch.nn.Module:
    """Instantiate the vendored robust ResNet-50 and load pretrained weights."""
    from enczoo._vendor.robust_resnet50.resnet import resnet50

    model = resnet50()
    checkpoint = torch.hub.load_state_dict_from_url(
        _WEIGHTS_URLS[weights_name],
        progress=True,
        map_location="cpu",
    )
    raw_state_dict = checkpoint["model"]

    prefix = "module.model."
    state_dict: dict[str, object] = {}
    for key, value in raw_state_dict.items():
        if not key.startswith(prefix):
            continue
        key = key.removeprefix(prefix)

        if key in state_dict:
            raise ValueError(key)
        state_dict[key] = value

    model.load_state_dict(state_dict, strict=True)
    return model


class RobustResNet50(_ImageNeuralNetwork):
    """Robust ResNet-50, pretrained on ImageNet with L2 Epsilon 3.0 adversarial training."""

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
        """Initialize a robust ResNet-50 encoder.

        Args:
            layer_name: Name of the hooked layer to return.
            weights_name: Which published robust checkpoint to load.
            device: Whether computations should run on the CPU or a GPU.
            device_index: Optional zero-based GPU index used when device="gpu".
        """
        if layer_name not in self.layer_names:
            raise ValueError(
                f"Unknown layer_name: {layer_name}. Available:\n{self.layer_names}"
            )
        weights_name: str = "ImageNetL2Epsilon3.0"
        if weights_name not in _WEIGHTS_URLS:
            raise ValueError(f"Unknown weights_name: {weights_name}.")

        self.weights_name = weights_name
        super().__init__(
            image_loader=StandardImageLoader(),
            model=_load_model(weights_name=weights_name),
            layer_name=layer_name,
            device=device,
            device_index=device_index,
        )


if __name__ == "__main__":
    mod = RobustResNet50()

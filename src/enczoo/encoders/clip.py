from enczoo.base import DeviceType
from enczoo.encoders.torchvision import _PretrainedNN


class CLIPResNet50(_PretrainedNN):
    """CLIP RN50 visual encoder with named layer outputs."""

    layer_names = [
        "relu3",
        "layer1.0.relu3",
        "layer1.1.relu3",
        "layer1.2.relu3",
        "layer2.0.relu3",
        "layer2.1.relu3",
        "layer2.2.relu3",
        "layer2.3.relu3",
        "layer3.0.relu3",
        "layer3.1.relu3",
        "layer3.2.relu3",
        "layer3.3.relu3",
        "layer3.4.relu3",
        "layer3.5.relu3",
        "layer4.0.relu3",
        "layer4.1.relu3",
        "layer4.2.relu3",
        "attnpool",
    ]

    def __init__(
        self,
        layer_name: str = "attnpool",
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize a CLIP RN50 encoder."""
        super().__init__(
            layer_name=layer_name,
            device=device,
            device_index=device_index,
        )

    def _load_modules(self):
        """Load the CLIP RN50 image loader and visual model."""
        import clip

        model, image_loader = clip.load("RN50", device="cpu")
        return image_loader, model.visual

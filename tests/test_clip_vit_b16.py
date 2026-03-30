import PIL.Image
import numpy as np
import torch

from enczoo.transformers.vit import CLIPViTB16, DINOv2ViTB14


class _FakeImageProcessor:
    def __call__(self, images, return_tensors: str):
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.zeros((len(images), 3, 224, 224), dtype=torch.float32)
        }


class _FakeOutput:
    def __init__(self, batch_size: int, pooled_value: float):
        self.last_hidden_state = torch.zeros(
            (batch_size, 197, 768), dtype=torch.float32
        )
        self.pooler_output = torch.full(
            (batch_size, 768), pooled_value, dtype=torch.float32
        )


class _FakeVisionModel(torch.nn.Module):
    def __init__(self, pooled_value: float):
        super().__init__()
        self.pooled_value = pooled_value

    def forward(self, pixel_values: torch.Tensor):
        return _FakeOutput(
            batch_size=pixel_values.shape[0], pooled_value=self.pooled_value
        )


def test_clip_vit_b16_returns_pooled_features(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained",
        lambda model_id, use_fast=False: _FakeImageProcessor(),
    )
    monkeypatch.setattr(
        "transformers.CLIPVisionModel.from_pretrained",
        lambda model_id: _FakeVisionModel(pooled_value=1.0),
    )

    encoder = CLIPViTB16()
    image = PIL.Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    features = encoder.forward(images=[image, image])

    assert features.shape == (2, 768)
    assert torch.all(features == 1)


def test_dinov2_vit_b14_returns_pooled_features(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoImageProcessor.from_pretrained",
        lambda model_id, use_fast=False: _FakeImageProcessor(),
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda model_id: _FakeVisionModel(pooled_value=2.0),
    )

    encoder = DINOv2ViTB14()
    image = PIL.Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    features = encoder.forward(images=[image, image])

    assert features.shape == (2, 768)
    assert torch.all(features == 2)

import PIL.Image
import numpy as np
import pytest
import torch
from typing import Any, cast

from enczoo.base import ImageEncoding
from enczoo.encoders.pixels import Pixels
import enczoo.wrappers.random_projection as random_projection_layer


@pytest.fixture
def input_tensor() -> torch.Tensor:
    return torch.tensor(
        np.array(
            [
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                    0.15494743,
                    0.37816252,
                    -0.88778575,
                    -1.98079647,
                ],
                [
                    -0.34791215,
                    0.15634897,
                    1.23029068,
                    1.20237985,
                    -0.38732682,
                    -0.30230275,
                    -1.04855297,
                    -1.42001794,
                    -1.70627019,
                    1.9507754,
                    -0.50965218,
                    -0.4380743,
                    -1.25279536,
                    0.77749036,
                    -1.61389785,
                    -0.21274028,
                ],
                [
                    -0.89546656,
                    0.3869025,
                    -0.51080514,
                    -1.18063218,
                    -0.02818223,
                    0.42833187,
                    0.06651722,
                    0.3024719,
                    -0.63432209,
                    -0.36274117,
                    -0.67246045,
                    -0.35955316,
                    -0.81314628,
                    -1.7262826,
                    0.17742614,
                    -0.40178094,
                ],
                [
                    -1.63019835,
                    0.46278226,
                    -0.90729836,
                    0.0519454,
                    0.72909056,
                    0.12898291,
                    1.13940068,
                    -1.23482582,
                    0.40234164,
                    -0.68481009,
                    -0.87079715,
                    -0.57884966,
                    -0.31155253,
                    0.05616534,
                    -1.16514984,
                    0.90082649,
                ],
            ]
        ),
        dtype=torch.float,
    )


def test_deterministic(input_tensor):
    mod = random_projection_layer.RandomProjectionLayer(
        in_features=input_tensor.shape[1],
        out_features=1000,
        seed=0,
    )

    y = mod(input_tensor)
    y2 = mod(input_tensor)

    assert torch.allclose(y, y2)


def test_projection_weight_variance():
    out_features = 1000
    mod = random_projection_layer.RandomProjectionLayer(
        in_features=256,
        out_features=out_features,
        seed=0,
    )

    weights = cast(torch.Tensor, mod.projection_weights)
    expected_std = 1 / np.sqrt(out_features)
    assert torch.isclose(
        weights.mean(), torch.tensor(0.0, dtype=weights.dtype), atol=1e-2
    )
    assert torch.isclose(
        weights.std(),
        torch.tensor(expected_std, dtype=weights.dtype),
        atol=2e-3,
    )


class _ToyEncoding(ImageEncoding):
    def compute_features(
        self,
        images: list[PIL.Image.Image],
        flatten: bool = False,
        seed: int | None = None,
    ) -> np.ndarray:
        del seed
        self.validate_images(images)
        features = np.arange(len(images) * 6, dtype=np.float32).reshape(
            len(images), 2, 3
        )
        if flatten:
            return features.reshape(len(images), -1)
        return features


def test_device_index_requires_gpu():
    with pytest.raises(ValueError, match="device_index can only be set"):
        _ToyEncoding(device_index=0)


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="Unknown device"):
        _ToyEncoding(device=cast(Any, "tpu"))


def test_resolve_requested_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    encoder = Pixels(device="gpu", device_index=1)

    assert encoder.torch_device == torch.device("cuda:1")


def test_projection_wrapper_matches_projection_layer():
    image = PIL.Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    encoder = _ToyEncoding()
    projected = random_projection_layer.RandomProjection(
        encoder=encoder,
        out_features=4,
        seed=7,
    )

    result = projected.compute_features(images=[image, image], flatten=False)
    expected = random_projection_layer.RandomProjectionLayer(
        in_features=6,
        out_features=4,
        seed=7,
    )(torch.from_numpy(encoder.compute_features(images=[image, image], flatten=True)))

    assert result.shape == (2, 4)
    assert np.allclose(result, expected.detach().cpu().numpy())

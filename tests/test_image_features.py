import os
from collections.abc import Callable
from pathlib import Path

import PIL.Image
import numpy as np
import pytest

import enczoo

_dir = Path(__file__).parent


# %%
pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Skipped on GitHub Actions because this test loads heavyweight pretrained models.",
)


@pytest.fixture
def test_images() -> list[PIL.Image.Image]:
    imagesdir = _dir / "test_images"
    images = []
    for path in sorted(imagesdir.glob("*.png")):
        with PIL.Image.open(path) as img:
            img = img.convert("RGB")
            images.append(img.copy())

    assert len(images) == 5
    return images


@pytest.mark.parametrize(
    argnames="target_filename, model_constructor",
    argvalues=[
        (
            "target_alexnet_classifier5.npy",
            lambda: enczoo.AlexNet(layer_name="classifier.5"),
        ),
        (
            "target_rn50_avgpool.npy",
            lambda: enczoo.ResNet50(layer_name="avgpool"),
        ),
        (
            "target_rn50_avgpool_proj20_seed0.npy",
            lambda: enczoo.RandomProjection(
                encoder=enczoo.ResNet50(layer_name="avgpool"),
                out_features=20,
                seed=0,
            ),
        ),
        (
            "target_rn50_avgpool_proj1000_seed0.npy",
            lambda: enczoo.RandomProjection(
                encoder=enczoo.ResNet50(layer_name="avgpool"),
                out_features=1000,
                seed=0,
            ),
        ),
        (
            "target_rn50_avgpool_proj1000_seed1.npy",
            lambda: enczoo.RandomProjection(
                encoder=enczoo.ResNet50(layer_name="avgpool"),
                out_features=1000,
                seed=1,
            ),
        ),
    ],
)
def test_feature_regression(
    test_images,
    target_filename: str,
    model_constructor: Callable[[], enczoo.ImageEncoding],
):
    model = model_constructor()

    # Load test target:
    test_target = np.load(_dir / "test_targets" / target_filename)

    # Run forward:
    result = model.compute_features(images=test_images)

    # Run forward again:
    result2 = model.compute_features(images=test_images)

    print(np.max(np.abs(result - test_target)))
    assert result.shape == result2.shape == test_target.shape
    rtol = 1e-3
    atol = 1e-5
    assert np.allclose(result, test_target, rtol=rtol, atol=atol)
    assert np.allclose(result2, test_target, rtol=rtol, atol=atol)

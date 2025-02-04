import PIL.Image
import enczoo
from typing import List
from pathlib import Path
import pytest
import numpy as np
import json

_dir = Path(__file__).parent


@pytest.fixture
def test_images() -> List[PIL.Image]:
    imagesdir = _dir / 'test_images'
    images = []
    for path in sorted(imagesdir.glob('*.png')):
        with PIL.Image.open(path) as img:
            img = img.convert("RGB")
            images.append(img.copy())

    assert len(images) == 5
    return images


def test_alexnet_regresses(test_images):
    # Spot checks the penultimate layer of AlexNet
    test_target = np.load(_dir / 'test_targets'/ 'target_alexnet_classifier5.npy')

    enc = enczoo.AlexNet(
        layer_name='classifier.5',
        random_projection_dim=None,
        random_projection_seed=0
    )

    result = enc.compute_features(
        images=test_images,
    )

    result = result.detach().cpu().numpy()
    assert result.shape == test_target.shape
    assert np.allclose(result, test_target, atol=1e-3)


def test_rn50_regresses(test_images):
    # Spot checks the penultimate layer of ResNet50

    test_target = np.load(_dir / 'test_targets'/ 'target_rn50_avgpool.npy')

    enc = enczoo.ResNet50(
        layer_name='avgpool',
        random_projection_dim=None,
        random_projection_seed=0
    )

    result = enc.compute_features(
        images=test_images,
    )

    result = result.detach().cpu().numpy()
    assert result.shape == test_target.shape
    assert np.allclose(result, test_target, atol=1e-3)


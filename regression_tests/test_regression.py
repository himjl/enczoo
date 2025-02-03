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

    assert len(images) == 1
    return images


@pytest.fixture
def test_target() -> np.ndarray:
    x = json.loads((_dir / 'test_targets' / 'alexnet_target.json').read_text())
    x = np.array(x)
    return x

def test_alexnet_regresses(test_images, test_target):
    enc = enczoo.AlexNet(
        layer_name = enczoo.AlexNet.layer_names[-2],
        random_projection_dim=None,
        random_projection_seed=0
    )

    result = enc.compute_features(
        images=test_images,
    )

    result = result.detach().cpu().numpy()[0]
    assert np.allclose(result, test_target, atol=1e-3)
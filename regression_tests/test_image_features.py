import PIL.Image
import numpy as np
import pytest
from pathlib import Path
from typing import List, Callable

import enczoo
from enczoo.config import ImageEncodingConfig

_dir = Path(__file__).parent


# %%
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


@pytest.mark.parametrize(
    argnames="target_filename, model_constructor",
    argvalues=[
        ('target_alexnet_classifier5.npy', lambda config: enczoo.AlexNet(layer_name='classifier.5', config=config)),
        ('target_rn50_avgpool.npy', lambda config: enczoo.ResNet50(layer_name='avgpool', config=config)),
        ('target_rn50_avgpool_proj20_seed0.npy', lambda config: enczoo.ResNet50(layer_name='avgpool', config=config, random_projection_dim=20, random_projection_seed=0)),
        ('target_rn50_avgpool_proj1000_seed0.npy', lambda config: enczoo.ResNet50(layer_name='avgpool', config=config, random_projection_dim=1000, random_projection_seed=0)),
        ('target_rn50_avgpool_proj1000_seed1.npy', lambda config: enczoo.ResNet50(layer_name='avgpool', config=config, random_projection_dim=1000, random_projection_seed=1)),
    ]
)
def test_feature_regression(
        test_images,
        target_filename: str,
        model_constructor: Callable[[ImageEncodingConfig], enczoo.ImageEncoding],
        tmpdir,
):
    config = ImageEncodingConfig(
        cachedir=Path(tmpdir)
    )
    model = model_constructor(config)

    # Load test target:
    test_target = np.load(_dir / 'test_targets' / target_filename)

    # Run forward:
    result = model.compute_features(images=test_images).detach().cpu().numpy()

    # Run forward again:
    result2 = model.compute_features(images=test_images).detach().cpu().numpy()

    # Try using load_features:
    result3 = model.load_features(images=test_images, cache_new_features=True).detach().cpu().numpy()
    result4 = model.load_features(images=test_images, cache_new_features=False).detach().cpu().numpy()  # Cache hit

    print(np.max(np.abs(result - test_target)))
    assert result.shape == result2.shape == result3.shape == result4.shape == test_target.shape
    rtol = 1e-3
    atol = 1e-5
    assert np.allclose(result, test_target, rtol=rtol, atol=atol)
    assert np.allclose(result2, test_target, rtol=rtol, atol=atol)
    assert np.allclose(result3, test_target, rtol=rtol, atol=atol)
    assert np.allclose(result4, test_target, rtol=rtol, atol=atol)

from typing import List

import PIL.Image
import numpy as np
import pytest
import torch

from enczoo import AlexNet


@pytest.fixture
def alexnet():
    return AlexNet(
        layer_name=AlexNet.layer_names[-1],
        random_projection_dim=None,
        random_projection_seed=0,
    )


@pytest.fixture
def images() -> List[PIL.Image.Image]:
    np.random.seed(0)
    img_dat1 = PIL.Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    img_dat2 = PIL.Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    return [img_dat1, img_dat2]


@pytest.fixture
def image_batch1(images):
    return [images[0]]


@pytest.fixture
def image_batch2(images):
    return images


def test_default_is_not_training(alexnet):
    assert not alexnet.training


def test_alexnet_forward(alexnet, image_batch1, image_batch2):
    # Test output is deterministic
    result = alexnet(images=image_batch1)
    result2 = alexnet(images=image_batch1)
    assert torch.allclose(result, result2)

    # Test output does not depend on batch size
    result_bigger_batch = alexnet(images=image_batch2)
    assert torch.allclose(result[0], result_bigger_batch[0], rtol=1e-3)

# %%
import pytest
import torch

import enczoo
import enczoo.utils as utils


@pytest.mark.parametrize(
    argnames="model_factory",
    argvalues=[
        lambda: enczoo.Pixels(size=16),
        lambda: enczoo.Pixels(
            size=16, random_projection_dim=10, random_projection_seed=0
        ),
        lambda: enczoo.ResNet50(layer_name="layer1.1.relu"),
        lambda: enczoo.ResNet50(layer_name="avgpool"),
        lambda: enczoo.ResNet50(
            layer_name="avgpool",
            random_projection_dim=1000,
            random_projection_seed=0,
        ),
        lambda: enczoo.ResNet50(
            layer_name="avgpool",
            random_projection_dim=1000,
            random_projection_seed=1,
        ),
        lambda: enczoo.ResNet50(
            layer_name="avgpool",
            random_projection_dim=500,
            random_projection_seed=0,
        ),
        lambda: enczoo.AlexNet(
            layer_name="classifier.5",
            random_projection_dim=1000,
            random_projection_seed=0,
        ),
    ],
)
def test_model_hashing(model_factory):
    model = model_factory()
    hash_1 = model.module_hash
    hash_2 = model.module_hash
    assert hash_1 == hash_2
    assert utils.hash_torch_module(model) == hash_1

    model_2 = model_factory()
    assert model_2.module_hash == hash_1


def test_module_hash_device_invariant():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = enczoo.Pixels(
        size=16,
        random_projection_dim=10,
        random_projection_seed=0,
    )
    cpu_hash = utils.hash_torch_module(model)
    gpu_hash = utils.hash_torch_module(model.to("cuda"))
    assert cpu_hash == gpu_hash

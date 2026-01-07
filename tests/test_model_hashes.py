# %%
import pytest
import torch

import enczoo
import enczoo.utils as utils


@pytest.mark.parametrize(
    argnames="target_hash, model",
    argvalues=[
        (
            "8cc79290dc3859ce6bfaebc75272bd054ac638f51aba2403cc73158d8af05a49",
            enczoo.Pixels(size=16),
        ),
        (
            "8b6b33b3e0f8ca48a21515c1fdf00693379eed1079501604b1821e3c45725faf",
            enczoo.Pixels(size=16, random_projection_dim=10, random_projection_seed=0),
        ),
        (
            "6c0180097dfa07ceb88d615aaafe3204292ce210082648da4723ca854100fb0e",
            enczoo.ResNet50(layer_name="layer1.1.relu"),
        ),
        (
            "aecd510a627ffe402f89589bd57474764443e9c49f43f63193aafc73a26f59bb",
            enczoo.ResNet50(layer_name="avgpool"),
        ),
        (
            "4845c1120560d42b2b6ab801701becb9dfb57de5b26040ac19f07598c0a64339",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=1000,
                random_projection_seed=0,
            ),
        ),
        (
            "d5d49cd7e65556c845c6adbf5eddf54cc733002a11ff2e1f9118df4c3d10f0e4",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=1000,
                random_projection_seed=1,
            ),
        ),
        (
            "9ae436a02da38e2a16d7e5440e253edb3a7c721b5835fdaf18a99567c647942e",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=500,
                random_projection_seed=0,
            ),
        ),
        (
            "7ab43a470323fd8870202d36adf089c63cfc7c33844bf416abb998ec6e5a3b31",
            enczoo.AlexNet(
                layer_name="classifier.5",
                random_projection_dim=1000,
                random_projection_seed=0,
            ),
        ),
    ],
)
def test_model_hashing(target_hash: str, model: enczoo.ImageEncoding):
    # b33ff99fc766b5af5535ffa87973d1db905ac1183bdf23e0dac9e11a07f016d9
    # Self-reported hash
    assert model.module_hash == target_hash

    # Manual hash
    assert utils.hash_torch_module(model) == target_hash


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

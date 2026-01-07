# %%
import pytest
import torch

import enczoo
import enczoo.utils as utils


@pytest.mark.parametrize(
    argnames="target_hash, model",
    argvalues=[
        (
            "889b45d57f2c609b89addefcdb584560a330078de8bd5f3d6436e063b6b70d68",
            enczoo.Pixels(size=16),
        ),
        (
            "57497bf1c2a7fbfd1f14643f4ff5f5b052b39843d929647624c76c350dd06069",
            enczoo.Pixels(size=16, random_projection_dim=10, random_projection_seed=0),
        ),
        (
            "dcf5cf6703147b21602fd76241fe4c0ac3c3da6ca577155d7497826ec8e6c006",
            enczoo.ResNet50(layer_name="layer1.1.relu"),
        ),
        (
            "a2839e27342e1736f1e1154902e025e46119a2d404c062f68e79547b6ede3f07",
            enczoo.ResNet50(layer_name="avgpool"),
        ),
        (
            "671006fac1f541e4b28ed3de60c824ae3b9969fd622ab8e2a81cbe72e6fc4eed",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=1000,
                random_projection_seed=0,
            ),
        ),
        (
            "b1d08c3564b38c3210a2c76518bab3f26ee61e5cbcc866289d41045cd9db57d8",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=1000,
                random_projection_seed=1,
            ),
        ),
        (
            "ab7f88b0371d9c899acbfbf4c4bf7e6c883106f5c8a298f489221d4b2e86851f",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=500,
                random_projection_seed=0,
            ),
        ),
        (
            "b3ba9cb815d3bfaf21c2c45a3b3e152dc6d40a35e49950ea76e4c2ba0b6e8413",
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

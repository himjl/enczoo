# %%
import pytest

import enczoo
import enczoo.utils as utils


@pytest.mark.parametrize(
    argnames="target_hash, model",
    argvalues=[
        ('8cc79290dc3859ce6bfaebc75272bd054ac638f51aba2403cc73158d8af05a49', enczoo.Pixels(size=16)),
        ('6c0180097dfa07ceb88d615aaafe3204292ce210082648da4723ca854100fb0e', enczoo.ResNet50(layer_name='layer1.1.relu')),
        ('aecd510a627ffe402f89589bd57474764443e9c49f43f63193aafc73a26f59bb', enczoo.ResNet50(layer_name='avgpool')),
        ('d9cb0ba6812fc9708adb307b0d749137cd6ff1198ea891569c0461bde70fb483', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=1000, random_projection_seed=0)),
        ('77413ddc04e9dcc9b9b4eea9e45b5602eb249a38c84f0cc9c90f7da1a9dff938', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=1000, random_projection_seed=1)),
        ('c01b48cd3574408a3fcbc176d9e24b75e782873a7ffe7733ca5f6e43f62ab252', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=500, random_projection_seed=0)),
        ('b33f496f66623b23c2e3e8e21a9a7de36d89c4557668090c6e410ba5955edc6a', enczoo.AlexNet(layer_name='classifier.5', random_projection_dim=1000, random_projection_seed=0)),
    ]
)
def test_model_hashing(
        target_hash: str,
        model: enczoo.ImageEncoding
):
    # b33ff99fc766b5af5535ffa87973d1db905ac1183bdf23e0dac9e11a07f016d9
    # Self-reported hash
    assert model.module_hash == target_hash

    # Manual hash
    assert utils.hash_torch_module(model) == target_hash

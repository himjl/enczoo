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
        #('473094d9523072acb77ac0acd03d0f0273db0dbf4969f1ab18987f15fa135fee', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=1000, random_projection_seed=0)),
        #('ef91ed28708f0dc4306afcbdd6cd9afbde71814a95e5b1dc5f00219b9c79cdca', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=1000, random_projection_seed=1)),
        #('239fd06b4253c6d252be4729b8cec8a257f4e6738d90106f910a40ba74d9fbc4', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=500, random_projection_seed=0)),
        #('dce2a341f61357daa6ad4acce4366a970c0df0aa796b617c7988ee1c895ee46c', enczoo.AlexNet(layer_name='classifier.5', random_projection_dim=1000, random_projection_seed=0)),
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

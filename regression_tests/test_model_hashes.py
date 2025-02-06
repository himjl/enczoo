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
        ('893c0e0d7e72729eb6f741e7fa3cca88afd34afaddf9a3993c6595803b24b546', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=1000, random_projection_seed=0)),
        ('ef1fecd65470df7709176a75ddca8145b6d84a81fba177bcb5c20e5ff72b9e46', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=1000, random_projection_seed=1)),
        ('ba0bf7c02ba4e991a8e934669447e08fb3aee7bab75753fe0294fe1dfabca9c7', enczoo.ResNet50(layer_name='avgpool', random_projection_dim=500, random_projection_seed=0)),
        ('14f5feda12caa94047dda42c319dc2daa46017c3272aa4900840f2ec00008747', enczoo.AlexNet(layer_name='classifier.5', random_projection_dim=1000, random_projection_seed=0)),
    ]
)
def test_model_hashing(
        target_hash: str,
        model: enczoo.ImageEncoding
):
    # Self-reported hash
    assert model.module_hash == target_hash

    # Manual hash
    assert utils.hash_torch_module(model) == target_hash

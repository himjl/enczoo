# %%
import pytest

import enczoo
import enczoo as utils


@pytest.mark.parametrize(
    argnames="target_hash, model",
    argvalues=[
        (
            "8cc79290dc3859ce6bfaebc75272bd054ac638f51aba2403cc73158d8af05a49",
            enczoo.Pixels(size=16),
        ),
        (
            "f22b76e262e4361405da19129993ff00c8b9266db7d8b1e92807afbb378fb92e",
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
            "9163abc4523fb7349d357d266e3f6b7c7c6751f15af36ffda2e8d1f9d4f81b92",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=1000,
                random_projection_seed=0,
            ),
        ),
        (
            "cd6540b12a82c605d0ea7ad3747acc3b886a735633f11ac3a8fd2f76617fa209",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=1000,
                random_projection_seed=1,
            ),
        ),
        (
            "309f29dac32c5a2f907b7f8ee3ecb3ef51f165aff2f9d66b79801574d0186387",
            enczoo.ResNet50(
                layer_name="avgpool",
                random_projection_dim=500,
                random_projection_seed=0,
            ),
        ),
        (
            "9af8f145bff0a399cafac5852f4a39a975b0a5ee246adb4d74f392f507301da8",
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

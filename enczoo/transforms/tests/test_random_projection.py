import numpy as np
import pytest
import torch

import enczoo.transforms.random_projection as random_projection_layer
import enczoo.utils as utils


@pytest.fixture
def input_tensor() -> torch.Tensor:
    return torch.tensor(
        np.array(
            [
                [0.3130677, -0.85409574, -2.55298982, 0.6536186, 0.8644362,
                 -0.74216502, 2.26975462, -1.45436567, 0.04575852, -0.18718385,
                 1.53277921, 1.46935877, 0.15494743, 0.37816252, -0.88778575,
                 -1.98079647],
                [-0.34791215, 0.15634897, 1.23029068, 1.20237985, -0.38732682,
                 -0.30230275, -1.04855297, -1.42001794, -1.70627019, 1.9507754,
                 -0.50965218, -0.4380743, -1.25279536, 0.77749036, -1.61389785,
                 -0.21274028],
                [-0.89546656, 0.3869025, -0.51080514, -1.18063218, -0.02818223,
                 0.42833187, 0.06651722, 0.3024719, -0.63432209, -0.36274117,
                 -0.67246045, -0.35955316, -0.81314628, -1.7262826, 0.17742614,
                 -0.40178094],
                [-1.63019835, 0.46278226, -0.90729836, 0.0519454, 0.72909056,
                 0.12898291, 1.13940068, -1.23482582, 0.40234164, -0.68481009,
                 -0.87079715, -0.57884966, -0.31155253, 0.05616534, -1.16514984,
                 0.90082649]
            ]
        ),
        dtype=torch.float
    )


def test_deterministic(input_tensor):
    mod = random_projection_layer.RandomProjection(
        in_features=input_tensor.shape[1],
        out_features=1000,
        seed=0,
    )
    # -1.1258000135421753
    raise Exception(str(float(mod.linear.weight[0, 0])))

    # Add a test for the module hash
    assert utils.hash_torch_module(mod) == '304b73063a1da16f5825b2867f6c8c54a577f259b8cb0614a074367392898578'

    # Check forward pass
    y = mod(input_tensor)
    assert torch.isclose(y[0, 0], torch.tensor(0.037887085, dtype=y.dtype), atol=1e-6)

    # Should be the same result across calls
    y2 = mod(input_tensor)
    assert torch.allclose(y, y2)

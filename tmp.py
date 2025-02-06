import enczoo
import glob
import PIL.Image
from pathlib import Path

imagesdir = Path('/Users/mjl/PycharmProjects/enczoo/regression_tests/test_images')
images = []
for path in sorted(imagesdir.glob('*.png')):
    with PIL.Image.open(path) as img:
        img = img.convert("RGB")
        images.append(img.copy())

assert len(images) == 5


for proj, seed in [(1000, 0), (1000, 1), (20, 0)]:
    model = enczoo.ResNet50(
        layer_name='avgpool',
        random_projection_dim=proj,
        random_projection_seed=seed
    )

    f = model.compute_features(
        images = images
    )
    f2 = model.compute_features(
        images = images
    )

    import numpy as np
    np.save(f'/Users/mjl/PycharmProjects/enczoo/regression_tests/test_targets/target_rn50_avgpool_proj{proj}_seed{seed}', f.detach().cpu().numpy())
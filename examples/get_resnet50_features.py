import enczoo
import PIL.Image
import numpy as np
from tqdm import trange


np.random.seed(0)
image = PIL.Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

model = enczoo.AligNetViTB16()  # layer_name="avgpool")

# model = model.to('cuda')
for i in trange(1000):
    f = model.compute_features(images=[image for _ in range(100)])

print(
    f
)  # array([0.20376027, 0.05251464, 0.00759117, ..., 0.33928823, 0., 0.70025563], shape=(2048,), dtype=float32)

print(f.shape)

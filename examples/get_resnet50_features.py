import enczoo
import PIL.Image
import numpy as np
from tqdm import trange


np.random.seed(0)
image = PIL.Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

model = enczoo.ResNet50(layer_name="avgpool")
model = model.to("cuda")
for i in trange(1000):
    f = model.compute_features(images=[image for _ in range(100)])

print(f)
print(f.shape)

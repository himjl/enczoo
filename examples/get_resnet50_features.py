import enczoo
import PIL.Image
import numpy as np

model = enczoo.ResNet50(layer_name="avgpool")

np.random.seed(0)
image = PIL.Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

f = model.compute_features(images=[image])
print(f) # array([0.20376027, 0.05251464, 0.00759117, ..., 0.33928823, 0., 0.70025563], shape=(2048,), dtype=float32)
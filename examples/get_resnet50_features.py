import enczoo
import PIL.Image
import numpy as np
np.random.seed(0)
model = enczoo.ResNet50(layer_name='avgpool')
image = PIL.Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

f = model.compute_features(images=[image])
# `enczoo`: easily extract image features from pretrained vision models

[![CI](https://github.com/himjl/enczoo/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/enczoo/actions/workflows/ci.yml)

`enczoo` is a Python library with a single goal: to enable you to map images (as `PIL.Images`) to image features (as `numpy` arrays) as easily as possible. 
Features may be extracted from a zoo of popular, pretrained vision models, such as Imagenet-pretrained ResNet50 and CLIP ViT-B/16. 

### Installation

`enczoo` requires Python 3.12 or above, and it's recommended you use the wonderful [uv](https://docs.astral.sh/uv/) to install it. Assuming you have `uv`, just run the following command in your project: 

    uv add enczoo

### Usage 

```python
import enczoo
from PIL import Image

image = Image.open('my-image.png')
model = enczoo.ResNet50(layer_name='avgpool') 
features = model.compute_features(images=[image]) # np.ndarray
# Want another layer? Check out: print(enczoo.ResNet50.layer_names)
```


### Why develop `enczoo`?
Under the hood, `enczoo` solves several tiny problems which make correctly computing image features more annoying and error-prone than it should be. For example, `enczoo` automatically: 
    
* performs model-specific image transforms ("_was it -1 to 1, 0 to 1, or 0-255...?_"),
* ensures images are in RGB format 
* puts the model in inference, not training, mode  
* turns off autograd
* returns tensors as `np.ndarray` (no more `.cpu().numpy()`)
* resizes the image while preserving aspect ratio 
* and more!

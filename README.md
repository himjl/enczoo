# `enczoo`: a zoo of encoding models for images

[![CI](https://github.com/himjl/enczoo/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/enczoo/actions/workflows/ci.yml)

`enczoo` is a Python library with a single goal: to map images (as `PIL.Images`) to features (as `numpy` arrays) extracted from off-the-shelf vision models, such as Imagenet-pretrained ResNet50 and CLIP ViT-B/16.

This library is meant for those who need to compute off-the-shelf image features once for their project.

### Installation

`enczoo` requires Python 3.12 or above, and it's recommended you use the wonderful [uv](https://docs.astral.sh/uv/) to install it. Assuming you have `uv`, just run the following command in your project: 

    uv add enczoo

You can also install `enczoo` using `pip` by running:

    pip install enczoo
 
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
`enczoo` solves several tiny problems which make correctly computing image features more annoying and error-prone than it should be. For example, `enczoo` automatically: 
    
* performs model-specific image transforms ("_was it -1 to 1, 0 to 1, or 0-255...?_"),
* ensures images are in RGB format 
* puts the model in inference, not training, mode  
* turns off autograd
* returns tensors as `np.ndarray` (no more `.cpu().numpy()`)
* resizes the image while preserving aspect ratio 
* and more!

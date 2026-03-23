# `enczoo`: a zoo of encoding models for images

[![CI](https://github.com/himjl/enczoo/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/enczoo/actions/workflows/ci.yml)

`enczoo` is a Python library with a single goal: to map images (as `PIL.Images`) to features (as `np.ndarray`) from intermediate layers of off-the-shelf vision models, such as AlexNet and ResNet50.

This library is meant for those who need to compute off-the-shelf image features once for their project (and perhaps cache them elsewhere). 

### Installation

`enczoo` requires Python 3.12 or above, and it's recommended you use the wonderful [uv](https://docs.astral.sh/uv/) to install it. Assuming you have `uv`, just run the following command in your project: 

    uv add enczoo

You can also install `enczoo` using `pip` by running:

    pip install enczoo
 
### Usage 

```python
import enczoo
import PIL.Image
image = PIL.Image.open('my-image.png')

model = enczoo.ResNet50(layer_name='avgpool') # try layer4, layer3, ...
features = model.compute_features(images=[image]) # np.ndarray

# Want another layer? Check out: print(enczoo.ResNet50.layer_names)
```

### Why develop `enczoo`?
`enczoo` solves several tiny problems which make correctly computing image features more annoying and error-prone than it should be. For example, `enczoo` automatically: 
    
* performs model-specific image normalization ("_was it -1 to 1, 0 to 1, or 0-255...? ImageNet channel normalization...?_"),
* correctly encodes images ("_my image was in mode L, not RGB!_")
* turns off batch normalization ("_was the model in training mode...?_")
* extracts intermediate layers by name ("_how do I do that forward hook thing again...?_")
* turns off autograd, and returns tensors as `np.ndarray` (no more `.cpu().numpy()`)
* performs image cropping to fit images to the expected input tensor shape
* and more!

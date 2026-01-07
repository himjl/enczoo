# `enczoo`: a zoo of encoding models for images

[![CI](https://github.com/himjl/enczoo/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/enczoo/actions/workflows/ci.yml)

`enczoo` is a Python library with a single goal: to map images (as `PIL.Images`) to intermediate representations (as `np.ndarray`) from off-the-shelf vision models, such as AlexNet and ResNet50.

This library is meant for those who just need to compute off-the-shelf image features once for their project (and perhaps cache them elsewhere).

### Installation

`enczoo` requires Python 3.12 or above, and may be installed using [uv](https://docs.astral.sh/uv/) with the following command: 

>`uv add enczoo`
 
### Usage 

```python
import enczoo
import PIL.Image
image = PIL.Image.open('my-image.png')

model = enczoo.ResNet50(layer_name='avgpool') # try layer4, layer3, ...
features = model.compute_features(images=[image]) # np.ndarray

# Want another layer? Check out: print(enczoo.ResNet50.layer_names)
```

### Things `enczoo` handles
`enczoo` aims to "just work" by solving several tiny problems which collectively make computing image features a bit annoying. `enczoo` handles: 
    
* performing model-specific image normalization ("_was it -1 to 1, 0 to 1, 0-255...? ImageNet channel normalization...?_"),
* correctly encoding images ("_my image was in mode L, not RGB!_")
* turning off any batch normalization ("_was the model in training mode...?_")
* extracting intermediate layers by name ("_how do I do that forward hook thing again...?_")
* turning off autograd, and returning tensors as `np.ndarray` (no more `.cpu().numpy()`)
* image cropping to fit input tensor shape (default: center cropping. no black bars!)
* and more!

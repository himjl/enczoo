# `enczoo`: a zoo of encoding models for images

[![CI](https://github.com/himjl/enczoo/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/enczoo/actions/workflows/ci.yml)

`enczoo` is a Python library with a single goal: to map images (as `PIL.Images`) to intermediate representations (as `np.ndarray`) from off-the-shelf vision models, such as AlexNet and ResNet50.

This library is meant for those who just need to compute off-the-shelf image features once for their project (and perhaps cache them elsewhere).

### Installation

`enczoo` requires Python 3.12 or above, and may be installed using [uv](https://docs.astral.sh/uv/) with the following command: 

>`uv add enczoo`
 

### Goal of `enczoo`
`enczoo` aims to "just work" by solving several tiny problems which collectively make computing image features a bit annoying. `enczoo` handles: 
    
* performing model-specific image normalization ("_was it -1 to 1, 0 to 1, 0-255...? ImageNet channel normalization...?_"),
* correctly encoding images ("_my image was in mode L, not RGB!_")
* turning off any batch normalization ("_was the model in training mode...?_")
* seeding randomness ("_why did I get different results when I ran the model again...?_")
* extracting intermediate layers by name ("_how do I do that forward hook thing again...?_")
* turning off autograd, and returning tensors as `np.ndarray` (no more `.cpu().numpy()`)
* image cropping to fit input tensor shape (default: center cropping. no black bars!)
* and more!

`enczoo` also has knobs for many other things one often find themselves doing, such as:  
* dimensionality reduction, for gigantic feature spaces that (default: seeded [random projection](https://en.wikipedia.org/wiki/Johnson–Lindenstrauss_lemma))



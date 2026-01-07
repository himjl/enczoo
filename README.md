# `enczoo`: a zoo of encoding models for images

[![CI](https://github.com/himjl/enczoo/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/enczoo/actions/workflows/ci.yml)

`enczoo` is a Python library with a single goal: to map images (as `PIL.Images`) to intermediate representations (as `np.ndarray`) from off-the-shelf vision models, such as AlexNet and ResNet50.

`enczoo` aims to "just work" by solving several tiny problems which collectively make computing image features a bit annoying. `enczoo` handles: 
    
* performing model-specific image normalization ("_was it -1 to 1, 0 to 1, 0-255...? ImageNet channel normalization...?_"),
* correctly encoding images ("_my image is in mode L, not RGB!_")
* turning off any batch normalization ("_was the model in training mode...?_")
* seeding randomness ("_I ran the model twice on the same image, and got different results...?_")
* extracting intermediate layers by name ("_how do I do that forward hook thing again...?_")
* turning off autograd, and returning tensors as `np.ndarray` (no more `.cpu().numpy()`)
* and other stuff too!

`enczoo` also has knobs set to good defaults for many other things one often  find themselves doing, such as:  
* image cropping to fit input tensor shape (default: center cropping. no black bars!) 
* dimensionality reduction, for gigantic feature spaces that (default: seeded [random projection](https://en.wikipedia.org/wiki/Johnson–Lindenstrauss_lemma))



### What `enczoo` does not do: 
* `enczoo` is not really optimized for performance (though it does its best). 
 * `enczoo` does not offer autograd capabilities or integrate with torch/tensorflow (its only job is to return `np.ndarray` from `PIL.Image`, after all). 

This library is best used for those who just need to compute off-the-shelf image features once for their project (and perhaps cache them elsewhere).
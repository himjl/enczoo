# `enczoo`: easily extract image features from pretrained vision models

[![CI](https://github.com/himjl/enczoo/actions/workflows/ci.yml/badge.svg)](https://github.com/himjl/enczoo/actions/workflows/ci.yml)

`enczoo` is a Python library with a simple goal: to make it as **easy as possible** to map images (as `PIL.Images`) to features (as `numpy` arrays) from state-of-the-art vision models, such as Imagenet-pretrained ResNet50 and CLIP ViT-B/16. 

### Installation

`enczoo` requires Python 3.12 or above, and is installed using the wonderful [uv](https://docs.astral.sh/uv/) project manager. Once you have uv installed, just run the following command in your project: 

    uv add enczoo

### Usage 

```python
import enczoo
from PIL import Image

image = Image.open('my-image.png')
model = enczoo.ResNet50(
    layer_name='avgpool', 
    # device=gpu
) 
features = model.compute_features(images=[image]) # np.ndarray
# Want another layer? Check out: print(enczoo.ResNet50.layer_names)
```

### Available models 

<details>
<summary><code>Pixels</code></summary>

- Family: raw pixels
- Returns: float32 RGB pixels after preprocessing
- Output shape: `[B, 224, 224, 3]`
- Academic reference: none; this is an `enczoo` convenience encoder

</details>

<details>
<summary><code>AlexNet</code></summary>

- Family: ImageNet-pretrained CNN
- Returns: intermediate activations from the requested layer
- Output shape: depends on `layer_name`
- Layer selection: inspect `enczoo.AlexNet.layer_names`
- Academic reference: AlexNet, "ImageNet Classification with Deep Convolutional Neural Networks" ([Krizhevsky et al., 2012](https://cacm.acm.org/research/imagenet-classification-with-deep-convolutional-neural-networks/))

</details>

<details>
<summary><code>ResNet50</code></summary>

- Family: ImageNet-pretrained CNN
- Returns: intermediate activations from the requested layer
- Output shape: depends on `layer_name`
- Layer selection: inspect `enczoo.ResNet50.layer_names`
- Academic reference: ResNet, "Deep Residual Learning for Image Recognition" ([He et al., 2015](https://arxiv.org/abs/1512.03385))

</details>

<details>
<summary><code>ConvNeXtB</code></summary>

- Family: ImageNet-pretrained CNN
- Returns: intermediate activations from the requested layer
- Output shape: depends on `layer_name`
- Layer selection: inspect `enczoo.ConvNeXtB.layer_names`
- Academic reference: ConvNeXt, "A ConvNet for the 2020s" ([Liu et al., 2022](https://arxiv.org/abs/2201.03545))

</details>

<details>
<summary><code>CLIPResNet50</code></summary>

- Family: CLIP ResNet visual encoder
- Returns: intermediate activations from the requested visual layer
- Output shape: depends on `layer_name`
- Layer selection: inspect `enczoo.CLIPResNet50.layer_names`
- Academic reference: CLIP, "Learning Transferable Visual Models From Natural Language Supervision" ([Radford et al., 2021](https://arxiv.org/abs/2103.00020))

</details>

<details>
<summary><code>CLIPViTB16</code></summary>

- Family: CLIP vision transformer
- Returns: the model's pooled CLS-based image embedding
- Output shape: `[B, 768]`
- Academic reference: CLIP, "Learning Transferable Visual Models From Natural Language Supervision" ([Radford et al., 2021](https://arxiv.org/abs/2103.00020))

</details>

<details>
<summary><code>DINOv2ViTB14</code></summary>

- Family: self-supervised vision transformer
- Returns: the model's pooled CLS-based image embedding
- Output shape: `[B, 768]`
- Academic reference: DINOv2, "DINOv2: Learning Robust Visual Features without Supervision" ([Oquab et al., 2023](https://arxiv.org/abs/2304.07193))

</details>

<details>
<summary><code>AligNetViTB16</code></summary>

- Family: AlignNet-aligned vision transformer
- Returns: the SavedModel feature tensor selected from the exported `pre_logits` output
- Output shape: depends on the downloaded model
- Weights: downloaded on first use and cached under `ENCZOO_CACHE_DIR` or the platform cache directory
- Academic reference: Muttenthaler et al. 2025; weights come from the [AlignNet model release](https://storage.googleapis.com/alignet/models/)

</details>

<details>
<summary><code>UnaligNetViTB16</code></summary>

- Family: unaligned vision transformer from the AlignNet release
- Returns: the SavedModel feature tensor selected from the exported `pre_logits` output
- Output shape: depends on the downloaded model
- Weights: downloaded on first use and cached under `ENCZOO_CACHE_DIR` or the platform cache directory
- Academic reference: Muttenthaler et al. 2025; weights come from the [AlignNet model release](https://storage.googleapis.com/alignet/models/)

</details>


### Why develop `enczoo`?
Under the hood, `enczoo` solves several tiny problems which make correctly computing image features more annoying and error-prone than it should be. For example, `enczoo` automatically: 
    
* performs model-specific image transforms ("_was it -1 to 1, 0 to 1, or 0-255...?_"),
* ensures images are in RGB format 
* puts the model in inference, not training, mode  
* turns off autograd
* returns tensors as `np.ndarray` (no more `detach().cpu().numpy()`)
* resizes the image while preserving aspect ratio 
* and more!

import PIL.Image
import numpy as np
import torch
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import List, Tuple, Union, Dict

import enczoo.utils as utils
import mref
import tensorbucket
from enczoo.config import ImageEncodingConfig, default_config


# %%
class ImageEncoding(
    torch.nn.Module, # Todo: move to PyTorch-based subclass
    ABC,
):
    """
    torch.nn.Module which executes a mapping from B-length lists of PIL.Images to [B, *] float tensors.
    Its parameters do not aggregate gradients.
    """

    def __init__(
            self,
            config: Union[ImageEncodingConfig, None],
    ):
        super().__init__()
        self.config: ImageEncodingConfig = config if config is not None else default_config
        self._module_hash = None
        self._tensor_bucket = None
        self._output_shape = None

        # Set the module's mode
        self.train(mode=self.config.trainable)

    @property
    def output_shape(self) -> Tuple[int, ...]:
        if self._output_shape is None:
            test_image = PIL.Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            test_result = self.compute_features(images=[test_image], flatten=False)
            if not isinstance(test_result, torch.Tensor):
                raise ValueError(f'Expected a torch.Tensor from self.forward, but got {type(test_result)}')
            if not test_result.shape[0] == 1:
                raise ValueError(f'Expected a batch size of 1, but got {test_result.shape}')
            if len(test_result.shape) == 1:
                output_shape = tuple()
            else:
                output_shape = test_result.shape[1:]

            self._output_shape = tuple(output_shape)

        return self._output_shape

    # Todo: move to PyTorch-based subclass or make abstract
    @property
    def module_hash(self) -> str:
        if self.config.trainable:
            raise ValueError('Cannot hash a trainable model.')

        # Turn off gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Hash self if unhashed
        if self._module_hash is None:
            self._module_hash = utils.hash_torch_module(module=self)
        return self._module_hash

    @property
    def tensor_bucket(self) -> tensorbucket.TensorBucket:
        if self.config.trainable:
            # Caching not supported for a trainable model
            raise ValueError('Cannot use a tensor bucket with a trainable model.')

        # Initialize tensor bucket (for caching)
        if self._tensor_bucket is not None:
            return self._tensor_bucket

        self._tensor_bucket = tensorbucket.TensorBucket(
            loc=self.config.cachedir / self.__class__.__name__ / (self.module_hash + '.h5'),
            in_memory_cache_size_mb=self.config.in_memory_cache_size_mb,
            shape=self.output_shape,
        )
        return self._tensor_bucket

    @torch.no_grad()
    def load_features(
            self,
            images: List[Union[PIL.Image, mref.ImageRef]],
            flatten: bool = False,
            media_store: mref.Storage = None,
            cache_new_features: bool = True,
            batch_size: int = 32,
    ) -> torch.Tensor:
        """
        A convenience method which loads the features for a list of images from a cache.
        Unlike self.__call__(), this method does not track gradients.

        Features for images not in the cache are computed using self.__call__, and cached,
        if cache_new_features=True and self.config.trainable=False.

        When new features are computed, this method performs image batching to avoid memory issues.

        If any ImageRefs are given, a media_store must be provided to retrieve the images.

        :param images:
        :param cache_new_features:
        :param batch_size:
        :return:
        """

        if self.config.trainable and cache_new_features:
            raise ValueError('Cannot cache new features unless the model has self.config.trainable=False.')

        if batch_size < 1:
            raise ValueError(f'batch_size must be at least 1, but got {batch_size}.')

        # If any ImageRefs are given, check if the tensor bucket has the corresponding tensors.
        image_refs = []
        ref_to_image: Dict[mref.ImageRef, PIL.Image] = {}
        for image in images:
            if isinstance(image, mref.ImageRef):
                image_refs.append(image)
            elif isinstance(image, PIL.Image.Image):
                image_ref = mref.ImageRef.from_image(image=image)
                image_refs.append(image_ref)
                ref_to_image[image_ref] = image
            else:
                raise ValueError(f'Unsupported image type: {type(image)}')

        tensor_already_cached_mask: List[bool] = self.tensor_bucket.check_keys_exist(keys=[v.sha256 for v in image_refs])

        # Collect ImageRefs for which new features must be computed
        compute_image_refs: List[mref.ImageRef] = []
        for already_cached, ref in zip(tensor_already_cached_mask, image_refs):
            if not already_cached:
                compute_image_refs.append(ref)

        compute_image_refs = sorted(set(compute_image_refs))

        # Compute and cache backbone features for any new ImageRefs:
        delete_keys = []
        ncompute_images = len(compute_image_refs)
        pbar = tqdm(total=len(compute_image_refs), desc='Computing image features', disable=ncompute_images <= batch_size)
        for batch_image_refs in utils.iterate_batches(compute_image_refs, batch_size=batch_size):
            # Resolve ImageRefs into PIL.Images:
            batch_images = []
            for image_ref in batch_image_refs:
                if image_ref in ref_to_image:
                    batch_images.append(ref_to_image[image_ref])
                else:
                    image = media_store.load_image(
                        ref=image_ref
                    )
                    batch_images.append(image)

            # Run forward pass:
            batch_features = self.compute_features(images=batch_images, flatten=False)
            batch_backbone_features = batch_features.detach().cpu().numpy()

            # Cache the backbone features in the store, possibly temporarily
            key_to_tensor = {image_ref.sha256: tensor for image_ref, tensor in zip(batch_image_refs, batch_backbone_features)}
            self.tensor_bucket.store_tensors(
                key_to_tensor=key_to_tensor,
                overwrite_if_exists=False,
            )

            if not cache_new_features:
                delete_keys += [v.sha256 for v in batch_image_refs]

            pbar.update(len(batch_image_refs))
        pbar.close()

        # Assemble return tensor:
        features = self.tensor_bucket.retrieve_tensors(keys=[v.sha256 for v in image_refs])
        features = np.stack(features, axis=0)
        features = torch.from_numpy(features)

        if flatten:
            features = features.reshape(features.shape[0], -1)

        # Delete any newly created keys from the tensor_bucket if caching was not requested
        if not cache_new_features:
            self.tensor_bucket.delete_tensors(keys=delete_keys)

        return features

    def compute_features(
            self,
            images: List[PIL.Image],
            flatten: bool = False,
    ) -> torch.Tensor:
        """
        Just an alias for __call__, to allow for type hinting by IDEs
        (PyCharm does not recognize __call__ signature of torch.nn.Modules, for some reason).
        :param images: a B length list of PIL.Images.
        :param flatten: if True, flattens the output tensor to [B, d].
        :return: a torch.Tensor of shape [B, *]. If flatten=True, returns [B, d] instead.
        """
        return self(images=images, flatten=flatten)

    def forward(
            self,
            images: List[PIL.Image],
            flatten: bool = False,
    ) -> torch.Tensor:
        """
        :param images: a B length list of PIL.Images.
        :param flatten: if True, flattens the output tensor to [B, d].
        :return: a torch.Tensor of shape [B, *]
        """
        if not isinstance(images, list):
            raise ValueError(f'Expected a list of PIL.Images, but got {type(images)}')
        if len(images) == 0:
            raise ValueError('Expected a non-empty list of PIL.Images.')
        if not isinstance(images[0], PIL.Image.Image):
            raise ValueError(f'Expected a list of PIL.Images, but element 0 is a {type(images[0])}')

        # Call the subclass implementation
        feats = self._images_to_features(images=images)
        if flatten:
            # Flatten the features
            feats = feats.reshape(feats.shape[0], -1)

        return feats

    @abstractmethod
    def _images_to_features(self, images: List[PIL.Image]) -> torch.Tensor:
        """
        :param images: a list of PIL.Images
        :return: a torch.Tensor of shape [B, *]
        """
        raise NotImplementedError

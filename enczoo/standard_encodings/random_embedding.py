

class RandomEmbedding(Backbone):

    def __init__(self, lookup_refs: List[schema.ImageRef]):
        raise NotImplementedError()
        super().__init__()
        self.sha256_to_index = {}
        lookup_refs = sorted(set(lookup_refs))
        for i, image in enumerate(lookup_refs):
            self.sha256_to_index[image.sha256] = i
        if len(lookup_refs) > 5000:
            warnings.warn(f'Lookup table has {len(lookup_refs)} entries. This may be slow.')

        # Register the lookup_refs as a buffer
        self.register_buffer('lookup_refs', torch.tensor([ord(c) for c in str(lookup_refs)], dtype=torch.int16))

    def _images_to_features(self, images: List[PIL.Image]) -> torch.Tensor:
        """
        :param images: a [B, C, H, W] torch.uint8 tensor.
        :return: a torch.Tensor of shape [B, *]
        """
        # Assemble the return
        f = torch.zeros((len(images), len(self.sha256_to_index)))

        # Preprocess the images
        for image in images:
            sha256 = utils.hash_image(image)
            if sha256 not in self.sha256_to_index:
                raise ValueError(f'Image with SHA256 {sha256} not found in lookup table.')

            i = self.sha256_to_index[sha256]
            f[:, i] = 1

        return f

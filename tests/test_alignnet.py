import io
import tarfile

import PIL.Image
import numpy as np
import tensorflow as tf

from enczoo.alignnet.alignnet import AligNetViTB16


class _BytesResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _FakeSavedModel:
    def __init__(self):
        self.signatures = {"serving_default": self._forward}

    def _forward(self, images):
        batch_size = int(images.shape[0])
        return {"features": tf.ones((batch_size, 768), dtype=tf.float32)}


def _build_model_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        saved_model_bytes = b"placeholder savedmodel"
        info = tarfile.TarInfo(name="ViT-B-alignet/saved_model.pb")
        info.size = len(saved_model_bytes)
        archive.addfile(info, io.BytesIO(saved_model_bytes))
    return buffer.getvalue()


def test_vit_b_alignnet_downloads_and_computes_features(monkeypatch, tmp_path):
    archive_bytes = _build_model_archive()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url: _BytesResponse(archive_bytes),
    )
    monkeypatch.setattr(
        "tensorflow.saved_model.load",
        lambda export_dir: _FakeSavedModel(),
    )

    encoder = AligNetViTB16(cache_dir=tmp_path)
    image = PIL.Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

    features = encoder.compute_features(images=[image, image])

    assert (tmp_path / "alignnet" / "ViT-B-alignet" / "saved_model.pb").exists()
    assert features.shape == (2, 768)
    assert np.all(features == 1.0)

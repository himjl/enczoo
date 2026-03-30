import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import PIL.Image
import numpy as np
import pytest
import tensorflow as tf

from enczoo.encoders.alignnet import AligNetViTB16, UnaligNetViTB16


class _BytesResponse(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class _FakeSavedModel:
    def __init__(self):
        self.signatures = {"serving_default": self._forward}

    def _forward(self, images):
        batch_size = int(images.shape[0])
        return {
            "pre_logits": tf.ones((batch_size, 768), dtype=tf.float32),
            "i1k_logits": tf.zeros((batch_size, 1000), dtype=tf.float32),
            "triplet_logits": tf.zeros((batch_size, 1024), dtype=tf.float32),
        }


def _build_model_archive(model_name: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        saved_model_bytes = b"placeholder savedmodel"
        info = tarfile.TarInfo(name=f"{model_name}/saved_model.pb")
        info.size = len(saved_model_bytes)
        archive.addfile(info, io.BytesIO(saved_model_bytes))
    return buffer.getvalue()


def _install_fake_saved_model(
    monkeypatch,
    archive_bytes_by_url: dict[str, bytes],
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda url: _BytesResponse(archive_bytes_by_url[url]),
    )
    monkeypatch.setattr(
        "tensorflow.saved_model.load",
        lambda export_dir: _FakeSavedModel(),
    )


def test_vit_b_alignnet_downloads_and_computes_features(monkeypatch, tmp_path):
    _install_fake_saved_model(
        monkeypatch,
        {
            AligNetViTB16.weights_url: _build_model_archive(AligNetViTB16.model_name),
        },
    )

    encoder = AligNetViTB16(cache_dir=tmp_path)
    image = PIL.Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

    features = encoder.compute_features(images=[image, image])

    assert (tmp_path / "alignnet" / "ViT-B-alignet" / "saved_model.pb").exists()
    assert features.shape == (2, 768)
    assert np.all(features == 1.0)


def test_unalignet_vit_b_uses_published_model_name_and_url(monkeypatch, tmp_path):
    _install_fake_saved_model(
        monkeypatch,
        {
            UnaligNetViTB16.weights_url: _build_model_archive(
                UnaligNetViTB16.model_name
            ),
        },
    )

    encoder = UnaligNetViTB16(cache_dir=tmp_path)

    assert encoder.model_dir == Path(tmp_path) / "alignnet" / "ViT-B-untransformed"
    assert (encoder.model_dir / "saved_model.pb").exists()


def test_alignnet_preprocessing_uses_resize_then_center_crop():
    image_array = np.zeros((224, 672, 3), dtype=np.uint8)
    image_array[:, :224] = [255, 0, 0]
    image_array[:, 224:448] = [0, 255, 0]
    image_array[:, 448:] = [0, 0, 255]
    image = PIL.Image.fromarray(image_array)

    processed = AligNetViTB16._preprocess_image(image)

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert np.allclose(processed[0, 0], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(processed[0, -1], np.array([0.0, 1.0, 0.0], dtype=np.float32))


def test_alignnet_uses_requested_tensorflow_gpu_index(monkeypatch, tmp_path):
    _install_fake_saved_model(
        monkeypatch,
        {
            AligNetViTB16.weights_url: _build_model_archive(AligNetViTB16.model_name),
        },
    )
    monkeypatch.setattr(
        tf.config,
        "list_logical_devices",
        lambda device_type: (
            [SimpleNamespace(name="/GPU:0"), SimpleNamespace(name="/GPU:1")]
            if device_type == "GPU"
            else [SimpleNamespace(name="/CPU:0")]
        ),
    )

    encoder = AligNetViTB16(cache_dir=tmp_path, device="gpu", device_index=1)

    assert encoder.compute_features(
        images=[PIL.Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))]
    ).shape == (1, 768)
    assert encoder.tensorflow_device_name == "/GPU:1"


def test_alignnet_raises_when_requested_gpu_is_unavailable(monkeypatch, tmp_path):
    _install_fake_saved_model(
        monkeypatch,
        {
            AligNetViTB16.weights_url: _build_model_archive(AligNetViTB16.model_name),
        },
    )
    monkeypatch.setattr(
        tf.config,
        "list_logical_devices",
        lambda device_type: [SimpleNamespace(name="/CPU:0")]
        if device_type == "CPU"
        else [],
    )

    with pytest.raises(ValueError, match="TensorFlow could not find an available GPU"):
        AligNetViTB16(cache_dir=tmp_path, device="gpu")

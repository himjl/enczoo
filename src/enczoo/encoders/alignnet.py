import os
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from abc import ABC
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import PIL.Image
import numpy as np
from tqdm import tqdm

from enczoo.base import DeviceType, TensorflowImageEncoding

_CACHE_ENV_VAR = "ENCZOO_CACHE_DIR"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_MODEL_INPUT_SIZE = 224


def _default_cache_dir() -> Path:
    """Return the default on-disk cache directory for enczoo assets."""
    override = os.environ.get(_CACHE_ENV_VAR)
    if override is not None:
        return Path(override).expanduser()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "enczoo"
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data is not None:
            return Path(local_app_data) / "enczoo"
        return Path.home() / "AppData" / "Local" / "enczoo"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "enczoo"


class _AlignNet(TensorflowImageEncoding, ABC):
    """TensorFlow SavedModel-backed AlignNet encoder."""

    model_name: str
    weights_url: str
    output_key: str | None = "pre_logits"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        import tensorflow as tf

        super().__init__(device=device, device_index=device_index)
        self.model_dir = self._ensure_model_dir(cache_dir=cache_dir)
        with tf.device(self.tensorflow_device_name):
            self.model = tf.saved_model.load(export_dir=str(self.model_dir))
            self._forward = self.model.signatures["serving_default"]

    @classmethod
    def _resolve_model_dir(cls, cache_dir: str | Path | None) -> Path:
        root = (
            Path(cache_dir).expanduser()
            if cache_dir is not None
            else _default_cache_dir()
        )
        return root / "alignnet" / cls.model_name

    @classmethod
    def _ensure_model_dir(cls, cache_dir: str | Path | None) -> Path:
        model_dir = cls._resolve_model_dir(cache_dir=cache_dir)
        if (model_dir / "saved_model.pb").exists():
            return model_dir

        model_dir.parent.mkdir(parents=True, exist_ok=True)
        archive_name = Path(urllib.parse.urlparse(cls.weights_url).path).name
        archive_path = model_dir.parent / archive_name

        if not archive_path.exists():
            cls._download_file(url=cls.weights_url, destination=archive_path)

        temp_root = Path(
            tempfile.mkdtemp(prefix=f"{cls.model_name}-", dir=model_dir.parent)
        )
        extract_root = temp_root / "extract"
        extract_root.mkdir()
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(path=extract_root, filter="data")

            extracted_model_dir = cls._find_saved_model_dir(search_root=extract_root)
            if model_dir.exists():
                return model_dir

            shutil.move(str(extracted_model_dir), str(model_dir))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        if not (model_dir / "saved_model.pb").exists():
            raise ValueError(
                f"Downloaded AlignNet weights for {cls.model_name} but could not find saved_model.pb in {model_dir}"
            )

        return model_dir

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        """Download a file to disk atomically."""
        temp_destination = destination.with_name(f"{destination.name}.tmp")
        try:
            with (
                urllib.request.urlopen(url) as response,
                temp_destination.open("wb") as file,
            ):
                total_bytes_header = response.headers.get("Content-Length")
                total_bytes = (
                    int(total_bytes_header) if total_bytes_header is not None else None
                )
                with tqdm(
                    total=total_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {destination.name}",
                ) as progress:
                    while True:
                        chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress.update(len(chunk))
            temp_destination.replace(destination)
        finally:
            temp_destination.unlink(missing_ok=True)

    @classmethod
    def _find_saved_model_dir(cls, search_root: Path) -> Path:
        """Locate the extracted SavedModel directory inside an extracted archive."""
        named_dir = search_root / cls.model_name
        if (named_dir / "saved_model.pb").exists():
            return named_dir

        candidates = {
            path.parent
            for path in search_root.rglob("saved_model.pb")
            if path.is_file()
        }
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one extracted SavedModel for {cls.model_name}, found {len(candidates)}"
            )
        return next(iter(candidates))

    def compute_features(
        self,
        images: list[PIL.Image.Image],
        seed: int | None = None,
    ) -> np.ndarray:
        """Compute AlignNet features as a NumPy array."""
        del seed
        self.validate_images(images)
        import tensorflow as tf

        batch = np.stack([self._preprocess_image(image) for image in images], axis=0)
        with tf.device(self.tensorflow_device_name):
            outputs = self._forward(images=tf.convert_to_tensor(batch))
        return self._select_features(outputs=outputs)

    def _select_features(self, outputs: Any) -> np.ndarray:
        """Select the feature tensor from the SavedModel outputs."""
        if isinstance(outputs, Mapping):
            if self.output_key is not None:
                tensor = outputs[self.output_key]
            elif len(outputs) == 1:
                tensor = next(iter(outputs.values()))
            else:
                raise ValueError(
                    f"Expected exactly one AlignNet output tensor, but got keys {sorted(outputs.keys())}"
                )
        else:
            tensor = outputs

        features = np.asarray(tensor)
        if features.shape[0] == 0:
            raise ValueError(
                "Expected AlignNet to return a non-empty batch of features."
            )
        return features

    @staticmethod
    def _preprocess_image(image: PIL.Image.Image) -> np.ndarray:
        """Convert a PIL image to a float32 BHWC-ready tensor."""
        image = image.convert("RGB")
        width, height = image.size
        scale = _MODEL_INPUT_SIZE / min(width, height)
        resized = image.resize(
            size=(round(width * scale), round(height * scale)),
            resample=PIL.Image.Resampling.BILINEAR,
        )

        left = (resized.width - _MODEL_INPUT_SIZE) // 2
        top = (resized.height - _MODEL_INPUT_SIZE) // 2
        cropped = resized.crop(
            (
                left,
                top,
                left + _MODEL_INPUT_SIZE,
                top + _MODEL_INPUT_SIZE,
            )
        )
        return np.asarray(cropped, dtype=np.float32) / 255.0


class AligNetViTB16(_AlignNet):
    """AlignNet-aligned ViT-B/16 encoder."""

    model_name = "ViT-B-alignet"
    weights_url = "https://storage.googleapis.com/alignet/models/ViT-B-alignet.tar.gz"


class UnaligNetViTB16(_AlignNet):
    """Unaligned ViT-B/16 encoder from the AlignNet release."""

    model_name = "ViT-B-untransformed"
    weights_url = (
        "https://storage.googleapis.com/alignet/models/ViT-B-untransformed.tar.gz"
    )

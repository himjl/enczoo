from abc import ABC
from typing import Any

from enczoo.base import DeviceType, ImageEncoding


class TensorflowImageEncoding(ImageEncoding, ABC):
    """TensorFlow-backed image encoder with shared device resolution."""

    def __init__(
        self,
        device: DeviceType = "cpu",
        device_index: int | None = None,
    ):
        """Initialize backend-neutral state and resolve the TensorFlow device."""
        super().__init__(device=device, device_index=device_index)
        import tensorflow as tf

        self._tensorflow_device_name = self._resolve_tensorflow_device_name(tf)

    @property
    def tensorflow_device_name(self) -> str:
        """Return the resolved TensorFlow device name for this encoder."""
        return self._tensorflow_device_name

    def _resolve_tensorflow_device_name(self, tf: Any) -> str:
        """Resolve the backend-neutral device to a concrete TensorFlow device."""
        if self.device == "cpu":
            logical_cpus = tf.config.list_logical_devices("CPU")
            if logical_cpus:
                return logical_cpus[0].name
            return "/CPU:0"

        logical_gpus = tf.config.list_logical_devices("GPU")
        if not logical_gpus:
            raise ValueError(
                "device='gpu' was requested, but TensorFlow could not find an available GPU."
            )

        gpu_index = 0 if self.device_index is None else self.device_index
        if gpu_index >= len(logical_gpus):
            raise ValueError(
                f"Requested TensorFlow GPU index {gpu_index}, but only {len(logical_gpus)} logical GPU device(s) are available."
            )
        return logical_gpus[gpu_index].name

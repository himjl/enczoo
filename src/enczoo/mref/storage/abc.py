from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import PIL.Image

from enczoo.mref.media_references import GzipRef, ImageRef, JsonRef, MediaRef, ZipRef


# %%
class Storage(ABC):
    @abstractmethod
    def check_data_exists(self, ref: MediaRef) -> bool:
        raise NotImplementedError

    # %% PIL.Image.Image
    @abstractmethod
    def load_image(self, ref: ImageRef) -> PIL.Image.Image:
        raise NotImplementedError

    @abstractmethod
    def download_image_from_url(self, url: str, register: bool) -> PIL.Image.Image:
        raise NotImplementedError

    @abstractmethod
    def register_image(self, image: PIL.Image.Image) -> ImageRef:
        raise NotImplementedError

    # %% JSON
    @abstractmethod
    def load_json(self, ref: JsonRef) -> Any:
        raise NotImplementedError

    @abstractmethod
    def download_json_from_url(self, url: str, register: bool) -> Any:
        raise NotImplementedError

    @abstractmethod
    def register_json(self, obj: Any) -> JsonRef:
        raise NotImplementedError

    # %% Zip files
    @abstractmethod
    def load_zip_path(self, ref: ZipRef) -> Path:
        raise NotImplementedError

    @abstractmethod
    def download_zip_path_from_url(self, url: str, register: bool) -> Path:
        raise NotImplementedError

    @abstractmethod
    def register_zip_path(self, zipfile_path: Path) -> ZipRef:
        raise NotImplementedError

    # %% GZip files
    @abstractmethod
    def load_gzip_path(self, ref: GzipRef) -> Path:
        raise NotImplementedError

    @abstractmethod
    def download_gzip_path_from_url(self, url: str, register: bool) -> Path:
        raise NotImplementedError

    @abstractmethod
    def register_gzip_path(self, gzipfile_path: Path) -> GzipRef:
        raise NotImplementedError

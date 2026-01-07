from pathlib import Path
from typing import Literal

import PIL.Image
import pydantic

from typing import Any
import mref.media_references.hash_functions as hashes


# %%
class MediaRef(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(frozen=True)

    sha256: str = pydantic.Field(pattern=r"^[a-f0-9]{64}$")
    mime_type: Literal[
        "image/png", "application/json", "application/zip", "application/gzip"
    ] = pydantic.Field(description="The MIME type of the media file.")

    def __lt__(self, other) -> bool:
        return self.sha256 < other.sha256

    def __hash__(self):
        return hash(self.sha256)


# %%
class ImageRef(MediaRef):
    """
    A reference to an image based on its RGBA uint8 representation. Images of other formats are cast to RGBA before being hashed.
    """

    mime_type: Literal["image/png"] = "image/png"

    @classmethod
    def from_image(cls, image: PIL.Image) -> "ImageRef":
        sha256 = hashes.hash_image(image=image)
        return cls(
            sha256=sha256,
        )


class JsonRef(MediaRef):
    """
    A reference to a JSON file.
    """

    mime_type: Literal["application/json"] = "application/json"

    @classmethod
    def from_obj(cls, obj: Any) -> "JsonRef":
        sha256 = hashes.hash_json(obj=obj)
        return cls(
            sha256=sha256,
        )


class ZipRef(MediaRef):
    """
    A reference to a ZIP file.
    """

    mime_type: Literal["application/zip"] = "application/zip"

    @classmethod
    def from_zipfile(cls, zipfile_path: Path) -> "ZipRef":
        if not zipfile_path.suffix == ".zip":
            raise ValueError(f"File is not a ZIP file: {zipfile_path}")

        sha256 = hashes.hash_file(path=zipfile_path)
        return cls(
            sha256=sha256,
            mime_type="application/zip",
        )


class GzipRef(MediaRef):
    """
    A reference to a GZIP file.
    """

    mime_type: Literal["application/gzip"] = "application/gzip"

    @classmethod
    def from_gzipfile(cls, gzipfile_path: Path) -> "GzipRef":
        if not gzipfile_path.suffix == ".gz":
            raise ValueError(f"File is not a GZIP file: {gzipfile_path}")

        sha256 = hashes.hash_file(path=gzipfile_path)
        return cls(
            sha256=sha256,
            mime_type="application/gzip",
        )

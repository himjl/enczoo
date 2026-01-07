import json
import shutil
import tempfile
from pathlib import Path
from typing import Set, Any, Callable

import PIL.Image

import mref.media_references.hash_functions as hashing
import mref.utils as utils
from mref.media_references import MediaRef, ImageRef, JsonRef, ZipRef, GzipRef
from mref.storage.exceptions import NotInStorageError, UnsupportedUrlMimeTypeError

from mref.storage.abc import Storage
import pydantic
from typing import Literal
import datetime


# %% Storage class
class FileSystemStorage(Storage):

    def __init__(self, cachedir: Path):
        self.cachedir = cachedir
        self._ref_manifest: Set[MediaRef] = set()

    # %% Images
    def load_image(self, ref: ImageRef) -> PIL.Image:
        loaded_image = self._load_core(
            ref=ref,
            load_func=utils.load_image
        )
        return loaded_image

    def download_image_from_url(self, url: str, register: bool) -> PIL.Image:

        def register_callback(path: Path):
            image = utils.load_image(path)
            # Cache the image
            ref = self.register_image(image=image)
            return ref

        ref = self._load_from_url_core(
            url=url,
            register_callback=register_callback,
        )

        # Cast to ImageRef
        ref = ImageRef(**ref.model_dump())

        image = self.load_image(ref=ref)
        return image

    def register_image(self, image: PIL.Image) -> ImageRef:
        ref = ImageRef.from_image(image=image)
        self._save_core(
            ref=ref,
            save_func=image.save,
            overwrite=False,
        )
        return ref

    # %% JSON
    def load_json(self, ref: JsonRef) -> Any:
        def _load_func(path: Path) -> Any:
            return json.loads(path.read_text())

        return self._load_core(
            ref=ref,
            load_func=_load_func,
        )

    def download_json_from_url(self, url: str, register: bool) -> Any:

        def register_callback(path: Path):
            json_string = path.read_text()
            obj = json.loads(json_string)
            ref = self.register_json(obj=obj)
            return ref

        ref = self._load_from_url_core(
            url=url,
            register_callback=register_callback,
        )

        # Cast to JsonRef
        ref = JsonRef(**ref.model_dump())

        json_object = self.load_json(ref=ref)
        return json_object

    def register_json(self, obj: Any) -> JsonRef:
        ref = JsonRef.from_obj(obj=obj)
        self._save_core(
            ref=ref,
            save_func=lambda path: path.write_text(json.dumps(obj=obj)),
            overwrite=False,
        )
        return ref

    # %% Zip
    def load_zip_path(self, ref: ZipRef) -> Path:
        return self._load_core(
            ref=ref,
            load_func=lambda path: path
        )

    def download_zip_path_from_url(self, url: str, register: bool) -> Path:

        def register_callback(path: Path):
            ref = self.register_zip_path(zipfile_path=path)
            return ref

        ref = self._load_from_url_core(
            url=url,
            register_callback=register_callback,
        )

        # Cast to ZipRef
        ref = ZipRef(**ref.model_dump())

        return self.load_zip_path(ref=ref)

    def register_zip_path(self, zipfile_path: Path) -> ZipRef:
        ref = ZipRef.from_zipfile(zipfile_path=zipfile_path)

        def save_func(path: Path):
            # Copy the file to the cache
            shutil.copy(src=zipfile_path, dst=path)

        self._save_core(
            ref=ref,
            save_func=save_func,
            overwrite=False,
        )
        return ref

    # %% GZip
    def load_gzip_path(self, ref: GzipRef) -> Path:
        return self._load_core(
            ref=ref,
            load_func=lambda path: path
        )

    def download_gzip_path_from_url(self, url: str, register: bool) -> Path:

        def register_callback(path: Path):
            ref = self.register_gzip_path(gzipfile_path=path)
            return ref

        ref = self._load_from_url_core(
            url=url,
            register_callback=register_callback,
        )

        # Cast to ZipRef
        ref = GzipRef(**ref.model_dump())

        return self.load_gzip_path(ref=ref)

    def register_gzip_path(self, gzipfile_path: Path) -> GzipRef:
        ref = GzipRef.from_gzipfile(gzipfile_path=gzipfile_path)

        def save_func(path: Path):
            # Copy the file to the cache
            shutil.copy(src=gzipfile_path, dst=path)

        self._save_core(
            ref=ref,
            save_func=save_func,
            overwrite=False,
        )
        return ref

    # %% Common
    def _get_cache_path(self, ref: MediaRef) -> Path:
        extension = utils.get_extension_from_mime_type(mime_type=ref.mime_type)
        filename = ref.sha256 + f'{extension}'
        return self.cachedir / 'media' / ref.mime_type / filename

    def _get_url_ref_path(self, url: str) -> Path:
        url_hash = hashing.hash_url(url=url)
        return self.cachedir / 'urls' / (url_hash + '.json')

    def check_data_exists(self, ref: MediaRef) -> bool:
        if ref not in self._ref_manifest:
            path = self._get_cache_path(ref=ref)
            exists = path.exists()
            if exists:
                self._ref_manifest.add(ref)
        else:
            exists = True

        return exists

    def _save_core(
            self,
            ref: MediaRef,
            save_func: Callable[[Path], None],
            overwrite: bool
    ) -> None:
        path = self._get_cache_path(ref=ref)
        if path.exists() and not overwrite:
            self._ref_manifest.add(ref)
            return

        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        save_func(path)
        print(f'Saved {ref.mime_type} to', path)
        self._ref_manifest.add(ref)

    def _load_core(self, ref: MediaRef, load_func: Callable[[Path], Any], ) -> Any:
        path = self._get_cache_path(ref=ref)
        if not path.exists():
            raise NotInStorageError(ref=ref, path=path)

        data = load_func(path)

        # Add the ref to the manifest
        self._ref_manifest.add(ref)
        return data

    class _UrlMediaRefAssociation(pydantic.BaseModel):
        url: str
        ref: MediaRef
        date_accessed: float = pydantic.Field(default_factory=lambda: datetime.datetime.now().timestamp())

    def _load_from_url_core(
            self,
            url: str,
            register_callback: Callable[[Path], MediaRef],
    ) -> MediaRef:
        """
        Downloads data from a URL and passes it to a callback function.
        If data from the URL has already been cached, the path to the cached data is passed.
        :param url:
        :param register_callback: Expected to register the data at Path, and return its corresponding MediaRef.
        :return:
        """
        # First, check if we have a MediaRef for this URL cached:
        url_ref_path = self._get_url_ref_path(url=url)
        if url_ref_path.exists():
            ref_string = url_ref_path.read_text()
            association = self._UrlMediaRefAssociation.model_validate_json(json_data=ref_string)

            # Return the MediaRef immediately if its backing data is in the cache
            if self.check_data_exists(ref=association.ref):
                return association.ref

        # Download the URL data to a temporary directory:
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            basename = Path(url).name
            temp_savepath = tempdir / basename
            utils.download_file(url=url, output_path=temp_savepath)

            # Register the data
            ref: MediaRef = register_callback(temp_savepath)
            association = self._UrlMediaRefAssociation(url=url, ref=ref)

            # Cache the URL reference
            if not url_ref_path.parent.exists():
                url_ref_path.parent.mkdir(parents=True)

            url_ref_path.write_text(association.model_dump_json())

        return ref

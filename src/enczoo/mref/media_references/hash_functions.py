import hashlib
import json

import PIL.Image
import numpy as np
from typing import Any

from pathlib import Path


def hash_image(image: PIL.Image) -> str:
    """
    Hash an image based on its np.uint8 RGBA representation.
    :param image:
    :return:
    """
    sha256_hash = hashlib.sha256()

    # Always cast the image to RGBA format
    image = image.convert('RGBA')

    # Convert the image to a np.ndarray
    image_array = np.array(image, dtype=np.uint8)

    # Update the hash with the image array
    sha256_hash.update(image_array.tobytes(order='C'))

    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()


def hash_json(obj: Any) -> str:
    """
    Hash a JSON string. Invariant to whitespace and key-ordering.
    :param json_data:
    :return:
    """
    sha256_hash = hashlib.sha256()
    json_data = json.dumps(obj, sort_keys=True, indent=0)
    sha256_hash.update(json_data.encode(encoding='utf-8'))
    return sha256_hash.hexdigest()


def hash_url(url: str) -> str:
    """
    Hash a string.
    :param url:
    :return:
    """

    sha256_hash = hashlib.sha256()

    # Convert string to bytes
    sha256_hash.update(url.encode('utf-8'))

    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()


def hash_file(path: Path):
    """
    Hash a file.
    :param path:
    :return:
    """

    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')
    if not path.is_file():
        raise IsADirectoryError(f'Path is not a file: {path}')

    sha256_hash = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()
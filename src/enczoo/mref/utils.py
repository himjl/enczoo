import mimetypes
from pathlib import Path

import PIL.Image
import requests  # type: ignore[unresolved-import]
from tqdm import tqdm


# %% Image
def load_image(path: Path) -> PIL.Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")

    with PIL.Image.open(path) as img:
        loaded_image = img.copy()

    return loaded_image


# %% mime types
def infer_mime_type(url: str) -> str:
    mime_type, _ = mimetypes.guess_type(url, strict=True)

    if mime_type:
        return mime_type

        # If guessing fails, try fetching the Content-Type header
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get("Content-Type")

        if content_type:
            return content_type.split(";")[0]  # Remove charset if present
    except requests.RequestException:
        pass

        # If HEAD fails, try a minimal GET request
    try:
        response = requests.get(url, stream=True)
        content_type = response.headers.get("Content-Type")

        if content_type:
            return content_type.split(";")[0]
    except requests.RequestException:
        pass

    raise Exception(f"Could not infer the MIME type of {url}")


def get_extension_from_mime_type(mime_type: str) -> str:
    """
    Returns the file extension (including the leading dot) for a given MIME type.
    :param mime_type:
    :return:
    """

    if mime_type == "application/gzip":
        # Not supported by mimetypes.guess_extension
        return ".gz"

    result = mimetypes.guess_extension(mime_type, strict=True)

    if result is None:
        raise ValueError(f"Could not infer extension for MIME type {mime_type}")

    return result


# %% Downloading
def download_file(url: str, output_path: Path) -> None:
    # Send a GET request to fetch the file
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    disable_pbar = total_size_in_bytes < 10000

    # Create the output directory if it does not exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    with tqdm(
        total=total_size_in_bytes,
        disable=disable_pbar,
        unit="B",
        unit_scale=True,
        desc="Download progress:",
    ) as pbar:
        with open(output_path.as_posix(), "wb") as file:
            # Iterate over the response data in chunks and write to file
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))

from pathlib import Path

from enczoo.mref.media_references import MediaRef


class NotInStorageError(Exception):
    def __init__(self, ref: MediaRef, path: Path):
        self.ref = ref
        super().__init__(
            f"{ref.mime_type} ref ({ref.sha256[:8]}...) not found in storage at {path}"
        )


class UnsupportedUrlMimeTypeError(Exception):
    def __init__(self, url: str, mime_type: str):
        super().__init__(f"URL {url} has unsupported MIME type {mime_type}")

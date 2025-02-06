import pydantic
from pathlib import Path


class ImageEncodingConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True,
    )

    frozen: bool = pydantic.Field(
        default=True,
        description='If True, the model is frozen and does not aggregate gradients. Caching is disabled if frozen=False.'
    )

    cachedir: Path = pydantic.Field(
        description='The directory where the cache is stored. Note if frozen=False, the cache is disabled.'
    )
    in_memory_cache_size_mb: int = pydantic.Field(
        default=256,
        description='The size of the in-memory cache in MB. Irrelevant if frozen=False.'
    )

    default_batch_size: int = pydantic.Field(
        default=32,
        description='The default batch size to use in the load_features (but not compute_features) method.'
    )


default_config = ImageEncodingConfig(
    frozen=True,
    cachedir=Path.home() / 'enczoo_cache',
    in_memory_cache_size_mb=256,
    default_batch_size=32
)

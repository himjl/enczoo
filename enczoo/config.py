import pydantic
from pathlib import Path


class ImageEncodingConfig(pydantic.BaseModel):

    # Pydantic configuration:
    model_config = pydantic.ConfigDict(
        frozen=True,
    )

    # Options:
    trainable: bool = pydantic.Field(
        default=False,
        description='If False, the model is considered fixed and does not aggregate gradients. Caching is disabled if trainable=True.'
    )

    cachedir: Path = pydantic.Field(
        default=Path.home() / 'enczoo_cache',
        description='The directory where the cache is stored. Note if trainable=True, the cache is disabled.'
    )
    in_memory_cache_size_mb: int = pydantic.Field(
        default=256,
        description='The size of the in-memory cache in MB. Irrelevant if trainable=True.'
    )

    default_batch_size: int = pydantic.Field(
        default=32,
        description='The default batch size to use in the load_features (but not compute_features) method.'
    )


default_config = ImageEncodingConfig()

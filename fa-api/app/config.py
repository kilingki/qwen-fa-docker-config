import os
from dataclasses import dataclass
from typing import Mapping

ALLOWED_DTYPES = ("bfloat16", "float16", "float32")
MAX_SUPPORTED_CHUNK_SECONDS = 180

_DEFAULT_MODEL_PATH = "/models/Qwen3-ForcedAligner-0.6B"
_DEFAULT_DTYPE = "bfloat16"
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_MAX_CHUNK_SECONDS = 180


@dataclass(frozen=True)
class Settings:
    model_path: str
    dtype: str
    batch_size: int
    max_chunk_seconds: int

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise RuntimeError("FA_MODEL_PATH is required")
        if self.dtype not in ALLOWED_DTYPES:
            allowed = ", ".join(ALLOWED_DTYPES)
            raise RuntimeError(f"FA_DTYPE must be one of: {allowed}")
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise RuntimeError("FA_BATCH_SIZE must be a positive integer")
        if self.batch_size < 1:
            raise RuntimeError("FA_BATCH_SIZE must be a positive integer")
        if (
            not isinstance(self.max_chunk_seconds, int)
            or isinstance(self.max_chunk_seconds, bool)
            or not 1 <= self.max_chunk_seconds <= MAX_SUPPORTED_CHUNK_SECONDS
        ):
            raise RuntimeError(
                "FA_MAX_CHUNK_SECONDS must be an integer from 1 to "
                f"{MAX_SUPPORTED_CHUNK_SECONDS}"
            )

    @property
    def max_chunk_samples(self) -> int:
        return self.max_chunk_seconds * 16000


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    source = os.environ if env is None else env
    return Settings(
        model_path=_text(source, "FA_MODEL_PATH", _DEFAULT_MODEL_PATH),
        dtype=_text(source, "FA_DTYPE", _DEFAULT_DTYPE),
        batch_size=_positive_int(source, "FA_BATCH_SIZE", _DEFAULT_BATCH_SIZE),
        max_chunk_seconds=_positive_int(
            source,
            "FA_MAX_CHUNK_SECONDS",
            _DEFAULT_MAX_CHUNK_SECONDS,
            upper=MAX_SUPPORTED_CHUNK_SECONDS,
        ),
    )


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _positive_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    upper: int | None = None,
) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        raise RuntimeError(f"{name} is required")
    if not text.isascii() or not text.isdigit():
        raise RuntimeError(f"{name} must be a positive integer")
    value = int(text)
    if value < 1 or (upper is not None and value > upper):
        if upper is None:
            raise RuntimeError(f"{name} must be a positive integer")
        raise RuntimeError(f"{name} must be an integer from 1 to {upper}")
    return value

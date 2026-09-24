from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, ValidationError, BeforeValidator

SAMPLE_RATE = 16000


class ClientError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class AlignmentError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _strict_int(value: Any) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError("must be an integer")
    return value


StrictInt = Annotated[int, BeforeValidator(_strict_int)]


class AudioInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sample_rate: StrictInt
    channels: StrictInt
    num_samples: StrictInt


class ChunkInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: StrictInt
    start_sample: StrictInt
    end_sample: StrictInt
    text: str
    language: str | None = None


class Payload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    audio: AudioInput
    chunks: list[ChunkInput]


def parse_payload(data: Any) -> Payload:
    if not isinstance(data, dict):
        raise ClientError("payload must be a JSON object")
    if "audio" not in data or "chunks" not in data:
        raise ClientError("payload must include audio and chunks")
    try:
        return Payload.model_validate(data)
    except ValidationError as exc:
        raise ClientError(_format_validation(exc)) from exc


def _format_validation(exc: ValidationError) -> str:
    messages: list[str] = []
    for err in exc.errors():
        loc = err.get("loc", ())
        msg = err.get("msg", "invalid value")
        if loc and loc[0] == "chunks" and len(loc) >= 2 and isinstance(loc[1], int):
            field = ".".join(str(part) for part in loc[2:]) or "chunk"
            messages.append(f"chunks[{loc[1]}].{field}: {msg}")
            continue
        place = ".".join(str(part) for part in loc) or "payload"
        messages.append(f"{place}: {msg}")
    return "; ".join(messages) if messages else "invalid payload"

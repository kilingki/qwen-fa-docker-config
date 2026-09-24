import math
from typing import Any

import numpy as np

from app.audio import read_canonical_wav, slice_waveform, validate_chunk_span
from app.config import Settings
from app.schemas import SAMPLE_RATE, AlignmentError, ChunkInput, ClientError, Payload, parse_payload

CANONICAL_LANGUAGES = (
    "Korean",
    "Japanese",
    "Chinese",
    "English",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Spanish",
)

_ALIASES = {
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spanish": "Spanish",
}

def canonical_language(value: str) -> str:
    if not isinstance(value, str):
        raise ClientError("language must be a string")
    key = value.strip().lower()
    try:
        return _ALIASES[key]
    except KeyError:
        raise ClientError(f"unsupported language: {value}") from None


def verify_model_languages(names: list[str] | None) -> None:
    if names is None:
        raise RuntimeError("model support_languages is unavailable")
    found = {str(name).strip().lower() for name in names}
    expected = {name.lower() for name in CANONICAL_LANGUAGES}
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(
            "model support_languages mismatch; "
            f"missing={missing}; extra={extra}"
        )


class ForcedAlignerService:
    def __init__(self, settings: Settings, aligner: Any | None = None) -> None:
        self.settings = settings
        self._aligner = aligner
        self.loaded = False

    def load(self) -> None:
        aligner = self._load_aligner()
        verify_model_languages(aligner.get_supported_languages())
        self._aligner = aligner
        self.loaded = True

    def release(self) -> None:
        self._aligner = None
        self.loaded = False

    def holds_model(self) -> bool:
        return self._aligner is not None

    def validate_request(self, wav_path: str, payload: Any) -> None:
        parsed = payload if isinstance(payload, Payload) else parse_payload(payload)
        waveform = read_canonical_wav(wav_path)
        self._check_audio_metadata(parsed, waveform)
        self._prepare_jobs(parsed)

    def _load_aligner(self) -> Any:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.settings.dtype]
        return Qwen3ForcedAligner.from_pretrained(
            self.settings.model_path,
            dtype=dtype,
            device_map="cuda:0",
            local_files_only=True,
        )

    def align_chunks(self, wav_path: str, payload: Any) -> dict:
        parsed = payload if isinstance(payload, Payload) else parse_payload(payload)
        waveform = read_canonical_wav(wav_path)
        self._check_audio_metadata(parsed, waveform)
        jobs = self._prepare_jobs(parsed)
        aligned: dict[int, dict] = {}
        batch_size = self.settings.batch_size
        for offset in range(0, len(jobs), batch_size):
            group = jobs[offset : offset + batch_size]
            waveforms = [
                slice_waveform(waveform, chunk.start_sample, chunk.end_sample)
                for _position, chunk, _language in group
            ]
            texts = [chunk.text for _position, chunk, _language in group]
            languages = [language for _position, _chunk, language in group]
            try:
                results = self._aligner.align(
                    audio=[(chunk, SAMPLE_RATE) for chunk in waveforms],
                    text=texts,
                    language=languages,
                )
            except AlignmentError:
                raise
            except Exception as exc:
                raise AlignmentError(f"alignment failed: {exc}") from exc
            if not isinstance(results, list) or len(results) != len(group):
                got = len(results) if isinstance(results, list) else type(results).__name__
                raise AlignmentError(
                    f"alignment result count mismatch: expected {len(group)}, got {got}"
                )
            for (position, chunk, language), result in zip(group, results):
                aligned[position] = self._chunk_result(chunk, language, result)
            del group, waveforms, texts, languages, results

        chunks = []
        for position, chunk in enumerate(parsed.chunks):
            if chunk.text.strip() == "":
                chunks.append(self._skipped_chunk(chunk))
            else:
                chunks.append(aligned[position])
        return {
            "audio": {
                "sample_rate": parsed.audio.sample_rate,
                "channels": parsed.audio.channels,
                "num_samples": parsed.audio.num_samples,
            },
            "time_reference": "audio_start",
            "overlap_deduplicated": False,
            "chunks": chunks,
        }

    def _check_audio_metadata(self, parsed: Payload, waveform: np.ndarray) -> None:
        audio = parsed.audio
        if audio.sample_rate != SAMPLE_RATE or audio.channels != 1:
            raise ClientError("audio metadata must be 16000 Hz mono")
        if audio.num_samples != int(waveform.shape[0]):
            raise ClientError(
                "audio.num_samples does not match wav frame count: "
                f"payload={audio.num_samples}, wav={int(waveform.shape[0])}"
            )

    def _prepare_jobs(self, parsed: Payload) -> list[tuple[int, ChunkInput, str]]:
        seen: set[int] = set()
        jobs: list[tuple[int, ChunkInput, str]] = []
        num_samples = parsed.audio.num_samples
        max_chunk_samples = self.settings.max_chunk_samples
        for position, chunk in enumerate(parsed.chunks):
            if chunk.index < 0:
                raise ClientError(f"chunk index {chunk.index}: index must be >= 0")
            if chunk.index in seen:
                raise ClientError(f"chunk index {chunk.index}: duplicate index")
            seen.add(chunk.index)
            validate_chunk_span(
                chunk.index,
                chunk.start_sample,
                chunk.end_sample,
                num_samples,
                max_chunk_samples,
            )
            if chunk.text.strip() == "":
                continue
            if chunk.language is None or chunk.language.strip() == "":
                raise ClientError(
                    f"chunk index {chunk.index}: language is required for non-empty text"
                )
            try:
                language = canonical_language(chunk.language)
            except ClientError as exc:
                raise ClientError(f"chunk index {chunk.index}: {exc.detail}") from exc
            jobs.append((position, chunk, language))
        return jobs

    def _skipped_chunk(self, chunk: ChunkInput) -> dict:
        return {
            "index": chunk.index,
            "start_sample": chunk.start_sample,
            "end_sample": chunk.end_sample,
            "text": chunk.text,
            "language": None,
            "status": "skipped_empty_text",
            "items": [],
        }

    def _chunk_result(self, chunk: ChunkInput, language: str, result: Any) -> dict:
        if isinstance(result, dict):
            raw_items = result.get("items")
        else:
            raw_items = getattr(result, "items", None)
        if not isinstance(raw_items, list):
            raise AlignmentError(
                f"chunk index {chunk.index}: alignment result is missing items"
            )
        duration = (chunk.end_sample - chunk.start_sample) / SAMPLE_RATE
        offset = chunk.start_sample / SAMPLE_RATE
        items = []
        for raw in raw_items:
            text, start_time, end_time = _read_raw_item(chunk.index, raw)
            if text == "":
                continue
            start_time, end_time = _clamp_local_times(chunk.index, start_time, end_time, duration)
            items.append(
                {
                    "text": text,
                    "start_time": start_time + offset,
                    "end_time": end_time + offset,
                }
            )
        return {
            "index": chunk.index,
            "start_sample": chunk.start_sample,
            "end_sample": chunk.end_sample,
            "text": chunk.text,
            "language": language,
            "status": "aligned",
            "items": items,
        }


def _read_raw_item(index: int, raw: Any) -> tuple[str, float, float]:
    if isinstance(raw, dict):
        text = raw.get("text")
        start = raw.get("start_time")
        end = raw.get("end_time")
    else:
        text = getattr(raw, "text", None)
        start = getattr(raw, "start_time", None)
        end = getattr(raw, "end_time", None)
    if not isinstance(text, str):
        raise AlignmentError(f"chunk index {index}: invalid alignment item")
    return text, _finite_time(index, start), _finite_time(index, end)


def _finite_time(index: int, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise AlignmentError(f"chunk index {index}: invalid timestamp")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise AlignmentError(f"chunk index {index}: invalid timestamp")
    number = float(value)
    if not math.isfinite(number):
        raise AlignmentError(f"chunk index {index}: invalid timestamp")
    return number


def _clamp_local_times(
    index: int,
    start_time: float,
    end_time: float,
    duration: float,
) -> tuple[float, float]:
    start_time = _clamp_bound(start_time, duration)
    end_time = _clamp_bound(end_time, duration)
    if start_time > end_time:
        raise AlignmentError(f"chunk index {index}: timestamp reversed")
    return start_time, end_time


def _clamp_bound(value: float, duration: float) -> float:
    if value < 0:
        return 0.0
    if value > duration:
        return duration
    return value

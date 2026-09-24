import numpy as np
import soundfile as sf

from app.schemas import SAMPLE_RATE, ClientError


def read_canonical_wav(path: str) -> np.ndarray:
    try:
        info = sf.info(path)
    except Exception as exc:
        raise ClientError("invalid or truncated wav") from exc

    if info.format != "WAV" or info.subtype != "PCM_16":
        raise ClientError("wav must be signed 16-bit PCM")
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ClientError("wav must be 16000 Hz mono")
    if info.frames <= 0:
        raise ClientError("wav must contain at least one frame")

    try:
        data, samplerate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise ClientError("invalid or truncated wav") from exc

    waveform = np.asarray(data)
    if samplerate != SAMPLE_RATE or waveform.ndim != 1 or waveform.shape[0] != info.frames:
        raise ClientError("wav header does not match readable PCM frames")
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32, copy=False)
    return waveform


def validate_chunk_span(
    index: int,
    start_sample: int,
    end_sample: int,
    num_samples: int,
    max_chunk_samples: int,
) -> None:
    if index < 0:
        raise ClientError(f"chunk index {index}: index must be >= 0")
    if not (0 <= start_sample < end_sample <= num_samples):
        raise ClientError(
            f"chunk index {index}: sample range "
            f"[{start_sample}, {end_sample}) is outside [0, {num_samples})"
        )
    length = end_sample - start_sample
    if length > max_chunk_samples:
        raise ClientError(
            f"chunk index {index}: chunk length {length} samples exceeds "
            f"{max_chunk_samples}"
        )


def slice_waveform(waveform: np.ndarray, start_sample: int, end_sample: int) -> np.ndarray:
    return waveform[start_sample:end_sample]

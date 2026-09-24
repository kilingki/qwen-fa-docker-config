import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fa-api"))

from app.aligner import ForcedAlignerService
from app.config import Settings


def settings(**overrides) -> Settings:
    values = {
        "model_path": "/models/Qwen3-ForcedAligner-0.6B",
        "dtype": "bfloat16",
        "batch_size": 4,
        "max_chunk_seconds": 180,
    }
    values.update(overrides)
    return Settings(**values)


class Span:
    def __init__(self, text="가", start_time=0.0, end_time=0.0):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class AlignResult:
    def __init__(self, items):
        self.items = items


class FakeModel:
    def __init__(self, fn=None):
        self.calls = []
        self.fn = fn

    def align(self, audio, text, language):
        self.calls.append(
            {
                "lengths": [int(np.asarray(item[0]).shape[0]) for item in audio],
                "rates": [item[1] for item in audio],
                "arrays": [np.asarray(item[0]) for item in audio],
                "text": list(text),
                "language": list(language),
            }
        )
        if self.fn is not None:
            return self.fn(audio, text, language)
        return [AlignResult([Span(text=item)]) for item in text]


def make_service(model=None, **overrides) -> tuple[ForcedAlignerService, FakeModel]:
    aligner = model or FakeModel()
    return ForcedAlignerService(settings(**overrides), aligner=aligner), aligner

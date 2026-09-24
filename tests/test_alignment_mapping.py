import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.aligner import canonical_language
from app.audio import validate_chunk_span
from app.schemas import SAMPLE_RATE, AlignmentError, ClientError
from conftest import AlignResult, FakeModel, Span, make_service


def write_wav(path: Path, frames: int, sample_rate=SAMPLE_RATE, channels=1, sampwidth=2) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sampwidth)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00" * (frames * channels * sampwidth))


def chunk(index, start, end, text, language="ko"):
    body = {
        "index": index,
        "start_sample": start,
        "end_sample": end,
        "text": text,
    }
    if language is not None:
        body["language"] = language
    return body


def payload(num_samples, chunks, **extra):
    body = {
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "num_samples": num_samples,
        },
        "chunks": chunks,
    }
    body.update(extra)
    return body


def test_language_aliases_passed_as_canonical_names(tmp_path):
    frames = 1600
    path = tmp_path / "canonical.wav"
    write_wav(path, frames)
    aliases = {
        "ko": "Korean",
        " Korean ": "Korean",
        "ja": "Japanese",
        "jp": "Japanese",
        "zh-CN": "Chinese",
        "zh-tw": "Chinese",
        "en": "English",
        "yue": "Cantonese",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "es": "Spanish",
    }
    for alias, canonical in aliases.items():
        assert canonical_language(alias) == canonical
        service, model = make_service()
        service.align_chunks(
            str(path),
            payload(frames, [chunk(0, 0, frames, "텍스트", alias)]),
        )
        assert model.calls[-1]["language"] == [canonical]


def test_empty_text_is_not_sent_and_keeps_its_position(tmp_path):
    frames = 3000
    path = tmp_path / "canonical.wav"
    write_wav(path, frames)
    service, model = make_service(
        FakeModel(
            lambda audio, text, language: [
                AlignResult([Span(text=f"tok-{item}")]) for item in text
            ]
        )
    )
    result = service.align_chunks(
        str(path),
        payload(
            frames,
            [
                chunk(0, 0, 1000, "첫째"),
                chunk(1, 1000, 2000, "  ", language=None),
                chunk(2, 2000, 3000, "", language=""),
                chunk(3, 2000, 3000, "셋째", "en"),
            ],
        ),
    )
    assert model.calls[0]["text"] == ["첫째", "셋째"]
    assert [item["status"] for item in result["chunks"]] == [
        "aligned",
        "skipped_empty_text",
        "skipped_empty_text",
        "aligned",
    ]
    assert result["chunks"][1]["text"] == "  "
    assert result["chunks"][1]["language"] is None
    assert result["chunks"][1]["items"] == []
    assert result["chunks"][0]["items"][0]["text"] == "tok-첫째"
    assert result["chunks"][3]["items"][0]["text"] == "tok-셋째"
    assert result["chunks"][3]["language"] == "English"
    assert result["chunks"][0]["text"] == "첫째"


def test_unsupported_or_missing_language_rejects_non_empty_text(tmp_path):
    frames = 800
    path = tmp_path / "canonical.wav"
    write_wav(path, frames)
    service, model = make_service()
    with pytest.raises(ClientError, match="chunk index 4: unsupported language"):
        service.align_chunks(
            str(path),
            payload(frames, [chunk(4, 0, frames, "말", "ar")]),
        )
    with pytest.raises(ClientError, match="chunk index 0: language is required"):
        service.align_chunks(
            str(path),
            payload(frames, [chunk(0, 0, frames, "말", None)]),
        )
    assert model.calls == []


def test_nine_chunks_are_batched_as_four_four_one(tmp_path):
    frames = 900
    path = tmp_path / "canonical.wav"
    write_wav(path, frames)
    chunks = [
        chunk(index, index * 100, (index + 1) * 100, f"c{index}")
        for index in range(9)
    ]
    service, model = make_service(batch_size=4)
    result = service.align_chunks(str(path), payload(frames, chunks))
    assert [len(call["text"]) for call in model.calls] == [4, 4, 1]
    assert [call["text"] for call in model.calls] == [
        ["c0", "c1", "c2", "c3"],
        ["c4", "c5", "c6", "c7"],
        ["c8"],
    ]
    assert all(rate == SAMPLE_RATE for call in model.calls for rate in call["rates"])
    assert [item["items"][0]["text"] for item in result["chunks"]] == [f"c{i}" for i in range(9)]
    assert [item["index"] for item in result["chunks"]] == list(range(9))


def test_global_offset_applied_once(tmp_path, monkeypatch):
    samples = 30 * 60 * SAMPLE_RATE
    monkeypatch.setattr(
        "app.aligner.read_canonical_wav",
        lambda path: np.zeros(samples, dtype=np.float32),
    )
    service, _model = make_service(
        FakeModel(lambda audio, text, language: [AlignResult([Span(text="다음", start_time=0.120, end_time=0.200)])])
    )
    result = service.align_chunks(
        "ignored.wav",
        payload(samples, [chunk(1, 1_888_000, 1_888_000 + SAMPLE_RATE, "다음 구간의 전사입니다.")]),
    )
    item = result["chunks"][0]["items"][0]
    assert item["start_time"] == pytest.approx(118.120)
    assert item["end_time"] == pytest.approx(118.200)
    assert item["start_time"] != pytest.approx(236.120)
    assert result["chunks"][0]["text"] == "다음 구간의 전사입니다."
    monkeypatch.undo()

    frames = SAMPLE_RATE
    path = tmp_path / "short.wav"
    write_wav(path, frames)
    service, _model = make_service(
        FakeModel(lambda audio, text, language: [AlignResult([Span(text="a", start_time=0.5, end_time=0.6)])])
    )
    result = service.align_chunks(
        str(path),
        payload(frames, [chunk(0, 123, frames, "임의")]),
    )
    start = result["chunks"][0]["items"][0]["start_time"]
    assert start == pytest.approx(123 / SAMPLE_RATE + 0.5)
    assert start != pytest.approx(round(123 / SAMPLE_RATE, 3) + 0.5)


def test_overlap_chunks_stay_in_input_order(tmp_path):
    frames = 12_000
    path = tmp_path / "canonical.wav"
    write_wav(path, frames)
    service, _model = make_service()
    result = service.align_chunks(
        str(path),
        payload(
            frames,
            [
                chunk(2, 4_000, 12_000, "둘째"),
                chunk(0, 0, 8_000, "첫째"),
            ],
        ),
    )
    assert result["overlap_deduplicated"] is False
    assert result["time_reference"] == "audio_start"
    assert [item["index"] for item in result["chunks"]] == [2, 0]
    assert result["chunks"][0]["start_sample"] == 4_000
    assert result["chunks"][1]["end_sample"] == 8_000


def test_result_mismatch_mid_batch_failure_and_bad_timestamps_fail_the_request(tmp_path):
    frames = 900
    path = tmp_path / "canonical.wav"
    write_wav(path, frames)
    chunks = [chunk(index, index * 100, (index + 1) * 100, f"c{index}") for index in range(9)]

    def fail_second(audio, text, language):
        if fail_second.calls == 1:
            raise RuntimeError("batch failed")
        fail_second.calls += 1
        return [AlignResult([Span()]) for _ in text]

    fail_second.calls = 0
    service, _model = make_service(FakeModel(fail_second), batch_size=4)
    with pytest.raises(AlignmentError, match="batch failed"):
        service.align_chunks(str(path), payload(frames, chunks))

    service, _model = make_service(FakeModel(lambda audio, text, language: []))
    with pytest.raises(AlignmentError, match="result count mismatch"):
        service.align_chunks(str(path), payload(frames, [chunks[0]]))

    frames = SAMPLE_RATE
    path = tmp_path / "one.wav"
    write_wav(path, frames)
    body = payload(frames, [chunk(7, 0, frames, "말")])
    for bad in (float("nan"), float("inf"), True, "0.1"):
        service, _model = make_service(
            FakeModel(lambda audio, text, language, bad=bad: [AlignResult([Span(start_time=bad, end_time=0.2)])])
        )
        with pytest.raises(AlignmentError, match="chunk index 7: invalid timestamp"):
            service.align_chunks(str(path), body)

    service, _model = make_service(
        FakeModel(lambda audio, text, language: [AlignResult([Span(start_time=0.4, end_time=0.2)])])
    )
    with pytest.raises(AlignmentError, match="chunk index 7: timestamp reversed"):
        service.align_chunks(str(path), body)

    service, _model = make_service(
        FakeModel(lambda audio, text, language: [AlignResult([Span(text="말", start_time=-0.081, end_time=0.2)])])
    )
    clamped = service.align_chunks(str(path), body)
    assert clamped["chunks"][0]["items"][0]["start_time"] == pytest.approx(0.0)
    assert clamped["chunks"][0]["items"][0]["end_time"] == pytest.approx(0.2)


def test_timestamp_outside_chunk_is_clamped_to_bounds(tmp_path):
    frames = SAMPLE_RATE * 2
    path = tmp_path / "canonical.wav"
    write_wav(path, frames)
    service, _model = make_service(
        FakeModel(
            lambda audio, text, language: [
                AlignResult(
                    [
                        Span(text="", start_time=0.0, end_time=0.1),
                        Span(text="단어", start_time=-0.08, end_time=1.08),
                    ]
                )
            ]
        )
    )
    result = service.align_chunks(
        str(path),
        payload(frames, [chunk(3, SAMPLE_RATE, frames, "단어")]),
    )
    item = result["chunks"][0]["items"]
    assert len(item) == 1
    assert item[0]["text"] == "단어"
    assert item[0]["start_time"] == pytest.approx(1.0)
    assert item[0]["end_time"] == pytest.approx(2.0)
    assert result["chunks"][0]["status"] == "aligned"

    service, _model = make_service(
        FakeModel(lambda audio, text, language: [AlignResult([Span(text="단어", start_time=0.0, end_time=1.081)])])
    )
    overrun = service.align_chunks(
        str(path),
        payload(frames, [chunk(3, SAMPLE_RATE, frames, "단어")]),
    )
    overrun_item = overrun["chunks"][0]["items"][0]
    assert overrun_item["start_time"] == pytest.approx(1.0)
    assert overrun_item["end_time"] == pytest.approx(2.0)

    service, _model = make_service(FakeModel(lambda audio, text, language: [AlignResult([])]))
    result = service.align_chunks(str(path), payload(frames, [chunk(0, 0, frames, "...")]))
    assert result["chunks"][0]["status"] == "aligned"
    assert result["chunks"][0]["items"] == []


def test_thirty_minute_audio_is_allowed_and_chunk_limit_is_180_seconds(tmp_path, monkeypatch):
    max_samples = 180 * SAMPLE_RATE
    validate_chunk_span(0, 0, max_samples, max_samples, max_samples)
    with pytest.raises(ClientError, match="exceeds"):
        validate_chunk_span(0, 0, max_samples + 1, max_samples + 1, max_samples)

    samples = 30 * 60 * SAMPLE_RATE
    monkeypatch.setattr(
        "app.aligner.read_canonical_wav",
        lambda path: np.zeros(samples, dtype=np.float32),
    )
    service, model = make_service()
    result = service.align_chunks(
        "ignored.wav",
        payload(samples, [chunk(0, 0, 120 * SAMPLE_RATE, "삼십분")]),
    )
    assert result["chunks"][0]["status"] == "aligned"
    assert model.calls

    too_long = max_samples + 1
    monkeypatch.setattr(
        "app.aligner.read_canonical_wav",
        lambda path: np.zeros(too_long, dtype=np.float32),
    )
    service, model = make_service()
    with pytest.raises(ClientError, match="chunk index 0: chunk length"):
        service.align_chunks(
            "ignored.wav",
            payload(too_long, [chunk(0, 0, too_long, "너무김")]),
        )
    assert model.calls == []


def test_whitespace_text_still_validates_sample_range(tmp_path):
    frames = 100
    path = tmp_path / "canonical.wav"
    write_wav(path, frames)
    service, model = make_service()
    with pytest.raises(ClientError, match="chunk index 2:"):
        service.align_chunks(
            str(path),
            payload(frames, [chunk(2, 0, frames + 1, "   ")]),
        )
    assert model.calls == []


def test_slice_uses_half_open_sample_range(tmp_path):
    frames = 10
    path = tmp_path / "canonical.wav"
    samples = np.arange(frames, dtype=np.float32)
    sf.write(str(path), samples, SAMPLE_RATE, subtype="PCM_16")
    seen = {}

    def capture(audio, text, language):
        seen["audio"] = np.array(audio[0][0], copy=True)
        return [AlignResult([Span()])]

    service, _model = make_service(FakeModel(capture))
    service.align_chunks(str(path), payload(frames, [chunk(0, 2, 5, "구간")]))
    assert seen["audio"].shape == (3,)

import asyncio
import json
import os
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.aligner import CANONICAL_LANGUAGES, ForcedAlignerService
from app.config import load_settings
from app.main import complete_alignment, create_app
from app.schemas import SAMPLE_RATE
from conftest import FakeModel, make_service, settings


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
        "audio": {"sample_rate": SAMPLE_RATE, "channels": 1, "num_samples": num_samples},
        "chunks": chunks,
    }
    body.update(extra)
    return body


def post_align(client, wav_path, body, payload_text=None):
    with open(wav_path, "rb") as handle:
        return client.post(
            "/align",
            files={"file": ("canonical.wav", handle, "audio/wav")},
            data={"payload": json.dumps(body) if payload_text is None else payload_text},
        )


def test_config_rejects_invalid_values_and_keeps_defaults():
    assert load_settings({}).dtype == "bfloat16"
    assert load_settings({}).batch_size == 4
    assert load_settings({}).max_chunk_seconds == 180
    with pytest.raises(RuntimeError):
        load_settings({"FA_DTYPE": "float64"})
    with pytest.raises(RuntimeError):
        load_settings({"FA_BATCH_SIZE": "0"})
    with pytest.raises(RuntimeError):
        load_settings({"FA_BATCH_SIZE": "4.0"})
    with pytest.raises(RuntimeError):
        load_settings({"FA_MAX_CHUNK_SECONDS": "181"})
    with pytest.raises(RuntimeError):
        load_settings({"FA_MODEL_PATH": "  "})


def test_load_checks_model_languages_and_failure_stops_startup(monkeypatch):
    service = ForcedAlignerService(settings())

    class Match:
        def get_supported_languages(self):
            return [name.upper() for name in CANONICAL_LANGUAGES]

    monkeypatch.setattr(service, "_load_aligner", lambda: Match())
    service.load()
    assert service.loaded is True

    mismatched = ForcedAlignerService(settings())

    class Mismatch:
        def get_supported_languages(self):
            return ["english"]

    monkeypatch.setattr(mismatched, "_load_aligner", lambda: Mismatch())
    with pytest.raises(RuntimeError, match="support_languages mismatch"):
        mismatched.load()
    assert mismatched.loaded is False

    failing = ForcedAlignerService(settings())

    def explode():
        raise RuntimeError("missing weights")

    monkeypatch.setattr(failing, "_load_aligner", explode)
    app = create_app(service=failing, load_on_startup=True)
    with pytest.raises(RuntimeError, match="missing weights"):
        with TestClient(app):
            pass
    assert failing.loaded is False


def test_health_reports_loaded_and_not_ready():
    service, _model = make_service()
    service.loaded = True
    app = create_app(service=service, load_on_startup=False)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

    service.loaded = False
    app = create_app(service=service, load_on_startup=False)
    with TestClient(app) as client:
        response = client.get("/health")
        denied = client.post(
            "/align",
            files={"file": ("canonical.wav", b"RIFFxxxx", "audio/wav")},
            data={"payload": "{}"},
        )
    assert response.status_code == 503
    assert response.json() == {"status": "loading", "model_loaded": False}
    assert denied.status_code == 503


def test_rejects_bad_wav_metadata_range_and_duplicate_index_before_model(tmp_path):
    frames = 1600
    wav_path = tmp_path / "canonical.wav"
    write_wav(wav_path, frames)
    service, model = make_service()
    service.loaded = True
    app = create_app(service=service, load_on_startup=False)

    float_path = tmp_path / "float.wav"
    sf.write(str(float_path), np.zeros(frames, dtype=np.float32), SAMPLE_RATE, subtype="FLOAT")
    stereo_path = tmp_path / "stereo.wav"
    write_wav(stereo_path, frames, channels=2)
    rate_path = tmp_path / "rate.wav"
    write_wav(rate_path, frames, sample_rate=8000)
    wide_path = tmp_path / "pcm24.wav"
    write_wav(wide_path, frames, sampwidth=3)
    truncated = tmp_path / "truncated.wav"
    write_wav(truncated, frames)
    truncated.write_bytes(truncated.read_bytes()[:80])

    with TestClient(app) as client:
        cases = [
            (float_path, payload(frames, [chunk(0, 0, frames, "말")])),
            (stereo_path, payload(frames, [chunk(0, 0, frames, "말")])),
            (rate_path, payload(frames, [chunk(0, 0, frames, "말")])),
            (wide_path, payload(frames, [chunk(0, 0, frames, "말")])),
            (truncated, payload(frames, [chunk(0, 0, frames, "말")])),
            (wav_path, payload(frames + 5, [chunk(0, 0, frames, "말")])),
            (wav_path, payload(frames, [chunk(0, 0, frames + 1, "말")])),
            (wav_path, payload(frames, [chunk(1, 0, 10, "말"), chunk(1, 10, 20, "말")])),
            (wav_path, payload(frames, [chunk(True, 0, frames, "말")])),
            (wav_path, payload(frames, [chunk(1.5, 0, frames, "말")])),
            (wav_path, payload(frames, [chunk("1", 0, frames, "말")])),
        ]
        # Rebuild payloads whose index is not a strict int. chunk() would put them in JSON.
        cases[-3] = (
            wav_path,
            {
                "audio": {"sample_rate": SAMPLE_RATE, "channels": 1, "num_samples": frames},
                "chunks": [
                    {"index": True, "start_sample": 0, "end_sample": frames, "text": "말", "language": "ko"}
                ],
            },
        )
        cases[-2] = (
            wav_path,
            {
                "audio": {"sample_rate": SAMPLE_RATE, "channels": 1, "num_samples": frames},
                "chunks": [
                    {"index": 1.5, "start_sample": 0, "end_sample": frames, "text": "말", "language": "ko"}
                ],
            },
        )
        cases[-1] = (
            wav_path,
            {
                "audio": {"sample_rate": SAMPLE_RATE, "channels": 1, "num_samples": frames},
                "chunks": [
                    {"index": "1", "start_sample": 0, "end_sample": frames, "text": "말", "language": "ko"}
                ],
            },
        )
        for path, body in cases:
            response = post_align(client, path, body)
            assert response.status_code == 400, response.text
            assert "detail" in response.json()
        missing = client.post("/align", data={"payload": "{}"})
        assert missing.status_code == 400
    assert model.calls == []


def test_full_asr_payload_uses_only_audio_and_chunks(tmp_path):
    frames = 1600
    wav_path = tmp_path / "canonical.wav"
    write_wav(wav_path, frames)
    service, model = make_service()
    service.loaded = True
    app = create_app(service=service, load_on_startup=False)
    body = payload(
        frames,
        [chunk(0, 0, frames, "청크 전사")],
        task="transcribe",
        language="ko",
        duration=frames / SAMPLE_RATE,
        text="병합된 전사",
        segments=[{"id": 0, "start": 0.0, "end": 1.0, "text": "병합된 전사", "words": []}],
    )
    with TestClient(app) as client:
        response = post_align(client, wav_path, body)
        missing_audio = post_align(
            client,
            wav_path,
            {"language": "ko", "segments": body["segments"], "chunks": body["chunks"]},
        )
        segments_only = post_align(
            client,
            wav_path,
            {"text": "병합된 전사", "segments": body["segments"]},
        )
        invalid_json = post_align(client, wav_path, body, payload_text="{")
    assert response.status_code == 200, response.text
    assert response.json()["chunks"][0]["text"] == "청크 전사"
    assert model.calls[0]["text"] == ["청크 전사"]
    assert missing_audio.status_code == 400
    assert segments_only.status_code == 400
    assert "audio and chunks" in segments_only.json()["detail"]
    assert invalid_json.status_code == 400
    assert len(model.calls) == 1


def test_empty_cleaned_text_is_preserved_and_null_language_is_rejected(tmp_path):
    frames = 800
    wav_path = tmp_path / "canonical.wav"
    write_wav(wav_path, frames)
    service, model = make_service()
    service.loaded = True
    app = create_app(service=service, load_on_startup=False)
    kept = payload(frames, [chunk(0, 0, frames, "", language=None)])
    rejected = payload(frames, [chunk(1, 0, frames, "남은 말", None)])
    with TestClient(app) as client:
        empty = post_align(client, wav_path, kept)
        missing_language = post_align(client, wav_path, rejected)
    assert empty.status_code == 200, empty.text
    body = empty.json()
    assert body["chunks"][0]["status"] == "skipped_empty_text"
    assert body["chunks"][0]["text"] == ""
    assert body["chunks"][0]["language"] is None
    assert body["chunks"][0]["items"] == []
    assert missing_language.status_code == 400
    assert "chunk index 1" in missing_language.json()["detail"]
    assert model.calls == []


def test_alignment_failure_is_500_and_temporary_files_are_removed(tmp_path, monkeypatch):
    frames = 400
    wav_path = tmp_path / "canonical.wav"
    write_wav(wav_path, frames)
    created = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        directory = real_mkdtemp(*args, **kwargs)
        created.append(directory)
        return directory

    monkeypatch.setattr("app.main.tempfile.mkdtemp", tracking_mkdtemp)
    service, _model = make_service(FakeModel(lambda audio, text, language: (_ for _ in ()).throw(RuntimeError("oom"))))
    service.loaded = True
    app = create_app(service=service, load_on_startup=False)
    with TestClient(app) as client:
        failed = post_align(client, wav_path, payload(frames, [chunk(0, 0, frames, "말")]))
        rejected = post_align(client, wav_path, payload(frames, []), payload_text="not-json")
    assert failed.status_code == 500
    assert "oom" in failed.json()["detail"]
    assert "Traceback" not in failed.text
    assert rejected.status_code == 400
    assert created
    assert all(not os.path.exists(directory) for directory in created)


def test_concurrent_requests_do_not_overlap_model_execution(tmp_path):
    frames = 800
    wav_path = tmp_path / "canonical.wav"
    write_wav(wav_path, frames)
    state = {"current": 0, "peak": 0}
    gate = threading.Lock()
    started = threading.Event()

    def align(audio, text, language):
        with gate:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            if state["current"] == 1:
                started.set()
        time.sleep(0.2)
        with gate:
            state["current"] -= 1
        return FakeModel().align(audio, text, language)

    service, _model = make_service(FakeModel(align))
    service.loaded = True
    app = create_app(service=service, load_on_startup=False)

    async def run():
        transport = ASGITransport(app=app)
        body = payload(frames, [chunk(0, 0, frames, "동시")])
        async with AsyncClient(transport=transport, base_url="http://fa") as client:
            with open(wav_path, "rb") as handle:
                wav_bytes = handle.read()

            async def one():
                return await client.post(
                    "/align",
                    files={"file": ("canonical.wav", wav_bytes, "audio/wav")},
                    data={"payload": json.dumps(body)},
                )

            first = asyncio.create_task(one())
            assert await asyncio.to_thread(started.wait, 5)
            health = await client.get("/health")
            second = asyncio.create_task(one())
            done = await asyncio.gather(first, second)
            return health, done

    health, responses = asyncio.run(run())
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    assert [response.status_code for response in responses] == [200, 200]
    assert state["peak"] == 1


def test_cancel_waits_for_worker_before_releasing_lock_and_temp_files():
    asyncio.run(_cancel_waits())


async def _cancel_waits():
    lock = asyncio.Lock()
    directory = tempfile.mkdtemp(prefix="fa-align-")
    path = os.path.join(directory, "input.wav")
    with open(path, "wb") as handle:
        handle.write(b"pcm")
    started = threading.Event()
    release = threading.Event()
    seen = {}

    def work():
        seen["exists_start"] = os.path.exists(path)
        seen["locked_start"] = lock.locked()
        started.set()
        assert release.wait(5)
        seen["exists_end"] = os.path.exists(path)
        seen["locked_end"] = lock.locked()
        return {"ok": True}

    task = asyncio.create_task(complete_alignment(lock, directory, work))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    assert lock.locked()
    assert os.path.exists(path)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not lock.locked()
    assert seen["exists_start"] and seen["exists_end"]
    assert seen["locked_start"] and seen["locked_end"]
    assert not os.path.exists(directory)

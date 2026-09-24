import asyncio
import json
import os
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager, contextmanager
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

_CONTROL_CASES: list[dict] = []
_CONTROL_OUTPUT = Path(__file__).resolve().parent / "outputs" / "control_api.md"


def _remember(test_name: str, method: str, url: str, response) -> None:
    path = url.split("?", 1)[0]
    if path.startswith("http://"):
        path = "/" + path.split("/", 3)[-1]
    try:
        body = response.json()
    except Exception:
        body = response.text
    _CONTROL_CASES.append(
        {
            "test": test_name,
            "method": method.upper(),
            "path": path,
            "status_code": response.status_code,
            "body": body,
        }
    )


def _wrap_sync(client, test_name: str):
    original = client.request

    def request(method, url, **kwargs):
        response = original(method, url, **kwargs)
        _remember(test_name, method, str(url), response)
        return response

    client.request = request
    return client


def _wrap_async(client, test_name: str):
    original = client.request

    async def request(method, url, **kwargs):
        response = await original(method, url, **kwargs)
        _remember(test_name, method, str(url), response)
        return response

    client.request = request
    return client


@contextmanager
def recorded_client(app, test_name: str):
    with TestClient(app) as client:
        yield _wrap_sync(client, test_name)


@asynccontextmanager
async def recorded_async(transport, test_name: str):
    async with AsyncClient(transport=transport, base_url="http://fa") as client:
        yield _wrap_async(client, test_name)


def _write_control_api_document() -> None:
    if not _CONTROL_CASES:
        return
    _CONTROL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Control API 테스트 결과",
        "",
        "가중치 없이 fake aligner로 control API를 호출한 기록이다.",
        "",
    ]
    current = None
    for case in _CONTROL_CASES:
        if case["test"] != current:
            current = case["test"]
            lines.extend([f"## {current}", ""])
        lines.extend(
            [
                f"### {case['method']} {case['path']}",
                "",
                f"HTTP {case['status_code']}",
                "",
                "```json",
                json.dumps(case["body"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    _CONTROL_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(scope="module", autouse=True)
def control_api_document():
    yield
    _write_control_api_document()


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


def test_startup_stays_unloaded_and_load_reports_weight_or_language_failure():
    service = ForcedAlignerService(settings())

    class Match:
        def get_supported_languages(self):
            return [name.upper() for name in CANONICAL_LANGUAGES]

    service._load_aligner = lambda: Match()
    app = create_app(service=service, load_on_startup=True)
    with recorded_client(app, "startup_stays_unloaded") as client:
        status = client.get("/control/status")
        assert status.status_code == 200
        assert status.json() == {
            "state": "unloaded",
            "residency": "not_resident",
            "active_requests": 0,
            "last_error": None,
        }
        loaded = client.post("/control/load")
    assert loaded.status_code == 200
    assert loaded.json()["state"] == "ready"
    assert loaded.json()["residency"] == "resident"
    assert loaded.json()["last_error"] is None
    assert service.loaded is True

    mismatched = ForcedAlignerService(settings())

    class Mismatch:
        def get_supported_languages(self):
            return ["english"]

    mismatched._load_aligner = lambda: Mismatch()
    mismatch_app = create_app(service=mismatched)
    with recorded_client(mismatch_app, "load_language_mismatch") as client:
        failed = client.post("/control/load")
        still_up = client.get("/control/status")
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "LOAD_FAILED"
    assert still_up.status_code == 200
    assert still_up.json()["state"] == "failed"
    assert still_up.json()["residency"] == "not_resident"
    assert still_up.json()["last_error"]["code"] == "LOAD_FAILED"
    assert mismatched.loaded is False

    failing = ForcedAlignerService(settings())

    def explode():
        raise RuntimeError("missing weights")

    failing._load_aligner = explode
    failed_app = create_app(service=failing, load_on_startup=True)
    with recorded_client(failed_app, "load_missing_weights") as client:
        denied = client.post(
            "/align",
            files={"file": ("canonical.wav", b"RIFFxxxx", "audio/wav")},
            data={"payload": "{}"},
        )
        failed = client.post("/control/load")
        status = client.get("/control/status")
    assert denied.status_code == 503
    assert denied.json() == {"status": "loading", "model_loaded": False}
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "LOAD_FAILED"
    assert status.json()["state"] == "failed"
    assert status.json()["residency"] == "not_resident"
    assert failing.loaded is False


def test_health_reports_loaded_and_not_ready():
    service, _model = make_service()
    service.loaded = True
    app = create_app(service=service, load_on_startup=False)
    with recorded_client(app, "health_when_ready") as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}

    service.loaded = False
    app = create_app(service=service, load_on_startup=False)
    with recorded_client(app, "health_when_not_ready") as client:
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


class _Mem:
    def __init__(self, value=0):
        self.value = value

    def __call__(self):
        return self.value


class _Languages:
    def get_supported_languages(self):
        return list(CANONICAL_LANGUAGES)


def _ready_service():
    service, model = make_service()
    service.loaded = True
    return service, model


def _block_load(service, entered, release, calls):
    def load_aligner():
        calls.append("load")
        entered.set()
        assert release.wait(5)
        return _Languages()

    service._load_aligner = load_aligner


def _post_json(client, path, body):
    return client.post(path, content=json.dumps(body), headers={"content-type": "application/json"})


def test_control_body_status_and_idempotent_load_unload(tmp_path):
    service, _model = make_service()
    calls = []
    service._load_aligner = lambda: calls.append("load") or _Languages()
    app = create_app(service=service)
    frames = 1600
    wav_path = tmp_path / "canonical.wav"
    write_wav(wav_path, frames)
    with recorded_client(app, "control_body_and_idempotent_load") as client:
        assert client.get("/health").status_code == 503
        denied = post_align(client, wav_path, payload(frames, [chunk(0, 0, frames, "안녕")]))
        assert denied.status_code == 503
        assert denied.json() == {"status": "loading", "model_loaded": False}
        bad = _post_json(client, "/control/load", {"model": "x"})
        assert bad.status_code == 400
        assert bad.json()["error"]["code"] == "BAD_REQUEST"
        assert client.post("/control/unload").status_code == 200
        assert client.post("/control/unload").json()["state"] == "unloaded"
        assert calls == []
        loaded = client.post("/control/load", content=b"{}")
        again = client.post("/control/load")
        status = client.get("/control/status")
    assert loaded.status_code == 200
    assert loaded.json()["state"] == "ready"
    assert again.status_code == 200
    assert again.json() == loaded.json()
    assert calls == ["load"]
    assert status.json()["active_requests"] == 0
    assert client_health_ok(app)


def client_health_ok(app):
    with recorded_client(app, "health_after_load") as client:
        health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True}
    return True


def test_load_and_unload_share_inflight_work_and_reject_conflicts():
    service, _model = make_service()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    _block_load(service, entered, release, calls)
    app = create_app(service=service)

    async def run():
        transport = ASGITransport(app=app)
        async with recorded_async(transport, "shared_load") as client:
            first = asyncio.create_task(client.post("/control/load"))
            assert await asyncio.to_thread(entered.wait, 5)
            status = await client.get("/control/status")
            conflict = await client.post("/control/unload")
            second = asyncio.create_task(client.post("/control/load"))
            await asyncio.sleep(0.05)
            release.set()
            done = await asyncio.gather(first, second)
            return status, conflict, done

    status, conflict, done = asyncio.run(run())
    assert status.status_code == 200
    assert status.json()["state"] == "loading"
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "LIFECYCLE_CONFLICT"
    assert [item.status_code for item in done] == [200, 200]
    assert done[0].json() == done[1].json()
    assert done[0].json()["state"] == "ready"
    assert calls == ["load"]

    entered.clear()
    release.clear()
    calls.clear()

    def slow_release():
        calls.append("unload")
        entered.set()
        assert release.wait(5)
        service._aligner = None
        service.loaded = False

    service.release = slow_release

    async def unload_pair():
        transport = ASGITransport(app=app)
        async with recorded_async(transport, "shared_unload") as client:
            first = asyncio.create_task(client.post("/control/unload"))
            assert await asyncio.to_thread(entered.wait, 5)
            status = await client.get("/control/status")
            conflict = await client.post("/control/load")
            second = asyncio.create_task(client.post("/control/unload"))
            await asyncio.sleep(0.05)
            release.set()
            done = await asyncio.gather(first, second)
            return status, conflict, done

    status, conflict, done = asyncio.run(unload_pair())
    assert status.json()["state"] == "unloading"
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "LIFECYCLE_CONFLICT"
    assert [item.status_code for item in done] == [200, 200]
    assert done[0].json()["state"] == "unloaded"
    assert done[0].json()["residency"] == "not_resident"
    assert done[0].json()["active_requests"] == 0
    assert calls == ["unload"]


def test_load_disconnect_and_failure_residency_then_recovery():
    service, _model = make_service()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    _block_load(service, entered, release, calls)
    mem = _Mem(0)
    app = create_app(service=service, memory=mem)

    async def disconnect():
        transport = ASGITransport(app=app)
        async with recorded_async(transport, "load_disconnect") as client:
            task = asyncio.create_task(client.post("/control/load"))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            during = (await client.get("/control/status")).json()
            release.set()
            for _ in range(50):
                body = (await client.get("/control/status")).json()
                if body["state"] != "loading":
                    return during, body
                await asyncio.sleep(0.02)
            return during, body

    during, after = asyncio.run(disconnect())
    assert during["state"] == "loading"
    assert after["state"] == "ready"
    assert after["residency"] == "resident"
    assert calls == ["load"]

    entered.clear()
    release.clear()
    original_release = service.release

    def slow_release():
        calls.append("unload")
        entered.set()
        assert release.wait(5)
        service._aligner = None
        service.loaded = False

    service.release = slow_release

    async def disconnect_unload():
        transport = ASGITransport(app=app)
        async with recorded_async(transport, "unload_disconnect") as client:
            task = asyncio.create_task(client.post("/control/unload"))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            during = (await client.get("/control/status")).json()
            release.set()
            for _ in range(50):
                body = (await client.get("/control/status")).json()
                if body["state"] != "unloading":
                    return during, body
                await asyncio.sleep(0.02)
            return during, body

    during, after = asyncio.run(disconnect_unload())
    assert during["state"] == "unloading"
    assert after["state"] == "unloaded"
    assert after["residency"] == "not_resident"
    assert calls == ["load", "unload"]

    entered.clear()
    release.clear()
    calls.clear()
    service.release = original_release
    service.release()
    app.state.lifecycle.state = "unloaded"
    app.state.lifecycle.residency = "not_resident"
    app.state.lifecycle.last_error = None
    app.state.lifecycle._baseline = None
    service.loaded = False

    def partial():
        calls.append("partial")
        mem.value = 32
        raise RuntimeError("partial allocation")

    service._load_aligner = partial
    with recorded_client(app, "load_failure_unknown_residency") as client:
        failed = client.post("/control/load")
        status = client.get("/control/status")
        retry = client.post("/control/load")
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "LOAD_FAILED"
    assert status.json()["state"] == "failed"
    assert status.json()["residency"] == "unknown"
    assert status.json()["last_error"]["code"] == "LOAD_FAILED"
    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "LIFECYCLE_CONFLICT"

    mem.value = 0
    with recorded_client(app, "recovery_unload_from_unknown") as client:
        recovered = client.post("/control/unload")
    assert recovered.status_code == 200
    assert recovered.json()["state"] == "unloaded"
    assert recovered.json()["residency"] == "not_resident"
    assert recovered.json()["last_error"] is None

    def missing():
        raise RuntimeError("missing weights")

    service._load_aligner = missing
    with recorded_client(app, "load_failure_not_resident") as client:
        failed = client.post("/control/load")
        status = client.get("/control/status")
    assert failed.status_code == 500
    assert status.json()["state"] == "failed"
    assert status.json()["residency"] == "not_resident"
    with recorded_client(app, "unload_noop_after_failed_not_resident") as client:
        noop = client.post("/control/unload")
    assert noop.status_code == 200
    assert noop.json()["state"] == "unloaded"
    assert noop.json()["last_error"] is None
    service._load_aligner = lambda: calls.append("retry") or _Languages()
    with recorded_client(app, "load_retry_after_not_resident") as client:
        retried = client.post("/control/load")
    assert retried.status_code == 200
    assert retried.json()["state"] == "ready"
    assert retried.json()["last_error"] is None

    def keep_resident():
        mem.value = 64

    service.release = keep_resident
    with recorded_client(app, "unload_failure_resident") as client:
        failed = client.post("/control/unload")
        status = client.get("/control/status")
        blocked = client.post("/control/load")
    assert failed.status_code == 500
    assert failed.json()["error"]["code"] == "UNLOAD_FAILED"
    assert status.json()["state"] == "failed"
    assert status.json()["residency"] == "resident"
    assert blocked.status_code == 409

    def free():
        mem.value = 0
        service._aligner = None
        service.loaded = False

    service.release = free
    with recorded_client(app, "recovery_unload_from_resident") as client:
        recovered = client.post("/control/unload")
    assert recovered.status_code == 200
    assert recovered.json() == {
        "state": "unloaded",
        "residency": "not_resident",
        "active_requests": 0,
        "last_error": None,
    }

    app.state.lifecycle.status_broken = True
    with recorded_client(app, "status_unavailable") as client:
        broken = client.get("/control/status")
    assert broken.status_code == 500
    assert broken.json()["error"]["code"] == "STATUS_FAILED"


def test_active_requests_batches_races_and_cancel(tmp_path):
    frames = 1600
    wav_path = tmp_path / "canonical.wav"
    write_wav(wav_path, frames)
    seen = []

    def align(audio, text, language):
        seen.append(app.state.lifecycle.snapshot()["active_requests"])
        return FakeModel().align(audio, text, language)

    service, _model = make_service(FakeModel(align), batch_size=4)
    service.loaded = True
    app = create_app(service=service)
    chunks = [chunk(index, 0, frames, f"말{index}") for index in range(9)]
    with recorded_client(app, "active_requests_one_per_request") as client:
        rejected = post_align(client, wav_path, payload(frames, []), payload_text="not-json")
        response = post_align(client, wav_path, payload(frames, chunks))
        status = client.get("/control/status")
    assert rejected.status_code == 400
    assert response.status_code == 200
    assert seen == [1, 1, 1]
    assert status.json()["active_requests"] == 0

    started = threading.Event()
    release = threading.Event()
    calls = []

    def blocked(audio, text, language):
        calls.append("align")
        started.set()
        assert release.wait(5)
        return FakeModel().align(audio, text, language)

    service, _model = make_service(FakeModel(blocked))
    service.loaded = True
    app = create_app(service=service)
    body = payload(frames, [chunk(0, 0, frames, "동시")])

    async def two_requests():
        transport = ASGITransport(app=app)
        async with recorded_async(transport, "active_requests_two_and_busy") as client:
            wav = wav_path.read_bytes()

            async def one():
                return await client.post(
                    "/align",
                    files={"file": ("canonical.wav", wav, "audio/wav")},
                    data={"payload": json.dumps(body)},
                )

            first = asyncio.create_task(one())
            assert await asyncio.to_thread(started.wait, 5)
            second = asyncio.create_task(one())
            for _ in range(50):
                if app.state.lifecycle.snapshot()["active_requests"] == 2:
                    break
                await asyncio.sleep(0.02)
            during = (await client.get("/control/status")).json()
            busy = await client.post("/control/unload")
            release.set()
            done = await asyncio.gather(first, second)
            after = (await client.get("/control/status")).json()
            return during, busy, done, after

    during, busy, done, after = asyncio.run(two_requests())
    assert during["state"] == "ready"
    assert during["active_requests"] == 2
    assert busy.status_code == 409
    assert busy.json()["error"]["code"] == "BUSY"
    assert [item.status_code for item in done] == [200, 200]
    assert after["active_requests"] == 0

    unload_entered = threading.Event()
    unload_release = threading.Event()

    def slow_release():
        unload_entered.set()
        assert unload_release.wait(5)
        service._aligner = None
        service.loaded = False

    service.release = slow_release
    service._aligner = FakeModel()
    service.loaded = True
    app.state.lifecycle.state = "ready"
    app.state.lifecycle.residency = "resident"
    app.state.lifecycle._baseline = 0

    async def unload_wins():
        transport = ASGITransport(app=app)
        async with recorded_async(transport, "unload_before_accept") as client:
            unloading = asyncio.create_task(client.post("/control/unload"))
            assert await asyncio.to_thread(unload_entered.wait, 5)
            denied = await client.post(
                "/align",
                files={"file": ("canonical.wav", wav_path.read_bytes(), "audio/wav")},
                data={"payload": json.dumps(body)},
            )
            unload_release.set()
            unloaded = await unloading
            return denied, unloaded

    denied, unloaded = asyncio.run(unload_wins())
    assert denied.status_code == 503
    assert calls == ["align", "align"]
    assert unloaded.status_code == 200

    hold = threading.Event()
    let_go = threading.Event()

    def holding(audio, text, language):
        hold.set()
        assert let_go.wait(5)
        return FakeModel().align(audio, text, language)

    service, _model = make_service(FakeModel(holding))
    service.loaded = True
    app = create_app(service=service)

    async def cancel_after_accept():
        transport = ASGITransport(app=app)
        async with recorded_async(transport, "cancel_after_accept") as client:
            task = asyncio.create_task(
                client.post(
                    "/align",
                    files={"file": ("canonical.wav", wav_path.read_bytes(), "audio/wav")},
                    data={"payload": json.dumps(body)},
                )
            )
            assert await asyncio.to_thread(hold.wait, 5)
            during = app.state.lifecycle.snapshot()["active_requests"]
            task.cancel()
            await asyncio.sleep(0.05)
            while_cancelled = app.state.lifecycle.snapshot()["active_requests"]
            let_go.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            return during, while_cancelled, app.state.lifecycle.snapshot()["active_requests"]

    during, while_cancelled, finished = asyncio.run(cancel_after_accept())
    assert during == 1
    assert while_cancelled == 1
    assert finished == 0

    gate = threading.Event()
    opened = threading.Event()

    def pause_validate(self, wav_path, parsed):
        del self, wav_path, parsed
        gate.set()
        assert opened.wait(5)

    service.validate_request = pause_validate.__get__(service, type(service))

    async def cancel_before_accept():
        transport = ASGITransport(app=app)
        async with recorded_async(transport, "cancel_before_accept") as client:
            task = asyncio.create_task(
                client.post(
                    "/align",
                    files={"file": ("canonical.wav", wav_path.read_bytes(), "audio/wav")},
                    data={"payload": json.dumps(body)},
                )
            )
            assert await asyncio.to_thread(gate.wait, 5)
            before = app.state.lifecycle.snapshot()["active_requests"]
            task.cancel()
            opened.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            return before, app.state.lifecycle.snapshot()["active_requests"]

    before, after_cancel = asyncio.run(cancel_before_accept())
    assert before == 0
    assert after_cancel == 0

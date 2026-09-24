#!/usr/bin/env python3
"""Run ASR, then FA, on one canonical WAV from a YouTube URL.

The external step only builds a 16 kHz mono signed-16 PCM WAV. ASR receives
that file once. FA receives the same WAV and the unmodified ASR JSON once.
Chunking stays inside each server.

tests/outputs
- source.wav: PCM file sent to ASR and FA.
- asr.json: Original ASR response.
- alignment.json: FA response.
- timings.json: transcription_elapsed_sec and alignment_elapsed_sec.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any
from urllib import error, request


PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (project_root() / ".env")
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue

        val = value.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ[key] = val


def read_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value if value else default


load_dotenv()

# Same target as tests/test_asr_for_fa_input.py.
YOUTUBE_URL = "https://www.youtube.com/watch?v=ZYbKUUrbatI"

BASE_URL = read_env("STT_BASE_URL", "http://localhost:8080").rstrip("/")
FA_BASE_URL = read_env("FA_BASE_URL", "http://localhost:8090").rstrip("/")
MODEL = read_env("STT_MODEL", "qwen3-asr")
LANGUAGE = read_env("DEFAULT_LANGUAGE", "ko")
RETRIES = int(read_env("STT_HEALTH_RETRIES", "30"))
BACKOFF_SECONDS = float(read_env("STT_HEALTH_BACKOFF_SEC", "2"))
REQUEST_TIMEOUT_SECONDS = int(read_env("STT_REQUEST_TIMEOUT_SECONDS", "7200"))
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def http_json(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    req = request.Request(url=url, method=method, headers=headers or {})
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body) if body else {}
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Unexpected JSON payload type from {url}: {type(payload).__name__}"
                )
            return resp.status, payload
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[HTTP ERROR] status={exc.code} url={url}")
        print(body)
        raise
    except error.URLError as exc:
        print(f"[NETWORK ERROR] url={url} reason={exc.reason}")
        raise


def wait_until_ready(
    health_url: str,
    ready,
    unavailable_message: str,
) -> None:
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            status, payload = http_json("GET", health_url)
            if status == 200 and ready(payload):
                print(f"[INFO] health ready: {payload}")
                return
            print(f"[WARN] Unexpected /health payload: {payload}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[WAIT] attempt={attempt}/{RETRIES} failed, retrying...")

        if attempt < RETRIES:
            time.sleep(BACKOFF_SECONDS)

    print("[FAIL] /health did not become ready in time.")
    if last_error is not None:
        raise last_error
    raise RuntimeError(unavailable_message)


def wait_for_health() -> None:
    wait_until_ready(
        f"{BASE_URL}/health",
        lambda payload: payload.get("status") == "ok" and bool(payload.get("backend_reachable")),
        "STT health endpoint unavailable",
    )


def wait_for_fa_health() -> None:
    status, payload = http_json("POST", f"{FA_BASE_URL}/control/load")
    if status != 200 or payload.get("state") != "ready":
        raise RuntimeError(f"FA load did not become ready: {status} {payload}")
    wait_until_ready(
        f"{FA_BASE_URL}/health",
        lambda payload: payload.get("status") == "ok" and payload.get("model_loaded") is True,
        "FA health endpoint unavailable",
    )


def ensure_command(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Required command not found in PATH: {name}")


def download_audio(youtube_url: str, target_dir: Path) -> Path:
    if not youtube_url:
        raise RuntimeError(
            "YouTube URL is empty. Set YOUTUBE_URL at the top of this script."
        )

    ensure_command("yt-dlp")

    outtmpl = str(target_dir / "%(title).200s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--print",
        "after_move:filepath",
        "-o",
        outtmpl,
        youtube_url,
    ]
    print("[INFO] Downloading full audio with yt-dlp...")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if proc.stdout.strip():
            print(proc.stdout)
        if proc.stderr.strip():
            print(proc.stderr)
        raise RuntimeError(f"yt-dlp failed with exit code {proc.returncode}")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("yt-dlp did not report the downloaded file path.")

    audio_path = Path(lines[-1])
    if not audio_path.is_file():
        raise RuntimeError(f"Downloaded file not found: {audio_path}")

    safe_path = target_dir / f"audio{audio_path.suffix}"
    audio_path.rename(safe_path)
    print(f"[INFO] downloaded_audio={safe_path}")
    return safe_path


def normalize_pcm_wav(src: Path, dst: Path) -> dict[str, int]:
    ensure_command("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-vn",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}: {detail}")

    info = inspect_pcm_wav(dst)
    print(
        "[INFO] normalized_wav="
        f"{dst} sample_rate={info['sample_rate']} "
        f"num_samples={info['num_samples']}"
    )
    return info


def inspect_pcm_wav(path: Path) -> dict[str, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            num_samples = wav.getnframes()
            comptype = wav.getcomptype()
    except (wave.Error, EOFError) as exc:
        raise RuntimeError(f"Not a PCM WAV: {path}") from exc

    if (
        channels != PCM_CHANNELS
        or sample_width != PCM_SAMPLE_WIDTH
        or sample_rate != PCM_SAMPLE_RATE
        or comptype != "NONE"
        or num_samples <= 0
    ):
        raise RuntimeError(
            "Normalized WAV must be 16 kHz, mono, signed 16-bit PCM. "
            f"Got channels={channels}, sample_width={sample_width}, "
            f"sample_rate={sample_rate}, comptype={comptype}, "
            f"num_samples={num_samples}."
        )
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "num_samples": num_samples,
    }


def post_multipart(
    url: str,
    form_args: list[str],
    elapsed_label: str,
    failure_label: str,
) -> tuple[dict[str, Any], float]:
    ensure_command("curl")
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".out", delete=False) as resp_tmp:
        response_path = Path(resp_tmp.name)

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "-o",
                str(response_path),
                "-w",
                "%{http_code}",
                "-X",
                "POST",
                *form_args,
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - t0
        print(f"[INFO] {elapsed_label}={elapsed:.3f}")
        if proc.returncode != 0:
            raise RuntimeError(
                f"curl failed with exit code {proc.returncode}: {proc.stderr.strip()}"
            )

        status_code = proc.stdout.strip()
        raw = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        if status_code != "200":
            raise RuntimeError(f"{failure_label} failed with status={status_code}: {raw}")

        payload = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected response type from {url}: {type(payload).__name__}"
            )
        return payload, elapsed
    finally:
        if response_path.exists():
            response_path.unlink()


def request_transcription(audio_path: Path) -> tuple[dict[str, Any], float]:
    return post_multipart(
        f"{BASE_URL}/v1/audio/transcriptions",
        [
            "-F",
            f"file=@{audio_path}",
            "-F",
            f"model={MODEL}",
            "-F",
            f"language={LANGUAGE}",
            "-F",
            "response_format=verbose_json",
            "-F",
            "include_chunks=true",
        ],
        "transcription_elapsed_sec",
        "STT request",
    )


def request_alignment(wav_path: Path, asr_payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as payload_tmp:
        payload_path = Path(payload_tmp.name)
    payload_path.write_text(json.dumps(asr_payload, ensure_ascii=False), encoding="utf-8")
    try:
        return post_multipart(
            f"{FA_BASE_URL}/align",
            [
                "-F",
                f"file=@{wav_path};type=audio/wav",
                "-F",
                f"payload=<{payload_path}",
            ],
            "alignment_elapsed_sec",
            "FA request",
        )
    finally:
        if payload_path.exists():
            payload_path.unlink()


def as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{field} must be an integer, got {value!r}")
    return value


def chunk_text(chunk: dict[str, Any], index: int) -> str:
    text = chunk.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"chunks[{index}].text must be a string")
    return text


def require_chunk_languages(asr_payload: dict[str, Any]) -> list[dict[str, Any]]:
    chunks = asr_payload.get("chunks")
    if not isinstance(chunks, list):
        raise RuntimeError(
            "Response has no chunks. "
            "The request must use response_format=verbose_json and include_chunks=true."
        )
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise RuntimeError(f"Chunk is not an object: {chunk!r}")
        index = as_int(chunk.get("index"), f"chunks[{position}].index")
        text = chunk_text(chunk, index)
        if text.strip() == "":
            continue
        language = chunk.get("language")
        if not isinstance(language, str) or language.strip() == "":
            raise RuntimeError(
                f"chunks[{index}] has text but no language; FA will return 400"
            )
    return chunks


def assert_alignment(asr_chunks: list[dict[str, Any]], fa_body: dict[str, Any]) -> None:
    if fa_body.get("time_reference") != "audio_start":
        raise RuntimeError(f"time_reference={fa_body.get('time_reference')!r}")
    if fa_body.get("overlap_deduplicated") is not False:
        raise RuntimeError("overlap_deduplicated must stay false")

    fa_chunks = fa_body.get("chunks")
    if not isinstance(fa_chunks, list):
        raise RuntimeError("FA response has no chunks")
    if len(fa_chunks) != len(asr_chunks):
        raise RuntimeError(f"chunk count mismatch: asr={len(asr_chunks)} fa={len(fa_chunks)}")

    for src, dst in zip(asr_chunks, fa_chunks):
        if not isinstance(dst, dict):
            raise RuntimeError(f"FA chunk is not an object: {dst!r}")
        index = as_int(src.get("index"), "chunks.index")
        text = chunk_text(src, index)
        start_sample = as_int(src.get("start_sample"), f"chunks[{index}].start_sample")
        end_sample = as_int(src.get("end_sample"), f"chunks[{index}].end_sample")
        for field, expected in (
            ("index", index),
            ("start_sample", start_sample),
            ("end_sample", end_sample),
            ("text", text),
        ):
            if dst.get(field) != expected:
                raise RuntimeError(
                    f"chunks[{index}].{field} mismatch: asr={expected!r} fa={dst.get(field)!r}"
                )

        start = start_sample / PCM_SAMPLE_RATE
        end = end_sample / PCM_SAMPLE_RATE
        items = dst.get("items")
        if not isinstance(items, list):
            raise RuntimeError(f"chunks[{index}].items must be a list")
        if text.strip() == "":
            if dst.get("status") != "skipped_empty_text" or items != []:
                raise RuntimeError(f"chunks[{index}] empty text was not skipped")
            continue
        if dst.get("status") != "aligned":
            raise RuntimeError(f"chunks[{index}] status={dst.get('status')!r}")
        language = dst.get("language")
        if not isinstance(language, str) or language.strip() == "":
            raise RuntimeError(f"chunks[{index}] missing canonical language")
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str) or item["text"] == "":
                raise RuntimeError(f"chunks[{index}] empty aligned item")
            start_time = item.get("start_time")
            end_time = item.get("end_time")
            if not isinstance(start_time, (int, float)) or isinstance(start_time, bool):
                raise RuntimeError(f"chunks[{index}] invalid start_time")
            if not isinstance(end_time, (int, float)) or isinstance(end_time, bool):
                raise RuntimeError(f"chunks[{index}] invalid end_time")
            if not (start - 1e-3 <= float(start_time) <= float(end_time) <= end + 1e-3):
                raise RuntimeError(
                    f"chunks[{index}] item time {start_time}..{end_time} "
                    f"outside [{start}, {end}]"
                )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] saved={path}")


def main() -> int:
    try:
        print(f"[INFO] youtube_url={YOUTUBE_URL}")
        print(f"[INFO] stt_base_url={BASE_URL}")
        print(f"[INFO] fa_base_url={FA_BASE_URL}")
        print(f"[INFO] model={MODEL}")
        print(f"[INFO] language={LANGUAGE}")
        print("[INFO] response_format=verbose_json include_chunks=true")
        print(f"[INFO] output_dir={OUTPUT_DIR}")
        wait_for_health()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        source_wav = OUTPUT_DIR / "source.wav"
        with tempfile.TemporaryDirectory(prefix="yt-asr-fa-") as tmp_dir:
            downloaded = download_audio(YOUTUBE_URL, Path(tmp_dir))
            normalize_pcm_wav(downloaded, source_wav)

        asr_payload, transcription_elapsed_sec = request_transcription(source_wav)
        write_json(OUTPUT_DIR / "asr.json", asr_payload)
        asr_chunks = require_chunk_languages(asr_payload)

        wait_for_fa_health()
        aligned, alignment_elapsed_sec = request_alignment(source_wav, asr_payload)
        assert_alignment(asr_chunks, aligned)
        write_json(OUTPUT_DIR / "alignment.json", aligned)
        write_json(
            OUTPUT_DIR / "timings.json",
            {
                "transcription_elapsed_sec": transcription_elapsed_sec,
                "alignment_elapsed_sec": alignment_elapsed_sec,
            },
        )
        print("[SUCCESS] ASR then FA completed.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

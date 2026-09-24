#!/usr/bin/env python3
"""Build alignment inputs from the ASR chunk contract.

Keeps one 16 kHz mono signed-16 PCM WAV, sends it with include_chunks=true,
and writes each non-empty chunk as a WAV slice plus its pre-merge text.

outputs
- source.wav: PCM file sent to ASR.
- response.json: Original response include chunks.
- .txt named like chunks/chunk-0000.wav: sentences before merged with section be cut.
- manifest.json: start_sample, end_sample, list for skipped empty chunks.
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

# Set the test target directly here when you want to run the script.
YOUTUBE_URL = "https://www.youtube.com/watch?v=ZYbKUUrbatI"

BASE_URL = read_env("STT_BASE_URL", "http://localhost:8080").rstrip("/")
MODEL = read_env("STT_MODEL", "qwen3-asr")
LANGUAGE = read_env("DEFAULT_LANGUAGE", "ko")
RETRIES = int(read_env("STT_HEALTH_RETRIES", "30"))
BACKOFF_SECONDS = float(read_env("STT_HEALTH_BACKOFF_SEC", "2"))
REQUEST_TIMEOUT_SECONDS = int(read_env("STT_REQUEST_TIMEOUT_SECONDS", "7200"))
OUTPUT_DIR = Path(
    read_env("STT_OUTPUT_DIR", str(project_root() / "scripts" / "outputs"))
) / "alignment"


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


def wait_for_health() -> None:
    health_url = f"{BASE_URL}/health"
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            status, payload = http_json("GET", health_url)
            backend_reachable = bool(payload.get("backend_reachable"))
            if status == 200 and payload.get("status") == "ok" and backend_reachable:
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
    raise RuntimeError("STT health endpoint unavailable")


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


def request_transcription(audio_path: Path) -> dict[str, Any]:
    ensure_command("curl")
    url = f"{BASE_URL}/v1/audio/transcriptions"
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".out", delete=False) as resp_tmp:
        response_path = Path(resp_tmp.name)

    form_args = [
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
    ]

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
        print(f"[INFO] transcription_elapsed_sec={elapsed:.3f}")
        if proc.returncode != 0:
            raise RuntimeError(
                f"curl failed with exit code {proc.returncode}: {proc.stderr.strip()}"
            )

        status_code = proc.stdout.strip()
        raw = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        if status_code != "200":
            raise RuntimeError(f"STT request failed with status={status_code}: {raw}")

        payload = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected transcription response type: {type(payload).__name__}"
            )
        return payload
    finally:
        if response_path.exists():
            response_path.unlink()


def as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{field} must be an integer, got {value!r}")
    return value


def write_pcm_slice(
    src: Path,
    dst: Path,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
) -> None:
    frame_count = end_sample - start_sample
    with wave.open(str(src), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        reader.setpos(start_sample)
        frames = reader.readframes(frame_count)
    expected_bytes = frame_count * sample_width * channels
    if len(frames) != expected_bytes:
        raise RuntimeError(
            f"PCM slice [{start_sample}, {end_sample}) returned "
            f"{len(frames)} bytes, expected {expected_bytes}."
        )
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


def save_alignment_inputs(
    source_wav: Path,
    wav_info: dict[str, int],
    payload: dict[str, Any],
) -> None:
    audio = payload.get("audio")
    chunks = payload.get("chunks")
    if not isinstance(audio, dict) or not isinstance(chunks, list):
        raise RuntimeError(
            "Response has no audio/chunks. "
            "The request must use response_format=verbose_json and include_chunks=true."
        )

    sample_rate = as_int(audio.get("sample_rate"), "audio.sample_rate")
    channels = as_int(audio.get("channels"), "audio.channels")
    num_samples = as_int(audio.get("num_samples"), "audio.num_samples")
    if (
        sample_rate != wav_info["sample_rate"]
        or channels != wav_info["channels"]
        or num_samples != wav_info["num_samples"]
    ):
        raise RuntimeError(
            "Response audio header does not match the WAV that was uploaded. "
            f"local={wav_info} response={audio}"
        )

    chunks_dir = OUTPUT_DIR / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    response_path = OUTPUT_DIR / "response.json"
    response_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ordered = sorted(chunks, key=lambda item: as_int(item.get("index"), "chunks.index"))
    kept: list[dict[str, Any]] = []
    skipped_empty: list[int] = []

    for chunk in ordered:
        if not isinstance(chunk, dict):
            raise RuntimeError(f"Chunk is not an object: {chunk!r}")
        index = as_int(chunk.get("index"), "chunks.index")
        start_sample = as_int(chunk.get("start_sample"), "chunks.start_sample")
        end_sample = as_int(chunk.get("end_sample"), "chunks.end_sample")
        text = chunk.get("text")
        if not isinstance(text, str):
            raise RuntimeError(f"chunks[{index}].text must be a string")
        if not (0 <= start_sample < end_sample <= num_samples):
            raise RuntimeError(
                f"chunks[{index}] range [{start_sample}, {end_sample}) "
                f"is outside [0, {num_samples})"
            )
        if text == "":
            skipped_empty.append(index)
            print(f"[INFO] skip_empty index={index}")
            continue

        stem = f"chunk-{index:04d}"
        wav_path = chunks_dir / f"{stem}.wav"
        text_path = chunks_dir / f"{stem}.txt"
        write_pcm_slice(
            source_wav,
            wav_path,
            start_sample=start_sample,
            end_sample=end_sample,
            sample_rate=sample_rate,
        )
        text_path.write_text(text, encoding="utf-8")
        kept.append(
            {
                "index": index,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "wav": str(wav_path.relative_to(OUTPUT_DIR)),
                "text_path": str(text_path.relative_to(OUTPUT_DIR)),
                "text": text,
            }
        )
        print(
            f"[INFO] wrote index={index} "
            f"samples=[{start_sample}, {end_sample}) wav={wav_path.name}"
        )

    manifest = {
        "sample_rate": sample_rate,
        "channels": channels,
        "num_samples": num_samples,
        "source_wav": source_wav.name,
        "chunks": kept,
        "skipped_empty": skipped_empty,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] saved_response={response_path}")
    print(f"[INFO] saved_manifest={manifest_path}")
    print(f"[INFO] kept={len(kept)} skipped_empty={len(skipped_empty)}")


def main() -> int:
    try:
        print(f"[INFO] base_url={BASE_URL}")
        print(f"[INFO] model={MODEL}")
        print(f"[INFO] language={LANGUAGE}")
        print("[INFO] response_format=verbose_json include_chunks=true")
        print(f"[INFO] output_dir={OUTPUT_DIR}")
        wait_for_health()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        source_wav = OUTPUT_DIR / "source.wav"
        with tempfile.TemporaryDirectory(prefix="yt-alignment-input-") as tmp_dir:
            downloaded = download_audio(YOUTUBE_URL, Path(tmp_dir))
            wav_info = normalize_pcm_wav(downloaded, source_wav)

        payload = request_transcription(source_wav)
        save_alignment_inputs(source_wav, wav_info, payload)
        print("[SUCCESS] Alignment inputs written.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
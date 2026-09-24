#!/usr/bin/env python3
"""Measure load, inference, and unload on a real GPU.

Runs the API in this process so torch.cuda.memory_allocated matches the
server. Prints status latency and memory samples. Exits 1 when status does
not answer within 2 seconds during load or inference.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import torch
import uvicorn

os.environ.setdefault("FA_MODEL_PATH", "/models/Qwen3-ForcedAligner-0.6B")
os.environ.setdefault("FA_DTYPE", "bfloat16")

from app.main import app  # noqa: E402

PORT = 8091
BASE = f"http://127.0.0.1:{PORT}"
STATUS_LIMIT_SEC = 2.0
ROUNDS = 3
OUTPUT = Path(os.environ.get("FA_CHECK_OUTPUT", "/outputs/gpu_lifecycle.md"))


def memory_allocated() -> int:
    torch.cuda.synchronize()
    return int(torch.cuda.memory_allocated())


def request(method: str, path: str, body: bytes | None = None, timeout: float = 600) -> tuple[int, dict]:
    data = body if body not in (None, b"") else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode()
        return exc.code, json.loads(payload) if payload else {}


def wait_status() -> dict:
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            status, body = request("GET", "/control/status", timeout=2)
        except Exception:
            time.sleep(0.05)
            continue
        if status == 200:
            return body
    raise RuntimeError("control status did not become available")


def status_latency(stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        started = time.perf_counter()
        try:
            request("GET", "/control/status", timeout=STATUS_LIMIT_SEC)
            samples.append(time.perf_counter() - started)
        except Exception:
            samples.append(time.perf_counter() - started)
        time.sleep(0.05)


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00" * (16000 * 2))


def align(path: Path) -> tuple[int, dict]:
    boundary = "----fa-gpu-check"
    payload = json.dumps(
        {
            "audio": {"sample_rate": 16000, "channels": 1, "num_samples": 16000},
            "chunks": [
                {
                    "index": 0,
                    "start_sample": 0,
                    "end_sample": 16000,
                    "text": "안녕하세요",
                    "language": "ko",
                }
            ],
        }
    )
    body = "\r\n".join(
        [
            f"--{boundary}",
            'Content-Disposition: form-data; name="file"; filename="canonical.wav"',
            "Content-Type: audio/wav",
            "",
            path.read_bytes().decode("latin1"),
            f"--{boundary}",
            'Content-Disposition: form-data; name="payload"',
            "",
            payload,
            f"--{boundary}--",
            "",
        ]
    ).encode("latin1")
    req = urllib.request.Request(BASE + "/align", data=body, method="POST")
    req.add_header("content-type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        return exc.code, json.loads(raw) if raw else {}


def track(label: str, stop: threading.Event, latencies: list[float], peaks: list[int]) -> None:
    stop.clear()
    latencies.clear()
    watcher = threading.Thread(target=status_latency, args=(stop, latencies), daemon=True)
    watcher.start()
    peak = memory_allocated()
    while not stop.is_set():
        peak = max(peak, memory_allocated())
        time.sleep(0.05)
    peaks.append(peak)
    watcher.join(timeout=1)


def write_report(rounds: list[dict], summary: dict | None, error: str | None) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GPU lifecycle 테스트 결과",
        "",
        "같은 프로세스에서 load, align, unload를 세 번 반복한 기록이다.",
        "",
    ]
    if error:
        lines.extend([f"실패: {error}", ""])
    for item in rounds:
        lines.extend(
            [
                f"## round {item['round']} {item['step']}",
                "",
                "```json",
                json.dumps(item, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    if summary is not None:
        lines.extend(
            [
                "## summary",
                "",
                "```json",
                json.dumps(summary, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning"),
        daemon=True,
    )
    thread.start()
    initial = wait_status()
    print("startup", json.dumps(initial), "allocated", memory_allocated())
    if initial["state"] != "unloaded":
        raise SystemExit(f"startup state is {initial['state']}")

    wav_path = Path("/tmp/fa-gpu-check.wav")
    write_wav(wav_path)
    stop = threading.Event()
    latencies: list[float] = []
    load_peaks: list[int] = []
    infer_peaks: list[int] = []
    residuals: list[int] = []
    rounds: list[dict] = []

    for round_index in range(ROUNDS):
        before = memory_allocated()
        stop.clear()
        latencies.clear()
        watcher = threading.Thread(target=status_latency, args=(stop, latencies), daemon=True)
        watcher.start()
        load_peak = before

        def watch_load() -> None:
            nonlocal load_peak
            while not stop.is_set():
                load_peak = max(load_peak, memory_allocated())
                time.sleep(0.05)

        sampler = threading.Thread(target=watch_load, daemon=True)
        sampler.start()
        started = time.perf_counter()
        status, body = request("POST", "/control/load", b"{}")
        stop.set()
        watcher.join(timeout=1)
        sampler.join(timeout=1)
        load_sec = time.perf_counter() - started
        worst = max(latencies) if latencies else STATUS_LIMIT_SEC
        load_row = {
            "round": round_index,
            "step": "load",
            "http_status": status,
            "body": body,
            "seconds": round(load_sec, 3),
            "status_max_sec": round(worst, 3),
            "allocated_before": before,
            "allocated_after": memory_allocated(),
            "load_peak": load_peak,
        }
        rounds.append(load_row)
        print(
            f"round {round_index} load",
            status,
            body.get("state"),
            f"sec={load_sec:.2f}",
            f"status_max={worst:.3f}",
            f"allocated_before={before}",
            f"allocated_after={load_row['allocated_after']}",
            f"load_peak={load_peak}",
        )
        if status != 200 or worst >= STATUS_LIMIT_SEC:
            write_report(rounds, None, "load did not keep status responsive")
            raise SystemExit("load did not keep status responsive")
        load_peaks.append(load_peak)

        stop = threading.Event()
        infer_latencies: list[float] = []
        infer_peak = memory_allocated()
        watcher = threading.Thread(target=status_latency, args=(stop, infer_latencies), daemon=True)
        watcher.start()

        def watch_infer() -> None:
            nonlocal infer_peak
            while not stop.is_set():
                infer_peak = max(infer_peak, memory_allocated())
                time.sleep(0.05)

        sampler = threading.Thread(target=watch_infer, daemon=True)
        sampler.start()
        align_status, align_body = align(wav_path)
        stop.set()
        watcher.join(timeout=1)
        sampler.join(timeout=1)
        infer_worst = max(infer_latencies) if infer_latencies else STATUS_LIMIT_SEC
        align_row = {
            "round": round_index,
            "step": "align",
            "http_status": align_status,
            "chunk_status": align_body.get("chunks", [{}])[0].get("status") if align_status == 200 else None,
            "body": align_body if align_status != 200 else {"chunks": align_body.get("chunks")},
            "status_max_sec": round(infer_worst, 3),
            "infer_peak": infer_peak,
        }
        rounds.append(align_row)
        print(
            f"round {round_index} align",
            align_status,
            align_row["chunk_status"] if align_status == 200 else align_body,
            f"status_max={infer_worst:.3f}",
            f"infer_peak={infer_peak}",
        )
        if align_status != 200 or infer_worst >= STATUS_LIMIT_SEC:
            write_report(rounds, None, "inference did not keep status responsive")
            raise SystemExit("inference did not keep status responsive")
        infer_peaks.append(infer_peak)

        status, body = request("POST", "/control/unload")
        residual = memory_allocated()
        residuals.append(residual)
        rounds.append(
            {
                "round": round_index,
                "step": "unload",
                "http_status": status,
                "body": body,
                "allocated": residual,
            }
        )
        print(
            f"round {round_index} unload",
            status,
            body.get("state"),
            body.get("residency"),
            f"allocated={residual}",
        )
        if status != 200 or body.get("state") != "unloaded":
            write_report(rounds, None, "unload did not return to baseline")
            raise SystemExit("unload did not return to baseline")

    grew = residuals[-1] > residuals[0] + (32 * 1024 * 1024)
    summary = {
        "load_peak": max(load_peaks),
        "infer_peak": max(infer_peaks),
        "residuals": residuals,
        "residual_grew": grew,
    }
    print("summary", json.dumps(summary))
    write_report(rounds, summary, "allocated memory grew across unload rounds" if grew else None)
    if grew:
        raise SystemExit("allocated memory grew across unload rounds")


if __name__ == "__main__":
    main()

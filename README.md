# qwen-fa-docker-config

Single-container Docker deployment for Qwen3 ForcedAligner. The process starts without loading weights. `POST /control/load` loads `Qwen3-ForcedAligner-0.6B`, then `POST /align` aligns one canonical WAV and an ASR chunk payload.

The runtime is one Compose service:

- `fa-api`: FastAPI server for WAV checks, chunk slicing, in-request microbatching, and file-absolute timestamps

Calling ASR, assembling subtitles, removing overlap duplicates, and swapping runtimes stay outside this container.

## Architecture

```text
[client]
   |
   v
[fa-api :8090]
   |
   +--> Qwen3-ForcedAligner-0.6B on cuda:0
```

The client is expected to already have a 16 kHz mono signed-16 PCM WAV and the unmodified ASR JSON for that file. This server reads `audio` and `chunks` from that JSON and ignores every other field.

## Project structure

- `docker-compose.yml`: runtime definition for the FA API container
- `prepare-inferswap`: starts the container only when it does not already exist
- `.env.example`: ports, model path, dtype, batch size, and chunk limit
- `fa-api/`: image and FastAPI app
- `tests/test_contract.py`, `tests/test_alignment_mapping.py`: contract tests that run without model weights
- `tests/test_asr_then_fa.py`: downloads YouTube audio, calls ASR, then calls this API
- `tests/test_asr_for_fa_input.py`: downloads YouTube audio, calls ASR, and writes chunk slices without calling FA

## Requirements

- NVIDIA GPU
- NVIDIA Container Toolkit
- Docker / Docker Compose

Contract tests need Python and the app dependencies. The YouTube scripts also need `yt-dlp`, `ffmpeg`, and `curl` on the host, plus a running ASR server.

## Model download

Weights are mounted read-only and are not copied into the image. Hub access inside the container is disabled.

```bash
mkdir -p ../models/stt/hf

hf download Qwen/Qwen3-ForcedAligner-0.6B \
  --local-dir ../models/stt/hf/Qwen3-ForcedAligner-0.6B
```

If that directory already exists, no extra download is needed.

## Quick start

1. Copy the example environment file.

```bash
cp .env.example .env
```

2. Review the model directory and GPU id in `.env`.

```env
MODEL_HOST_DIR=../models/stt/hf
FA_MODEL_PATH=/models/Qwen3-ForcedAligner-0.6B
FA_GPU_DEVICE=0
```

`MODEL_HOST_DIR` is relative to the Compose file. Inside the container the model device is always `cuda:0`. `FA_GPU_DEVICE` selects the host GPU.

3. Build and start the container.

```bash
docker compose up --build -d
```

4. Confirm the process is up, then load the model.

```bash
curl --fail-with-body http://localhost:8090/control/status
curl --fail-with-body -X POST http://localhost:8090/control/load
curl --fail-with-body http://localhost:8090/health
```

Right after start, status is:

```json
{"state": "unloaded", "residency": "not_resident", "active_requests": 0, "last_error": null}
```

`/health` and `POST /align` return `503` until load finishes:

```json
{"status": "loading", "model_loaded": false}
```

A finished load returns `200` with `"state": "ready"` and `"residency": "resident"`. `/health` is then `200`:

```json
{"status": "ok", "model_loaded": true}
```

`POST /control/load` and `POST /control/unload` accept an empty body or `{}`. Calling load again when the model is ready does not load it a second time. Unload returns to `unloaded` / `not_resident`. Missing weights or a language-list mismatch stay in the process as `state=failed` on the load response. They do not exit the process.

The Compose healthcheck passes when `GET /control/status` returns `200`. It does not wait for the model.

`./prepare-inferswap` runs `docker compose up -d` only when the `fa-api` container does not exist. It does not load the model. If the container already exists and status returns `200`, it leaves that container alone. If the container exists but status fails, it exits non-zero and does not restart it.

## API example

Send the same canonical WAV that was sent to ASR, plus the ASR JSON unchanged:

```bash
curl --fail-with-body -X POST http://localhost:8090/align \
  -F 'file=@canonical.wav;type=audio/wav' \
  -F 'payload=<asr-result.json'
```

A single chunk is enough to exercise the same API. Build the WAV first if you do not already have one:

```bash
ffmpeg -i source_audio.m4a -vn -ar 16000 -ac 1 -c:a pcm_s16le canonical.wav

curl --fail-with-body -X POST http://localhost:8090/align \
  -F 'file=@canonical.wav;type=audio/wav' \
  -F 'payload={"audio":{"sample_rate":16000,"channels":1,"num_samples":16000},"chunks":[{"index":0,"start_sample":0,"end_sample":16000,"text":"안녕하세요","language":"ko"}]}'
```

## API behavior

`POST /align` is `multipart/form-data` with `file` and `payload`.

1. Store the upload under a `fa-align-*` temporary directory.
2. Reject a WAV that is not signed 16-bit PCM, 16,000 Hz, mono, with at least one frame. This server does not resample, mix channels, or trim silence.
3. Require `audio.sample_rate` `16000`, `audio.channels` `1`, and `audio.num_samples` equal to the WAV frame count.
4. Check each chunk's `index`, half-open `[start_sample, end_sample)`, and length. Indexes must be unique and `>= 0`. The span must lie inside `[0, num_samples)`.
5. Skip chunks whose `text` is empty or whitespace. Send the rest to the model in microbatches of `FA_BATCH_SIZE`.
6. Clamp each model timestamp to that chunk, then add `start_sample / 16000` once.
7. Return every input chunk in its original position and delete the temporary directory.

The whole file has no duration cap. `FA_MAX_CHUNK_SECONDS` applies to each chunk. The default and the maximum are 180. A chunk of that length is accepted. A longer chunk is `400`.

A skipped chunk stays in place with `status` `skipped_empty_text`, `language` `null`, and `items` `[]`. Its original `text` is preserved. A non-empty chunk with a missing or unsupported language is `400`.

Language aliases are normalized before the model call. The canonical name is what comes back on an aligned chunk. Supported names are Korean, Japanese, Chinese, English, Cantonese, French, German, Italian, Portuguese, Russian, and Spanish. Accepted aliases include `ko`, `ja`/`jp`, `zh`/`zh-cn`/`zh-tw`, `en`, `yue`, `fr`, `de`, `it`, `pt`, `ru`, and `es`, matched case-insensitively after trimming.

`items[].start_time` and `items[].end_time` are seconds from the start of the uploaded WAV. Do not add `start_sample / sample_rate` again. A reversed timestamp fails the request with `500`. Item text that is empty is dropped.

Overlapping chunks are left as aligned. `overlap_deduplicated` is always `false`. Deduplication, sentence reassembly, and SRT/VTT generation belong to the caller.

```json
{
  "audio": {"sample_rate": 16000, "channels": 1, "num_samples": 16000},
  "time_reference": "audio_start",
  "overlap_deduplicated": false,
  "chunks": [
    {
      "index": 0,
      "start_sample": 0,
      "end_sample": 16000,
      "text": "안녕하세요",
      "language": "Korean",
      "status": "aligned",
      "items": [
        {"text": "안녕", "start_time": 0.1, "end_time": 0.4}
      ]
    }
  ]
}
```

Invalid WAV, metadata mismatch, bad sample spans, duplicate indexes, and unsupported languages are `400` with `{"detail":"..."}`. A model failure, including CUDA OOM, is `500`. The server does not return a partial result and does not shrink the batch and retry.

Concurrent requests do not overlap on the model. Batching stays inside the one request that holds the lock. Cancelling the HTTP request does not stop the worker already running. The lock and the temporary directory are released only after that worker finishes. A request rejected before inference deletes its temporary directory immediately.

Match the ASR server's `CHUNK_SECONDS` to `FA_MAX_CHUNK_SECONDS`. This server does not call ASR.

## Environment variables

The main settings are in `.env.example`.

| Variable | Default | Role |
|---|---|---|
| `FA_API_PORT` | `8090` | Host port mapped to container port 8090 |
| `MODEL_HOST_DIR` | `../models/stt/hf` | Host model root, mounted at `/models` |
| `FA_MODEL_PATH` | `/models/Qwen3-ForcedAligner-0.6B` | Model directory inside the container |
| `FA_DTYPE` | `bfloat16` | `bfloat16`, `float16`, or `float32` |
| `FA_BATCH_SIZE` | `4` | Microbatch size inside one request, integer `>= 1` |
| `FA_MAX_CHUNK_SECONDS` | `180` | Per-chunk limit, integer from 1 to 180 |
| `FA_GPU_DEVICE` | `0` | Host GPU id |
| `LOG_LEVEL` | `info` | Uvicorn log level |

`FA_BATCH_SIZE=1` is valid. A larger batch can reduce runtime and increase GPU memory.

The YouTube scripts read extra variables from the environment or from `.env`. They are not part of the container config.

- `STT_BASE_URL`: ASR base URL, default `http://localhost:8080`
- `FA_BASE_URL`: this API, default `http://localhost:8090` (`tests/test_asr_then_fa.py` only)
- `STT_MODEL`: ASR model name, default `qwen3-asr`
- `DEFAULT_LANGUAGE`: language sent to ASR, default `ko`
- `STT_HEALTH_RETRIES`: health retries, default `30`
- `STT_HEALTH_BACKOFF_SEC`: seconds between health retries, default `2`
- `STT_REQUEST_TIMEOUT_SECONDS`: request timeout, default `7200`
- `STT_OUTPUT_DIR`: used only by `tests/test_asr_for_fa_input.py`

## Tests

### Contract tests

These run without model weights:

```bash
python -m pytest tests/test_contract.py tests/test_alignment_mapping.py
```

They cover invalid WAV input, metadata mismatch, the chunk-length boundary, language normalization, empty text, microbatch grouping, the global time offset, serialized concurrent requests, lock plus temp-file behavior on cancellation, and model load/unload status.

### ASR, then FA

`tests/test_asr_then_fa.py` downloads one YouTube video, writes a canonical WAV, sends that file to ASR with `response_format=verbose_json` and `include_chunks=true`, loads this model with `POST /control/load`, then sends the same WAV and the unmodified ASR JSON to this API. The host needs `yt-dlp`, `ffmpeg`, and `curl`. ASR must already be up, and this container must be up. The script loads the model itself.

```bash
python3 tests/test_asr_then_fa.py
```

Set the YouTube URL at the top of `tests/test_asr_then_fa.py`. Outputs go to `tests/outputs/`, which is gitignored.

- `source.wav`: PCM file sent to ASR and FA
- `asr.json`: original ASR response
- `alignment.json`: FA response
- `timings.json`: `transcription_elapsed_sec` and `alignment_elapsed_sec`

### Alignment inputs only

`tests/test_asr_for_fa_input.py` stops after ASR. It does not call this API. The host needs `yt-dlp`, `ffmpeg`, and `curl`, and the ASR server must be up.

```bash
python3 tests/test_asr_for_fa_input.py
```

Set the YouTube URL at the top of that file. The default output directory is `scripts/outputs/alignment/` (`STT_OUTPUT_DIR` plus `alignment`).

- `source.wav`: PCM file uploaded to ASR
- `response.json`: full `verbose_json` body, including `audio` and `chunks`
- `chunks/chunk-NNNN.wav`: PCM slice for `[start_sample, end_sample)`
- `chunks/chunk-NNNN.txt`: that chunk's pre-merge `text`. Empty `text` is not written
- `manifest.json`: sample counts, kept chunk paths, and `skipped_empty` indexes

To align those inputs, load the model, then send `source.wav` and `response.json` together to `POST /align`. Do not add the chunk offset to the returned times.

### GPU lifecycle check

`tests/check_gpu_lifecycle.py` loads, aligns one second of audio, and unloads three times inside the image. It records `torch.cuda.memory_allocated` and checks that `/control/status` keeps answering during load and alignment. Run it with the image's Python on a GPU, with the model mounted at `FA_MODEL_PATH`.

## Notes

- The image is `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` with `torch==2.8.0` from the cu128 wheel and `qwen-asr==0.0.6`. Load checks that the model's supported languages match the list above. Uvicorn uses one worker.
- Only `FA_API_PORT` is published. The default is `8090`.
- `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are set in Compose, so the container will not download weights at startup.

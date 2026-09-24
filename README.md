# Qwen3 ForcedAligner API

Torch 기반 Qwen3 ForcedAligner HTTP 서버입니다. 서버는 시작 시 모델을 한 번 로드하고, canonical WAV 전체와 ASR 청크 메타데이터를 `POST /align` 한 번으로 받습니다. 구간 추출, 요청 내부 microbatch 정렬, 전체 파일 기준 시간 보정은 FA 안에서 수행합니다.

ASR 호출, 자막 조립, 겹침 중복 제거, InferSwap 연동은 이 서버의 범위가 아닙니다. InferSwap은 이후 단계에서 런타임 전환을 담당합니다.

## 실행

```bash
cp .env.example .env
```

`MODEL_HOST_DIR`는 Compose 파일 기준의 호스트 모델 루트입니다. 기본값은 `../models/stt/hf`이고, 그 안에 `Qwen3-ForcedAligner-0.6B`가 있어야 합니다. 가중치는 이미지에 복사하지 않고 `/models`로 읽기 전용 마운트합니다.

```bash
docker compose up --build -d
curl --fail-with-body http://localhost:8090/health
```

모델 로드가 끝나면 `/health`가 `200`과 `{"status":"ok","model_loaded":true}`를 반환합니다. 로드에 실패하면 프로세스가 종료됩니다.

## 입력 WAV

FA에 보내는 파일은 ASR에 보낸 것과 같은 canonical WAV입니다.

- WAV 컨테이너
- 16,000 Hz
- mono
- signed 16-bit PCM

```bash
ffmpeg -i source_audio.m4a -vn -ar 16000 -ac 1 -c:a pcm_s16le canonical.wav
```

FA는 리샘플링, 채널 변환, 무음 제거를 하지 않습니다. 30분 파일 전체를 한 요청으로 보낼 수 있습니다. 180초 제한은 파일 전체가 아니라 ASR 청크 하나하나에 적용됩니다. 180초 청크는 허용하고, 그보다 긴 청크는 거부합니다.

## 정렬 요청

ASR에는 전체 WAV를 `response_format=verbose_json`, `include_chunks=true`로 한 번 보냅니다. 그 응답 JSON을 수정하지 않고 FA의 `payload`로 보냅니다. FA는 `audio`와 `chunks`만 읽고 나머지 필드는 무시합니다.

```bash
curl --fail-with-body -X POST http://localhost:8090/align \
  -F 'file=@canonical.wav;type=audio/wav' \
  -F 'payload=<asr-result.json'
```

청크가 하나인 입력으로도 같은 API를 시험할 수 있습니다.

```bash
curl --fail-with-body -X POST http://localhost:8090/align \
  -F 'file=@canonical.wav;type=audio/wav' \
  -F 'payload={"audio":{"sample_rate":16000,"channels":1,"num_samples":16000},"chunks":[{"index":0,"start_sample":0,"end_sample":16000,"text":"안녕하세요","language":"ko"}]}'
```

`chunks[].text`가 비어 있으면 그 청크는 모델에 들어가지 않고 `skipped_empty_text`로 원래 위치에 남습니다. 비어 있지 않은 전사의 언어가 없거나 지원 목록 밖이면 요청은 400입니다.

## 응답 시간

`items[].start_time`과 `items[].end_time`은 청크 시작이 아니라 업로드한 WAV 전체의 시작을 0초로 둔 시각입니다. 서버가 `start_sample / 16000`을 한 번 더합니다. 호출자가 같은 오프셋을 다시 더하지 않습니다.

인접 청크가 겹치면 각 청크의 정렬 항목이 그대로 남습니다. 응답의 `overlap_deduplicated`는 `false`입니다. 중복 제거, 문장 재조립, SRT/VTT 생성은 FA 응답을 받은 쪽에서 처리합니다.

## 배치와 메모리

`FA_BATCH_SIZE`는 한 요청 안의 microbatch 크기입니다. 기본값은 4이고, 1도 동작합니다. 요청이 여러 개 들어와도 모델 실행은 한 번에 하나이며, 그 요청 안에서만 배치합니다. 배치를 키우면 처리 시간이 줄 수 있지만 GPU 메모리도 같이 늘어납니다. CUDA OOM을 포함한 모델 실행 오류는 500이고, 일부 청크만 성공한 응답으로 바꾸거나 배치를 자동으로 줄여 재시도하지 않습니다.

`FA_DTYPE`은 `bfloat16`, `float16`, `float32` 중 하나입니다. 기본값은 `bfloat16`입니다.

## 구성

| 구성 | 역할 |
|---|---|
| 외부 서비스 | canonical WAV 생성·보관, ASR 다음 FA 호출, 정렬 이후 후처리 |
| ASR | 청크 전사와 `start_sample`/`end_sample` 반환 |
| 이 FA 서버 | WAV 검증, 구간 추출, 배치 정렬, 전체 파일 기준 시간 반환 |
| InferSwap | 이후 단계. 모델 상주와 런타임 전환 |

컨테이너 안에서 모델 장치는 `cuda:0`입니다. 호스트 GPU 선택은 Compose의 `FA_GPU_DEVICE`가 담당합니다.

## 검증

모델 없이 실행한 계약 테스트 20개가 통과했습니다. 잘못된 WAV, 메타데이터 불일치, 180초 청크 경계, 언어 정규화, 빈 전사, microbatch 대응, global offset, 동시 요청 직렬화, 취소 시 락과 임시 파일 유지가 여기 포함됩니다.

이미지는 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`와 `torch==2.8.0+cu128`로 빌드했습니다. 컨테이너에서 `Qwen3ForcedAligner` import와 CUDA 사용이 확인되었고, `vllm`은 설치되어 있지 않습니다.

RTX 3090에서 로컬 `Qwen3-ForcedAligner-0.6B`를 로드한 뒤 `/health`는 `200`이었습니다. 3초 WAV 한 청크는 `안녕하세요`를 `안녕`, `하세요`로 정렬했고, 시각은 파일 시작 기준이었습니다. 3초 청크 5개를 요청 한 번으로 처리하는 동안 `/health`는 `200`을 유지했고, 응답 후 `fa-align-` 임시 디렉터리는 남아 있지 않았습니다.

같은 5청크 입력을 warmup 뒤에 다시 측정하면 batch 1은 약 0.31초, peak 1791 MiB였고 batch 4는 약 0.15–0.18초, peak 1795 MiB였습니다. batch 1은 `align()`을 1개씩 5번, batch 4는 4개와 1개로 호출했습니다. 두 결과의 청크 순서와 항목 텍스트는 같았고, 시각은 각 청크 범위 안에 있었습니다. 부동소수점 결과의 bitwise 일치는 요구하지 않습니다.

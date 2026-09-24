#!/usr/bin/env bash
set -euo pipefail
exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8090 \
  --workers 1 \
  --log-level "${LOG_LEVEL:-info}"

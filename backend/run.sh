#!/usr/bin/env bash
# Start the FastAPI backend.
#
# Usage: ./run.sh [--host HOST] [--port PORT] [--no-reload]
#   --host       interface to bind (default 127.0.0.1)
#   --port       port to listen on (default 8000)
#   --no-reload  disable auto-reload on code changes (reload is on by default)
#
# API docs: http://<host>:<port>/docs
set -euo pipefail
cd "$(dirname "$0")"

HOST=127.0.0.1
PORT=8000
RELOAD=(--reload)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --no-reload) RELOAD=(); shift ;;
    -h|--help) sed -n '2,9p' "$(basename "$0")" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is not installed. See https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "note: no backend/.env - Claude/OpenAI need API keys (copy .env.example). Ollama works without." >&2
fi

uv sync --quiet

echo "Backend: http://$HOST:$PORT  (docs: http://$HOST:$PORT/docs)"
exec uv run uvicorn ocr_pipeline.api.app:app --host "$HOST" --port "$PORT" "${RELOAD[@]}"

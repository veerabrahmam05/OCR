#!/usr/bin/env bash
# Start the Next.js frontend.
#
# Usage: ./run.sh [dev|prod] [--port PORT]
#   dev   development server with hot reload (default)
#   prod  production build, then serve it
#   --port  port to listen on (default 3000)
#
# The backend URL comes from .env.local (NEXT_PUBLIC_API_URL, default http://127.0.0.1:8000).
# It is baked in at build time: restart dev / rebuild prod after changing it.
set -euo pipefail
cd "$(dirname "$0")"

MODE=dev
PORT=3000

while [[ $# -gt 0 ]]; do
  case "$1" in
    dev|prod) MODE="$1"; shift ;;
    --port) PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$(basename "$0")" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
  esac
done

if ! command -v pnpm >/dev/null 2>&1; then
  echo "error: pnpm is not installed. Run: npm install -g pnpm" >&2
  exit 1
fi

if [[ ! -d node_modules ]]; then
  pnpm install --frozen-lockfile
fi

if [[ ! -f .env.local ]]; then
  echo "note: no frontend/.env.local - using the backend at http://127.0.0.1:8000 (copy .env.example to change)." >&2
fi

echo "Frontend: http://localhost:$PORT  ($MODE)"
if [[ "$MODE" == prod ]]; then
  pnpm build
  exec pnpm start --port "$PORT"
fi
exec pnpm dev --port "$PORT"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -s .voice_password ]]; then
  python3 - <<'PY'
import secrets
from pathlib import Path
p = Path(".voice_password")
p.write_text(secrets.token_urlsafe(18), encoding="utf-8")
p.chmod(0o600)
PY
fi

export HERMES_VOICE_PASSWORD="$(cat .voice_password)"
# Default is now 0: authentication ON. This previously defaulted to 1, which made
# VOICE_PUBLIC_NO_AUTH true and bypassed the password entirely — the voice room was
# reachable through the public ngrok tunnel with no credentials at all, on a host where
# Hermes runs with approvals disabled. Set to 1 only for a deliberate, temporary test.
export HERMES_VOICE_PUBLIC_NO_AUTH="${HERMES_VOICE_PUBLIC_NO_AUTH:-0}"
export HERMES_VOICE_STT_PROVIDER="${HERMES_VOICE_STT_PROVIDER:-whisper_server}"
export HERMES_VOICE_STT_FALLBACKS="${HERMES_VOICE_STT_FALLBACKS:-local}"
export HERMES_WHISPER_SERVER_URL="${HERMES_WHISPER_SERVER_URL:-http://127.0.0.1:12712/inference}"
export HERMES_VOICE_FAST_PROVIDER="${HERMES_VOICE_FAST_PROVIDER:-xai}"
export HERMES_OLLAMA_URL="${HERMES_OLLAMA_URL:-http://127.0.0.1:11434}"
export HERMES_OLLAMA_VOICE_MODEL="${HERMES_OLLAMA_VOICE_MODEL:-qwen3:4b-instruct}"
export HERMES_OLLAMA_KEEP_ALIVE="${HERMES_OLLAMA_KEEP_ALIVE:-30m}"

if [[ ! -f certs/voice-room.crt || ! -f certs/voice-room.key ]]; then
  ./make_cert.sh
fi

exec "${HERMES_PYTHON:-python3}" server.py --host 0.0.0.0 --port 8765

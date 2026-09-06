#!/bin/zsh
set -euo pipefail
GROK_BIN="${VOICE_GROK_BIN:-$HOME/.grok/bin/grok}"
CWD="${HERMES_VOICE_CWD:-$HOME/Desktop/home screen folders/MY Business}"
SID="${1:-}"

if [[ ! -x "$GROK_BIN" ]]; then
  echo "Grok CLI not found: $GROK_BIN" >&2
  exit 1
fi

cd "$CWD"
if [[ -n "$SID" ]]; then
  exec "$GROK_BIN" --resume "$SID"
else
  exec "$GROK_BIN"
fi

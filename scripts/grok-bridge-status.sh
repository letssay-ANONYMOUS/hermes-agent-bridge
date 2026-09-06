#!/bin/zsh
set -euo pipefail
GROK_BIN="${VOICE_GROK_BIN:-$HOME/.grok/bin/grok}"
ACTIVE="$HOME/.grok/active_sessions.json"
LEADER="$HOME/.grok/leader.sock"
SESSIONS="$HOME/.grok/sessions"

echo "=== Grok Build bridge status ==="
if [[ -x "$GROK_BIN" ]]; then
  echo "grok: OK ($GROK_BIN)"
  "$GROK_BIN" --version 2>/dev/null || true
else
  echo "grok: MISSING at $GROK_BIN"
  exit 1
fi

if [[ -S "$LEADER" ]]; then
  echo "leader.sock: present"
else
  echo "leader.sock: missing (ok if no Grok TUI open)"
fi

if [[ -f "$ACTIVE" ]]; then
  echo "active_sessions:"
  python3 - <<'PY'
import json, os
from pathlib import Path
p = Path.home() / ".grok" / "active_sessions.json"
try:
    data = json.loads(p.read_text())
except Exception as e:
    print("  (unreadable)", e)
    raise SystemExit(0)
if not data:
    print("  (none)")
for item in data or []:
    sid = item.get("session_id", "?")
    pid = item.get("pid")
    cwd = item.get("cwd") or ""
    alive = False
    if pid is not None:
        try:
            os.kill(int(pid), 0)
            alive = True
        except Exception:
            alive = False
    flag = "LIVE" if alive else "stale"
    print(f"  [{flag}] {sid} pid={pid} cwd={cwd}")
PY
else
  echo "active_sessions: none"
fi

if [[ -d "$SESSIONS" ]]; then
  count=$(find "$SESSIONS" -name summary.json 2>/dev/null | wc -l | tr -d ' ')
  echo "session summaries on disk: $count"
else
  echo "sessions dir missing"
fi

# voice room health
if curl -sk --max-time 2 https://127.0.0.1:8765/api/health >/dev/null 2>&1 || curl -s --max-time 2 http://127.0.0.1:8765/api/health >/dev/null 2>&1; then
  echo "voice_room :8765: reachable"
else
  echo "voice_room :8765: not reachable (start with ./start.sh)"
fi
echo "=== done ==="

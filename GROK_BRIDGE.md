# Grok Build remote bridge (phone web app)

## What this is
The **local Hermes Voice Room** (`server.py` on :8765) is the only bridge.
It does **not** use Hermes.app (macOS).

```
Phone → voice_room → ~/.grok/bin/grok  (full tools, --always-approve)
```

## Features
- List Grok CLI sessions (incl. LIVE Terminal sessions)
- **Attach** existing session → writable web thread (full remote control)
- **New** Grok session from phone
- Stream Grok output into the web chat
- Optional: open same session in Terminal (`open-terminal` API / scripts)

## Phone UI
**+ menu → Agent → Grok Build** then **Grok Build sessions**:
- New Grok session
- Refresh
- Tap any row to attach (LIVE badge = open in Terminal)

## APIs
- `GET  /api/engines/grok/sessions`
- `POST /api/engines/grok/sessions/attach`  `{ "session_id": "..." }`
- `POST /api/engines/grok/sessions/new`
- `POST /api/engines/grok/sessions/{id}/open-terminal`

## Scripts
- `./scripts/grok-bridge-status.sh`
- `./scripts/open-grok-session.sh [session_id]`

## Notes
- One writer at a time if Terminal and phone share a session mid-turn.
- Text turns use full remote rules; voice turns stay speakable/short.
- Hermes agent engine is optional and separate from Grok path.

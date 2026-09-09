# Hermes Agent Bridge

**A tool for connecting multiple agentic backends — including the Hermes agent
app — behind one phone-friendly interface, so you can drive them from anywhere.**

Codex, Claude Code, Grok CLI, Antigravity and Hermes all run on your machine;
this gives them a single remote front door.

The server runs on your own machine. Agents execute locally with your own
credentials and your own files. Nothing is proxied through a third-party backend.

> **Status: work in progress.** This is an ongoing personal project, not a
> finished product. It was extracted from a working private build and opened up
> mid-development, so expect rough edges. It runs and is used daily, but the
> codebase is still moving — see [Project status](#project-status) for what is
> solid and what is not.

```
  phone / laptop browser
          │
          │  HTTPS + WebSocket  (password or device session)
          ▼
  server.py  ── engine router ──┬── codex        (CLI, per-thread session)
   :8765                        ├── claude       (CLI, per-thread session)
                                ├── grok         (CLI, attach to live sessions)
                                ├── antigravity  (local agent API)
                                └── hermes       (streaming voice agent)
```

## Why

Coding agents live on the machine that has your repos, your toolchain and your
logins. That machine is usually not the one in your pocket. This bridges the gap:
one local server, one web UI, every agent backend you already have installed.

## Features

- **Multi-engine threads.** Each conversation is pinned to an engine and keeps
  its own session continuity, so you can run Codex in one thread and Claude in
  another.
- **Attach to live sessions.** Grok CLI sessions already open in a terminal can
  be attached and driven from the browser.
- **Voice mode.** Continuous PCM streaming, local `whisper.cpp` transcription,
  streamed model tokens, and local Kokoro TTS — roughly 1.5–2.5 s from end of
  speech to first audio on a warm local stack.
- **Barge-in.** Turn IDs scope every event, so interrupting a reply cannot leak
  stale tokens or audio into the next turn.
- **Installable PWA.** Add to home screen; a service worker caches the shell.
- **Local-first.** Speech-to-text, the fast model and text-to-speech can all run
  on-device. Cloud providers are opt-in overrides, not defaults.

## Project status

Ongoing and unfinished. Opened up mid-development rather than at a release, so
it is honest about where it stands:

**Works and is used daily**

- Multi-engine thread routing across Codex, Claude Code and Grok
- Attaching to live Grok CLI sessions from the browser
- The local voice pipeline, including barge-in and rolling partial transcripts
- Password + device-session auth, and tunnel-vs-LAN origin classification

**Rough or incomplete**

- `server.py` is a single ~330 KB module and `static/index.html` a single
  ~410 KB file. Both grew organically and want splitting up.
- Test coverage is thin — one module is covered; the rest is manually tested.
  There is no CI yet.
- Heavily macOS-shaped: `say` for fallback TTS, `ipconfig getifaddr en0` for the
  cert, and an `/Applications` default for the Codex binary. Linux mostly works
  but is less travelled.
- The Antigravity engine needs its desktop app already running.
- The Hermes engine needs a separate Hermes checkout; without it that one engine
  is unavailable while everything else runs fine.
- Error surfacing is inconsistent — some engine failures land in the server log
  rather than the UI.

Issues and PRs are welcome, but treat the API and layout as unstable.

## Requirements

- macOS or Linux, Python 3.11+
- At least one agent CLI on `PATH` — otherwise there is nothing to drive
- Optional: [Ollama](https://ollama.com) for the local fast voice model
- Optional: a `whisper.cpp` server for local transcription
- Optional: Kokoro for local TTS (installed in its own venv — it pulls in torch)

## Quick start

```bash
git clone https://github.com/<your-username>/hermes-agent-bridge.git
cd hermes-agent-bridge

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

./start.sh
```

On first run `start.sh` generates:

- `.voice_password` — a random password, printed nowhere else. **This is your
  only access control.** Read it with `cat .voice_password`.
- `certs/voice-room.{crt,key}` — a self-signed cert, so browsers will allow the
  microphone. You will have to accept the warning once per device.

Then open `https://<your-machine-ip>:8765` and enter the password.

Both files are git-ignored. Do not commit them.

## Configuration

Everything is environment variables; there is no config file to edit.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HERMES_VOICE_CWD` | `~` | **The only directory agents may touch.** Set this. |
| `HERMES_VOICE_PASSWORD` | generated | Access password |
| `HERMES_VOICE_PUBLIC_NO_AUTH` | `0` | `1` disables auth. See the warning below. |
| `HERMES_PYTHON` | `python3` | Interpreter used to launch the server |
| `VOICE_CODEX_BIN` | ChatGPT.app bundle | Path to the Codex CLI |
| `VOICE_CLAUDE_BIN` | `~/.npm-global/bin/claude` | Path to the Claude CLI |
| `VOICE_GROK_BIN` | `~/.grok/bin/grok` | Path to the Grok CLI |
| `VOICE_ENGINE_TIMEOUT` | `600` | Seconds before an engine turn is killed |
| `HERMES_VOICE_STT_PROVIDER` | `whisper_server` | `whisper_server`, `local`, `openai`, `xai` |
| `HERMES_WHISPER_SERVER_URL` | `127.0.0.1:12712` | Local whisper.cpp endpoint |
| `HERMES_OLLAMA_VOICE_MODEL` | `qwen3:4b-instruct` | Local fast model |
| `HERMES_TTS_ENGINE` | `kokoro` | `kokoro` or `say` (macOS) |
| `HERMES_WHISPER_PROMPT` | tool names | Recogniser bias for your proper nouns |

`HERMES_VOICE_CWD` defaults to your home directory deliberately — a fresh clone
should not reach outside it. Point it at a specific project folder.

### Optional: Hermes agent integration

The `hermes` engine and the xAI OAuth model picker import from a separate Hermes
agent checkout. These imports are lazy and wrapped in `try/except`, so the app
runs fine without it — those features just stay unavailable. To enable, set
`HERMES_AGENT_DIR` and `HERMES_HOME`.

## Security

Read this part.

- **This server executes agent turns with full tool access on your machine.**
  Anyone who reaches it and knows the password can run commands as you.
- **Never expose it to the open internet unauthenticated.** Prefer a VPN or
  Tailscale over a public tunnel. If you do use a tunnel, keep auth on.
- `HERMES_VOICE_PUBLIC_NO_AUTH=1` disables authentication entirely and exists
  only for short local debugging. It defaults to `0`. An earlier version of this
  project defaulted it to `1`, which left the server reachable through a public
  tunnel with no credentials at all. Do not set it and walk away.
- Requests arriving via a tunnel are never treated as local, even though the TCP
  peer is `127.0.0.1`. Host-header checks classify the origin.
- The self-signed cert is not a security boundary — it only unlocks microphone
  access in the browser.

## Optional: hosted frontend

`deploy_vercel.sh` publishes the static UI shell to Vercel so you can load it on
a phone without hosting it yourself. **No agent code runs there** — the shell
calls back to your local server, whose URL you set in Settings.

Set `VERCEL_SCOPE` and `VERCEL_PROJECT`, and put a token in `.vercel_token`.
This step is entirely optional; serving from your own machine works the same.

## Layout

```
server.py            main FastAPI app: routing, engines, voice pipeline, API
engines/
  agent_activity.py    activity/status tracking
  antigravity_engine.py Antigravity local agent API client
  hermes_stream.py     streaming turn engine
  kokoro_service.py    warm Kokoro TTS service
  kokoro_tts.py        Kokoro synthesis entry point
static/index.html    the entire frontend (single file)
static/agent-activity.js activity UI
static/icon-*.png    app icon, generated by scripts/make_icons.py
scripts/             Grok bridge helpers, icon generator
tests/               pytest suite
GROK_BRIDGE.md       how the Grok session bridge works
```

## Team

A two-person project.

- **Omar** — partnerships
- **Hasan** — architecture, design and implementation

## AI assistance

Parts of this project were written with AI coding assistants, which is fitting
given what it does. The architecture, the product decisions and the review are
the author's; the assistants were used for implementation and refactoring. All
code here has been read and is run in daily use.

## Credits

The icon is original to this project and is regenerated from
`scripts/make_icons.py`, so it carries the same MIT license as the code. The
agent CLIs this connects to (Codex, Claude Code, Grok, Antigravity, Hermes) are
separate products under their own licenses and are not bundled here.

## Tests

```bash
python3 -m pytest tests/
```

## License

MIT — see [LICENSE](LICENSE).

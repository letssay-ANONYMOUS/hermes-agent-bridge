#!/usr/bin/env python3
"""Hermes Voice Room sidecar.

Phone browser -> Mac HTTPS server -> Hermes local CLI -> voice output.

This file intentionally stays separate from Hermes core. The current build is
thread-aware and interruption-safe, with a clean gate for a future realtime
voice provider. Until that provider is configured, voice turns use the existing
record/transcribe/respond/speak fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextvars
from contextlib import asynccontextmanager
import json
import logging
import os
import re
import secrets
import select
import shlex
import signal
import sqlite3
import functools
import io
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Reuse Uvicorn's configured handler so INFO lifecycle events are visible in
# the LaunchAgent log. A standalone logger inherited a warning-only handler,
# which hid successful WebSocket and PCM events during physical-device tests.
logger = logging.getLogger("uvicorn.error")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "voice_room.sqlite"
HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
HERMES_AGENT = Path(os.environ.get("HERMES_AGENT_DIR", str(HERMES_HOME / "hermes-agent")))
HERMES_PYTHON = Path(os.environ.get("HERMES_PYTHON", str(HERMES_AGENT / "venv/bin/python")))
HERMES_CONFIG_PATH = Path(os.environ.get("HERMES_CONFIG_PATH", str(HERMES_HOME / "config.yaml")))
# The single folder the agent is allowed to work in. Point HERMES_VOICE_CWD at
# your own workspace; it defaults to the home directory so a fresh clone never
# reaches outside it.
WORKSPACE_ROOT = Path(os.environ.get("HERMES_VOICE_CWD", str(Path.home())))
DEFAULT_CWD = WORKSPACE_ROOT
VOICE_AGENT_SCOPE = (
    "Voice-room scope guard: run terminal, file, git, project, browser-download, "
    f"and Finder-style work from {DEFAULT_CWD}. Treat this as the allowed "
    "workspace. Do not inspect, modify, or summarize files outside that folder "
    "unless the user explicitly names a different absolute path in the current "
    "request. If a task appears to require another folder, ask first."
)
DEFAULT_VOICE = os.environ.get("HERMES_MAC_VOICE", "Samantha")
DEFAULT_TTS_ENGINE = os.environ.get("HERMES_TTS_ENGINE", "kokoro").strip().lower()
KOKORO_PYTHON = Path(
    os.environ.get("HERMES_KOKORO_PYTHON", str(HERMES_HOME / "voice_engines/kokoro/bin/python"))
)
KOKORO_SCRIPT = Path(os.environ.get("HERMES_KOKORO_SCRIPT", str(ROOT / "engines/kokoro_tts.py")))
KOKORO_VOICE = os.environ.get("HERMES_KOKORO_VOICE", "af_heart")
KOKORO_LANG = os.environ.get("HERMES_KOKORO_LANG", "a")
KOKORO_SERVICE_URL = os.environ.get("HERMES_KOKORO_SERVICE_URL", "http://127.0.0.1:8789")

# ── live voice selection (Settings → Voice picker) ──────────────────────────
# The active TTS voice is user-selectable at runtime and persisted here. Both
# the streaming WS synth and the HTTP fallback consult _current_voice().
VOICE_SETTINGS_PATH = ROOT / "data" / "voice_settings.json"
VOICE_RUNTIME_PATH = ROOT / "data" / "voice_runtime.json"
_VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")   # blocks shell/path injection
_ALLOWED_LANGS = {"a", "b", "ar", "en"}
_voice_lock = threading.Lock()
_runtime_lock = threading.Lock()

# Runtime overrides (UI-persisted). Env vars remain the process defaults.
# fast_provider: "xai" | "ollama" — which backend answers in Fast mode
# stt_engine: "whisper_flow" | "browser" | "auto"
#   whisper_flow = local whisper.cpp (Whisper Flow server on :12712)
#   browser      = client Web Speech API (natural; no local whisper)
#   auto         = whisper_flow when reachable, else hermes configured STT
# allow_ollama_fallback: if false, xAI auth failure is reported instead of silent Qwen
_RUNTIME_DEFAULTS: dict[str, Any] = {
    "fast_provider": None,  # None = use HERMES_VOICE_FAST_PROVIDER env
    "stt_engine": "whisper_flow",
    "allow_ollama_fallback": False,
    "stt_language": "en",
}


def _load_voice_runtime() -> dict[str, Any]:
    data = dict(_RUNTIME_DEFAULTS)
    try:
        raw = json.loads(VOICE_RUNTIME_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            data.update({k: raw[k] for k in _RUNTIME_DEFAULTS if k in raw})
    except Exception:
        pass
    # Normalize
    fp = data.get("fast_provider")
    if fp is not None:
        fp = str(fp).strip().lower()
        data["fast_provider"] = fp if fp in {"xai", "ollama"} else None
    se = str(data.get("stt_engine") or "whisper_flow").strip().lower()
    data["stt_engine"] = se if se in {"whisper_flow", "browser", "auto"} else "whisper_flow"
    data["allow_ollama_fallback"] = bool(data.get("allow_ollama_fallback"))
    lang = str(data.get("stt_language") or "en").strip().lower() or "en"
    data["stt_language"] = lang[:8]
    return data


def _save_voice_runtime(patch: dict[str, Any]) -> dict[str, Any]:
    with _runtime_lock:
        current = _load_voice_runtime()
        if "fast_provider" in patch:
            fp = patch["fast_provider"]
            if fp is None or str(fp).strip() == "" or str(fp).strip().lower() == "default":
                current["fast_provider"] = None
            else:
                fp = str(fp).strip().lower()
                if fp not in {"xai", "ollama"}:
                    raise HTTPException(status_code=400, detail="fast_provider must be xai or ollama")
                current["fast_provider"] = fp
        if "stt_engine" in patch and patch["stt_engine"] is not None:
            se = str(patch["stt_engine"]).strip().lower()
            if se not in {"whisper_flow", "browser", "auto"}:
                raise HTTPException(status_code=400, detail="stt_engine must be whisper_flow, browser, or auto")
            current["stt_engine"] = se
        if "allow_ollama_fallback" in patch and patch["allow_ollama_fallback"] is not None:
            current["allow_ollama_fallback"] = bool(patch["allow_ollama_fallback"])
        if "stt_language" in patch and patch["stt_language"] is not None:
            current["stt_language"] = str(patch["stt_language"]).strip().lower()[:8] or "en"
        VOICE_RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
        VOICE_RUNTIME_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        return current


def _effective_fast_provider() -> str:
    """Provider used for Fast-mode turns. UI override beats env default."""
    runtime = _load_voice_runtime()
    override = runtime.get("fast_provider")
    if override in {"xai", "ollama"}:
        return override
    return VOICE_FAST_PROVIDER if VOICE_FAST_PROVIDER in {"xai", "ollama"} else "xai"


def _effective_stt_engine() -> str:
    return str(_load_voice_runtime().get("stt_engine") or "whisper_flow")


def _validate_voice(engine: str, voice: str, lang: str) -> dict[str, str]:
    if engine not in ("kokoro", "say"):
        raise HTTPException(status_code=400, detail="engine must be kokoro or say")
    if not voice or not _VOICE_NAME_RE.match(voice):
        raise HTTPException(status_code=400, detail="invalid voice name")
    return {"engine": engine, "voice": voice, "lang": lang if lang in _ALLOWED_LANGS else "a"}


def _load_voice_settings() -> dict[str, str]:
    try:
        data = json.loads(VOICE_SETTINGS_PATH.read_text())
        if isinstance(data, dict) and _VOICE_NAME_RE.match(str(data.get("voice", ""))):
            return _validate_voice(data.get("engine", "kokoro"), data["voice"], data.get("lang", "a"))
    except Exception:
        pass
    return {"engine": "kokoro", "voice": KOKORO_VOICE, "lang": KOKORO_LANG}


_active_voice = _load_voice_settings()

# Voice picker catalog. Kokoro US voices (verified working) + macOS `say` voices
# for accents/Arabic that Kokoro lacks. `say` entries are filtered to those
# actually installed on this Mac at startup.
_VOICE_CATALOG_RAW = [
    {"id": "af_heart", "name": "Heart", "desc": "Warm and natural", "category": "Signature", "engine": "kokoro", "voice": "af_heart", "lang": "a"},
    {"id": "af_bella", "name": "Bella", "desc": "Bright and expressive", "category": "Signature", "engine": "kokoro", "voice": "af_bella", "lang": "a"},
    {"id": "am_michael", "name": "Michael", "desc": "Friendly and easy", "category": "Signature", "engine": "kokoro", "voice": "am_michael", "lang": "a"},
    {"id": "am_adam", "name": "Adam", "desc": "Deep and steady", "category": "Signature", "engine": "kokoro", "voice": "am_adam", "lang": "a"},
    {"id": "am_onyx", "name": "Onyx", "desc": "Rich and cinematic", "category": "Signature", "engine": "kokoro", "voice": "am_onyx", "lang": "a"},
    {"id": "Samantha", "name": "Samantha", "desc": "American, clear", "category": "Accents", "engine": "say", "voice": "Samantha", "lang": "en"},
    {"id": "Daniel", "name": "Daniel", "desc": "British, refined", "category": "Accents", "engine": "say", "voice": "Daniel", "lang": "en"},
    {"id": "Karen", "name": "Karen", "desc": "Australian, upbeat", "category": "Accents", "engine": "say", "voice": "Karen", "lang": "en"},
    {"id": "Moira", "name": "Moira", "desc": "Irish, warm", "category": "Accents", "engine": "say", "voice": "Moira", "lang": "en"},
    {"id": "Rishi", "name": "Rishi", "desc": "Indian English", "category": "Accents", "engine": "say", "voice": "Rishi", "lang": "en"},
    {"id": "Majed", "name": "Majed", "desc": "Arabic — العربية", "category": "Arabic", "engine": "say", "voice": "Majed", "lang": "ar"},
]


@functools.lru_cache(maxsize=1)
def _installed_say_voices() -> frozenset[str]:
    try:
        out = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
        return frozenset(line.split()[0] for line in out.splitlines() if line.strip())
    except Exception:
        return frozenset()


def _voice_catalog() -> list[dict[str, str]]:
    installed = _installed_say_voices()
    return [v for v in _VOICE_CATALOG_RAW
            if v["engine"] != "say" or v["voice"] in installed]


def _current_voice() -> dict[str, str]:
    with _voice_lock:
        return dict(_active_voice)


def _set_voice(engine: str, voice: str, lang: str) -> dict[str, str]:
    v = _validate_voice(engine, voice, lang)
    with _voice_lock:
        global _active_voice
        _active_voice = v
        try:
            VOICE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            VOICE_SETTINGS_PATH.write_text(json.dumps(v))
        except Exception:
            pass
    return v
VOICE_PASSWORD = os.environ.get("HERMES_VOICE_PASSWORD", "")
VOICE_PUBLIC_NO_AUTH = os.environ.get("HERMES_VOICE_PUBLIC_NO_AUTH", "").strip().lower() in {
    "1", "true", "yes", "on",
}
DEVICE_SESSIONS_PATH = DATA_DIR / "device_sessions.json"
DEVICE_SESSION_TTL_SECONDS = 90 * 24 * 60 * 60
DEVICE_SESSION_REFRESH_SECONDS = 12 * 60 * 60
THREAD_SOURCE = "voice-room"
MAX_AUDIO_BYTES = 18 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
ATTACHMENTS_DIR = DATA_DIR / "attachments"
TEXT_ATTACHMENT_PREVIEW_BYTES = 192 * 1024
HERMES_SESSION_RE = re.compile(r"session_id:\s*([A-Za-z0-9_.:-]+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
XAI_PROVIDERS = ("xai-oauth", "xai")
REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh")
THREAD_REASONING_EFFORTS = (*REASONING_EFFORTS, "max", "ultra")
# Hermes approval gate: full = run everything (yolo), smart = auto-approve
# low-risk & ask on dangerous, ask = prompt on every dangerous command.
APPROVAL_MODES = ("ask", "smart", "full")
AVAILABLE_TOOLSETS = (
    {"id": "hermes-cli", "label": "Hermes CLI preset", "hint": "Default full Hermes desktop/CLI toolbelt."},
    {"id": "web", "label": "Web", "hint": "Search and extract web pages."},
    {"id": "search", "label": "Search", "hint": "Search only, no page extraction."},
    {"id": "browser", "label": "Browser", "hint": "Navigate, click, type, inspect browser pages."},
    {"id": "terminal", "label": "Terminal", "hint": "Run shell commands and manage processes."},
    {"id": "file", "label": "Files", "hint": "Read, write, patch, and search files."},
    {"id": "code_execution", "label": "Code execution", "hint": "Run code snippets in sandboxed executors when available."},
    {"id": "vision", "label": "Vision", "hint": "Analyze images and screenshots."},
    {"id": "video", "label": "Video", "hint": "Analyze video inputs when configured."},
    {"id": "image_gen", "label": "Image generation", "hint": "Generate images when provider keys exist."},
    {"id": "video_gen", "label": "Video generation", "hint": "Generate videos when provider keys exist."},
    {"id": "x_search", "label": "X search", "hint": "Search X/Twitter when configured."},
    {"id": "skills", "label": "Skills", "hint": "Load local Hermes skills."},
    {"id": "skills_hub", "label": "Skills hub", "hint": "Search and install online skills."},
    {"id": "todo", "label": "Todo", "hint": "Track task plans in-session."},
    {"id": "memory", "label": "Memory", "hint": "Save and retrieve persistent Hermes memory."},
    {"id": "session_search", "label": "Session search", "hint": "Search previous Hermes conversations."},
    {"id": "clarify", "label": "Clarify", "hint": "Ask focused clarification questions."},
    {"id": "delegation", "label": "Delegation", "hint": "Delegate work to sub-agents when available."},
    {"id": "kanban", "label": "Kanban", "hint": "Coordinate queued and background work."},
    {"id": "cronjob", "label": "Cron jobs", "hint": "Create and manage scheduled tasks."},
    {"id": "computer_use", "label": "Computer use", "hint": "Operate local desktop UI when available."},
    {"id": "tts", "label": "Text to speech", "hint": "Generate spoken audio."},
    {"id": "all", "label": "All", "hint": "Composite toolset exposing everything Hermes can load."},
    {"id": "debugging", "label": "Debugging", "hint": "Terminal, web, and file tools."},
    {"id": "safe", "label": "Safe", "hint": "Read-oriented web and vision tools, no terminal."},
)
AVAILABLE_TOOLSET_IDS = {item["id"] for item in AVAILABLE_TOOLSETS}
_REQUEST_AUTH_OK: contextvars.ContextVar[bool] = contextvars.ContextVar("hermes_voice_request_auth_ok", default=False)
_WS_TURN_ID: contextvars.ContextVar[str] = contextvars.ContextVar("hermes_voice_ws_turn_id", default="")
_TURN_EVENT_TYPES = frozenset({"state", "continue", "transcript", "activity", "token", "audio", "done", "error"})


def _read_approvals_mode(data: Any) -> str:
    """Map the raw approvals.mode config value to our 3 UI modes."""
    approvals = data.get("approvals") if isinstance(data.get("approvals"), dict) else {}
    raw = approvals.get("mode", "manual")
    # Unquoted `off` parses to the bool False; Hermes treats that as yolo.
    if raw is False or str(raw).strip().lower() == "off":
        return "full"
    if str(raw).strip().lower() == "smart":
        return "smart"
    return "ask"
# fast: no toolsets, no AGENTS.md/memory injection -> much smaller prompt, ~5s
# faster per turn. agent: the full Hermes toolbelt for real work.
THREAD_MODES = ("fast", "agent")

# ---------------------------------------------------------------------------
# Agent engines: which local AI agent answers a thread's turns.
# "hermes" keeps the original fast/agent behavior; the others run a headless
# CLI turn with full tool access and per-thread session continuity.
# ---------------------------------------------------------------------------
GROK_BIN = os.environ.get("VOICE_GROK_BIN", str(Path.home() / ".grok/bin/grok"))
CODEX_BIN = os.environ.get("VOICE_CODEX_BIN", "/Applications/ChatGPT.app/Contents/Resources/codex")
CLAUDE_BIN = os.environ.get("VOICE_CLAUDE_BIN", str(Path.home() / ".npm-global/bin/claude"))
ENGINE_TURN_TIMEOUT = float(os.environ.get("VOICE_ENGINE_TIMEOUT", "600"))
ANTIGRAVITY_TURN_TIMEOUT = float(os.environ.get("VOICE_ANTIGRAVITY_TIMEOUT", "300"))
ENGINE_SETTINGS_PATH = DATA_DIR / "engine_settings.json"

# Antigravity is driven through a separate poll-based adapter (its agentapi CLI
# is fire-and-forget into the IDE). Imported best-effort so a missing adapter
# never blocks server startup.
if str(ROOT / "engines") not in sys.path:
    sys.path.insert(0, str(ROOT / "engines"))
try:
    import antigravity_engine as _antigravity
except Exception:  # pragma: no cover - adapter optional
    _antigravity = None

ENGINE_VOICE_RULES = (
    "You are talking in a hands-free VOICE conversation. Reply in short, natural, speakable "
    "sentences — two to four unless asked for more. Never use markdown, headings, code blocks, "
    "or bullet lists; describe code and files in plain words. You have full tool access on this "
    "Mac: when asked to do real work, actually do it with your tools, then say what you did in "
    "a sentence or two."
)
# Full remote control from the phone web app (text turns). Full tools, normal coding replies.
ENGINE_REMOTE_RULES = (
    "You are Grok Build (or the selected coding agent) controlled from a remote web UI on the user's phone. "
    "You have FULL tool access on this Mac — run commands, edit files, use git, browse, and complete real work. "
    "Do the work; do not only describe it. Reply clearly for a small phone screen: structured when useful "
    "(short headings, bullets, code fences OK). Be decisive. This is NOT read-only."
)
# Codex has no system-prompt flag in exec mode, so it gets a per-prompt note instead.
ENGINE_VOICE_NOTE = "(Voice chat: reply in short speakable sentences, no markdown or code blocks.)"
ENGINE_REMOTE_NOTE = (
    "(Remote phone UI: full tools on this Mac. Do real work. Clear structured replies OK.)"
)

AGENT_ENGINES: dict[str, dict[str, Any]] = {
    "hermes": {"label": "Hermes", "bin": None, "detail": "Local Hermes agent (fast + agent modes)"},
    "grok": {"label": "Grok Build", "bin": GROK_BIN, "detail": "xAI Grok Build agent with tools"},
    "codex": {"label": "Codex", "bin": CODEX_BIN, "detail": "OpenAI Codex agent with tools"},
    "claude": {"label": "Claude Code", "bin": CLAUDE_BIN, "detail": "Anthropic Claude Code agent with tools"},
    "antigravity": {
        "label": "Antigravity", "bin": None, "detail": "Google Antigravity — parallel background agents",
        "poll": True,  # dispatch-and-poll adapter, not a token stream
    },
}

_ENGINE_ALIASES = {
    "hermes": "hermes",
    "grok": "grok", "grok build": "grok", "grock": "grok", "grock build": "grok",
    "codex": "codex", "codecs": "codex", "codex cli": "codex",
    "claude": "claude", "claude code": "claude", "cloud code": "claude", "clod": "claude",
    "antigravity": "antigravity", "anti gravity": "antigravity", "anti-gravity": "antigravity",
}


_claude_auth_cache: dict[str, Any] = {"ok": None, "at": 0.0}


def _claude_auth_ok() -> bool:
    """Cheap check that the Claude CLI is actually logged in.

    Reads ONLY the OAuth token expiry from ~/.claude/.credentials.json (never
    the token itself) so the frontend can grey Claude out when it would just
    401. Cached 30s to keep /api/engines fast. Falls back to True when we
    can't tell — the runtime login-hint still covers that case.
    """
    now = time.time()
    if _claude_auth_cache["ok"] is not None and now - _claude_auth_cache["at"] < 30:
        return bool(_claude_auth_cache["ok"])
    ok = True
    try:
        cred = Path(os.path.expanduser("~/.claude/.credentials.json"))
        if cred.exists():
            oauth = (json.loads(cred.read_text()).get("claudeAiOauth") or {})
            exp = oauth.get("expiresAt")
            if isinstance(exp, (int, float)):
                # expiresAt is epoch millis; an expired access token with no
                # working refresh is exactly the logged-out 401 case.
                ok = (float(exp) / 1000.0) > now
    except Exception:
        ok = True
    _claude_auth_cache["ok"] = ok
    _claude_auth_cache["at"] = now
    return ok


def _antigravity_note() -> str:
    """Why antigravity is unavailable right now (for spoken/UI hints)."""
    if _antigravity is None:
        return "the Antigravity adapter failed to load on this Mac"
    if not _antigravity.AGENTAPI_BIN.exists():
        return "Antigravity 2.0 isn't installed (no agentapi CLI found)"
    return "the Antigravity app isn't running — open it on the Mac first"


def _antigravity_app_running() -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-x", "Antigravity"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


MODEL_CATALOG_TTL_SECONDS = 60.0
ENGINE_PROBE_TTL_SECONDS = 12.0
_model_catalog_cache: dict[str, dict[str, Any]] = {}
_model_catalog_lock = threading.RLock()
_engine_probe_cache: dict[str, dict[str, Any]] = {}
_engine_probe_lock = threading.RLock()


def _reasoning_option(
    effort_id: str,
    *,
    label: str | None = None,
    detail: str | None = None,
    is_default: bool = False,
) -> dict[str, Any]:
    clean = str(effort_id or "").strip().lower()
    return {
        "id": clean,
        "label": label or clean.replace("xhigh", "XHigh").replace("_", " ").title(),
        "detail": detail or "",
        "is_default": bool(is_default),
    }


def _model_option(
    model_id: str,
    *,
    label: str | None = None,
    detail: str | None = None,
    is_default: bool = False,
    default_reasoning_effort: str | None = None,
    reasoning_efforts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    clean = str(model_id or "").strip()
    return {
        "id": clean,
        "label": label or clean,
        "detail": detail or "",
        "is_default": bool(is_default),
        "default_reasoning_effort": str(default_reasoning_effort or "").strip().lower() or None,
        "reasoning_efforts": reasoning_efforts or [],
    }


def _dedupe_model_options(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per exact model id while preserving provider order."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for model in models:
        model_id = str(model.get("id") or "").strip()
        key = model_id.casefold()
        if not model_id or key in seen:
            continue
        seen.add(key)
        model["id"] = model_id
        result.append(model)
    return result


def _grok_model_catalog() -> tuple[list[dict[str, Any]], str | None]:
    """Ask the authenticated Grok CLI which models are actually selectable."""
    proc = subprocess.run(
        [GROK_BIN, "models"],
        cwd=str(DEFAULT_CWD),
        env=_engine_env("grok"),
        text=True,
        capture_output=True,
        timeout=15,
    )
    output = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    if proc.returncode != 0 or "logged in" not in output.lower():
        detail = ANSI_RE.sub("", output).strip()[-300:] or "Grok CLI is not authenticated"
        raise RuntimeError(detail)
    default_match = re.search(r"^Default model:\s*(\S+)", output, re.MULTILINE | re.IGNORECASE)
    default_model = default_match.group(1).strip() if default_match else None
    ids = [
        match.group(1).strip()
        for match in re.finditer(r"^\s*[*-]\s+([^\s(]+)", output, re.MULTILINE)
    ]
    if default_model and default_model not in ids:
        ids.insert(0, default_model)
    if not ids:
        raise RuntimeError("Grok CLI returned no selectable models")

    metadata: dict[str, Any] = {}
    try:
        raw = json.loads(Path(os.path.expanduser("~/.grok/models_cache.json")).read_text(encoding="utf-8"))
        metadata = raw.get("models") if isinstance(raw, dict) and isinstance(raw.get("models"), dict) else {}
    except Exception:
        metadata = {}

    result: list[dict[str, Any]] = []
    for model_id in ids:
        raw = metadata.get(model_id) if isinstance(metadata, dict) else None
        info = raw.get("info") if isinstance(raw, dict) and isinstance(raw.get("info"), dict) else {}
        default_effort = str(info.get("reasoning_effort") or "").strip().lower() or None
        efforts: list[dict[str, Any]] = []
        for raw_effort in info.get("reasoning_efforts") or []:
            if not isinstance(raw_effort, dict):
                continue
            effort_id = str(raw_effort.get("id") or raw_effort.get("value") or "").strip().lower()
            if effort_id not in THREAD_REASONING_EFFORTS or not effort_id:
                continue
            efforts.append(_reasoning_option(
                effort_id,
                label=str(raw_effort.get("label") or effort_id.title()),
                detail=str(raw_effort.get("description") or ""),
                is_default=bool(raw_effort.get("default")) or effort_id == default_effort,
            ))
        result.append(_model_option(
            model_id,
            label=str(info.get("name") or model_id),
            detail=str(info.get("description") or "Grok Build model"),
            is_default=model_id == default_model,
            default_reasoning_effort=default_effort,
            reasoning_efforts=efforts,
        ))
    return _dedupe_model_options(result), default_model


def _codex_model_catalog() -> tuple[list[dict[str, Any]], str | None]:
    """Read the authenticated Codex app-server model list (never local presets)."""
    proc = subprocess.Popen(
        [CODEX_BIN, "app-server", "--stdio"],
        cwd=str(DEFAULT_CWD),
        env=_engine_env("codex"),
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1,
    )

    def send(payload: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise RuntimeError("Codex app-server stdin is unavailable")
        proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    def receive(response_id: int, deadline: float) -> dict[str, Any]:
        if proc.stdout is None:
            raise RuntimeError("Codex app-server stdout is unavailable")
        while time.time() < deadline:
            ready, _, _ = select.select([proc.stdout], [], [], max(0.0, deadline - time.time()))
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("id") == response_id:
                if payload.get("error"):
                    raise RuntimeError(f"Codex app-server: {payload['error']}")
                return payload
        raise RuntimeError("Codex app-server model lookup timed out")

    try:
        deadline = time.time() + 12.0
        send({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "hermes-voice-room", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            },
        })
        receive(1, deadline)
        send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        send({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "model/list",
            "params": {"includeHidden": False, "limit": 100},
        })
        response = receive(2, deadline)
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    raw_models = (response.get("result") or {}).get("data") or []
    result: list[dict[str, Any]] = []
    default_model: str | None = None
    for raw in raw_models:
        if not isinstance(raw, dict) or raw.get("hidden") is True:
            continue
        model_id = str(raw.get("id") or raw.get("model") or "").strip()
        if not model_id:
            continue
        is_default = bool(raw.get("isDefault"))
        if is_default:
            default_model = model_id
        default_effort = str(raw.get("defaultReasoningEffort") or "").strip().lower() or None
        efforts: list[dict[str, Any]] = []
        for raw_effort in raw.get("supportedReasoningEfforts") or []:
            if not isinstance(raw_effort, dict):
                continue
            effort_id = str(raw_effort.get("reasoningEffort") or "").strip().lower()
            if effort_id not in THREAD_REASONING_EFFORTS or not effort_id:
                continue
            efforts.append(_reasoning_option(
                effort_id,
                detail=str(raw_effort.get("description") or ""),
                is_default=effort_id == default_effort,
            ))
        result.append(_model_option(
            model_id,
            label=str(raw.get("displayName") or model_id),
            detail=str(raw.get("description") or "Codex model"),
            is_default=is_default,
            default_reasoning_effort=default_effort,
            reasoning_efforts=efforts,
        ))
    result = _dedupe_model_options(result)
    if not result:
        raise RuntimeError("Codex returned no selectable models")
    return result, default_model or result[0]["id"]


def _hermes_model_catalog() -> tuple[list[dict[str, Any]], str | None, str]:
    _yaml, data = _load_hermes_config_document()
    model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
    agent_cfg = data.get("agent") if isinstance(data.get("agent"), dict) else {}
    provider = str(model_cfg.get("provider") or "xai-oauth").strip()
    current = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    current_effort = str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    catalogs = _xai_model_options()
    ids = list(catalogs.get(provider) or [])
    if current and current not in ids:
        ids.insert(0, current)
    support = _xai_reasoning_support({provider: ids}).get(provider, {})
    models: list[dict[str, Any]] = []
    for model_id in ids:
        supported = bool(support.get(model_id))
        efforts = [
            _reasoning_option(
                effort,
                is_default=effort == current_effort and model_id == current,
            )
            for effort in REASONING_EFFORTS
            if effort and supported
        ]
        models.append(_model_option(
            model_id,
            detail=f"Hermes {provider} model",
            is_default=model_id == current,
            default_reasoning_effort=current_effort if supported and model_id == current else None,
            reasoning_efforts=efforts,
        ))
    models = _dedupe_model_options(models)
    return models, current or (models[0]["id"] if models else None), provider


def _engine_model_payload_uncached(engine_id: str) -> dict[str, Any]:
    if engine_id == "grok":
        models, default_model = _grok_model_catalog()
        provider, note = "grok.com", "Live catalog from the authenticated Grok CLI."
    elif engine_id == "codex":
        models, default_model = _codex_model_catalog()
        provider, note = "chatgpt", "Live catalog from the authenticated Codex app-server."
    elif engine_id == "hermes":
        models, default_model, provider = _hermes_model_catalog()
        note = "Models exposed by Hermes for its current provider."
    elif engine_id == "antigravity":
        models, default_model, provider = [], None, "antigravity"
        note = "Antigravity does not expose a truthful selectable model catalog while the IDE is closed."
    elif engine_id == "claude":
        models, default_model, provider = [], None, "anthropic"
        note = "Claude Code does not expose a local selectable model catalog to this app."
    else:
        raise RuntimeError(f"Unknown engine: {engine_id}")
    return {
        "engine_id": engine_id,
        "provider": provider,
        "models": models,
        "default_model": default_model,
        "note": note,
        "fetched_at": time.time(),
        "ttl_seconds": MODEL_CATALOG_TTL_SECONDS,
    }


def _engine_model_payload(engine_id: str, *, refresh: bool = False) -> dict[str, Any]:
    if engine_id not in AGENT_ENGINES:
        raise RuntimeError(f"Unknown engine: {engine_id}")
    now = time.time()
    with _model_catalog_lock:
        cached = _model_catalog_cache.get(engine_id)
        if not refresh and cached and now - float(cached.get("fetched_at") or 0) < MODEL_CATALOG_TTL_SECONDS:
            return cached
    payload = _engine_model_payload_uncached(engine_id)
    with _model_catalog_lock:
        _model_catalog_cache[engine_id] = payload
    return payload


def _engine_models(engine_id: str) -> list[dict[str, Any]]:
    try:
        return list(_engine_model_payload(engine_id).get("models") or [])
    except Exception as exc:
        logger.warning("Could not load %s model catalog: %s", engine_id, exc)
        return []


def _engine_default_model(engine_id: str) -> str | None:
    try:
        return _engine_model_payload(engine_id).get("default_model")
    except Exception:
        return None


def _probe_hermes() -> str:
    _yaml, data = _load_hermes_config_document()
    model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
    provider = str(model_cfg.get("provider") or "xai-oauth").strip()
    if provider not in XAI_PROVIDERS:
        raise RuntimeError(f"Hermes provider {provider} has no safe preflight probe")
    creds = _resolve_xai_stream_credentials(provider)
    api_key = str(creds.get("api_key") or "")
    if not api_key:
        raise RuntimeError(f"Hermes {provider} has no active credential")
    request = urllib.request.Request(
        f"{str(creds.get('base_url') or 'https://api.x.ai/v1').rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=7) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"Hermes provider returned HTTP {response.status}")
        json.loads(response.read().decode("utf-8"))
    return f"Authenticated to Hermes {provider}."


def _probe_engine_uncached(engine_id: str) -> dict[str, Any]:
    info = AGENT_ENGINES.get(engine_id)
    if not info:
        raise RuntimeError(f"Unknown engine: {engine_id}")
    if info.get("unavailable_reason"):
        raise RuntimeError(str(info["unavailable_reason"]))
    if engine_id == "hermes":
        note = _probe_hermes()
    elif engine_id == "grok":
        models = _engine_model_payload("grok", refresh=True).get("models") or []
        note = f"Authenticated Grok CLI; {len(models)} selectable models."
    elif engine_id == "codex":
        login = subprocess.run(
            [CODEX_BIN, "login", "status"],
            cwd=str(DEFAULT_CWD),
            env=_engine_env("codex"),
            text=True,
            capture_output=True,
            timeout=10,
        )
        login_text = f"{login.stdout or ''}\n{login.stderr or ''}"
        if login.returncode != 0 or "logged in" not in login_text.lower():
            raise RuntimeError(ANSI_RE.sub("", login_text).strip()[-300:] or "Codex is not logged in")
        models = _engine_model_payload("codex", refresh=True).get("models") or []
        note = f"Authenticated Codex app-server; {len(models)} selectable models."
    elif engine_id == "claude":
        if not os.access(CLAUDE_BIN, os.X_OK):
            raise RuntimeError("Claude CLI is not installed")
        if not _claude_auth_ok():
            raise RuntimeError("Claude needs a login — run `claude` then /login on the Mac")
        note = "Claude CLI credential is present and unexpired."
    elif engine_id == "antigravity":
        if not _antigravity_app_running() or _antigravity is None or not _antigravity.is_available():
            raise RuntimeError(_antigravity_note())
        note = "Antigravity IDE language server and agent adapter are running."
    else:
        raise RuntimeError(f"Unknown engine: {engine_id}")
    return {"available": True, "note": note, "checked_at": time.time()}


def _engine_probe(engine_id: str, *, refresh: bool = False) -> dict[str, Any]:
    now = time.time()
    with _engine_probe_lock:
        cached = _engine_probe_cache.get(engine_id)
        if not refresh and cached and now - float(cached.get("checked_at") or 0) < ENGINE_PROBE_TTL_SECONDS:
            return cached
    try:
        result = _probe_engine_uncached(engine_id)
    except Exception as exc:
        detail = ANSI_RE.sub("", str(exc)).strip()[-400:] or "Connection probe failed"
        result = {"available": False, "note": detail, "checked_at": time.time()}
    with _engine_probe_lock:
        _engine_probe_cache[engine_id] = result
    return result


def _engine_available(engine_id: str) -> bool:
    return bool(_engine_probe(engine_id).get("available"))


def _engine_startable(engine_id: str) -> bool:
    return bool(
        engine_id == "antigravity"
        and _antigravity is not None
        and _antigravity.can_start()
    )


def _ensure_engine_available(engine_id: str) -> bool:
    if _engine_available(engine_id):
        return True
    if _engine_startable(engine_id) and _antigravity.start():
        return bool(_engine_probe(engine_id, refresh=True).get("available"))
    return False


def _engine_unavailable_note(engine_id: str) -> str:
    """Human/spoken reason an engine can't be selected."""
    return str(_engine_probe(engine_id).get("note") or "Connection probe failed")


def _default_engine() -> str:
    try:
        data = json.loads(ENGINE_SETTINGS_PATH.read_text())
        engine = str(data.get("default_engine") or "").strip().lower()
    except Exception:
        engine = ""
    if engine in AGENT_ENGINES:
        return engine
    return "hermes"


def _set_default_engine(engine_id: str) -> None:
    ENGINE_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENGINE_SETTINGS_PATH.write_text(json.dumps({"default_engine": engine_id}, indent=2))


def _engines_payload(*, refresh: bool = False) -> dict[str, Any]:
    engines: list[dict[str, Any]] = []
    for engine_id, info in AGENT_ENGINES.items():
        probe = _engine_probe(engine_id, refresh=refresh)
        engines.append({
            "id": engine_id,
            "label": info["label"],
            "detail": info["detail"],
            "available": bool(probe.get("available")),
            "startable": _engine_startable(engine_id),
            "poll": bool(info.get("poll")),
            "note": str(probe.get("note") or ""),
            "checked_at": probe.get("checked_at"),
            # Compatibility for the existing web/native clients. New clients
            # use the per-engine endpoint so each catalog can load independently.
            "models": _engine_models(engine_id),
        })
    return {"default": _default_engine(), "engines": engines}


def _normalize_engine_selection(
    engine_id: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    choose_default: bool = False,
) -> tuple[str | None, str | None]:
    """Validate a per-thread model/effort against the live engine catalog."""
    try:
        payload = _engine_model_payload(engine_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load {engine_id} models: {exc}") from exc
    models = list(payload.get("models") or [])
    selected_model = str(model or "").strip() or None
    if selected_model is None and choose_default:
        selected_model = str(payload.get("default_model") or "").strip() or None
    selected = next((item for item in models if item.get("id") == selected_model), None)
    if selected_model and selected is None:
        raise HTTPException(status_code=400, detail=f"Model is not in {engine_id}'s live catalog: {selected_model}")

    if reasoning_effort is None:
        selected_effort = (
            str(selected.get("default_reasoning_effort") or "").strip().lower() or None
            if selected is not None and choose_default
            else None
        )
    else:
        selected_effort = str(reasoning_effort or "").strip().lower() or None
    if selected_effort and selected_effort not in THREAD_REASONING_EFFORTS:
        raise HTTPException(status_code=400, detail=f"Invalid reasoning effort: {selected_effort}")
    if selected_effort:
        supported = {
            str(item.get("id") or "").strip().lower()
            for item in (selected or {}).get("reasoning_efforts") or []
        }
        if selected is None or selected_effort not in supported:
            raise HTTPException(
                status_code=400,
                detail=f"{selected_model or engine_id} does not expose reasoning effort {selected_effort}",
            )
    return selected_model, selected_effort
VOICE_STT_PROVIDER = os.environ.get("HERMES_VOICE_STT_PROVIDER", "whisper_server").strip().lower()
VOICE_STT_FALLBACKS = tuple(
    p.strip().lower()
    for p in os.environ.get("HERMES_VOICE_STT_FALLBACKS", "local").split(",")
    if p.strip()
)
VOICE_STT_LOCAL_MODEL = os.environ.get("HERMES_VOICE_STT_LOCAL_MODEL", "base").strip() or "base"
VOICE_STT_OPENAI_MODEL = os.environ.get("HERMES_VOICE_STT_OPENAI_MODEL", "whisper-1").strip() or "whisper-1"
VOICE_STT_XAI_MODEL = os.environ.get("HERMES_VOICE_STT_XAI_MODEL", "grok-stt").strip() or "grok-stt"
WHISPER_SERVER_URL = os.environ.get(
    "HERMES_WHISPER_SERVER_URL", "http://127.0.0.1:12712/inference"
).rstrip("/")
# Rolling partial STT while the user is still speaking (single-flight Whisper).
ROLLING_PARTIAL_INTERVAL_S = float(os.environ.get("HERMES_VOICE_ROLLING_INTERVAL_S", "0.8"))
ROLLING_PARTIAL_MIN_AUDIO_S = float(os.environ.get("HERMES_VOICE_ROLLING_MIN_AUDIO_S", "0.55"))
ROLLING_PARTIAL_WINDOW_S = float(os.environ.get("HERMES_VOICE_ROLLING_WINDOW_S", "8.0"))
ROLLING_PARTIAL_MAX_AGE_S = float(os.environ.get("HERMES_VOICE_ROLLING_MAX_AGE_S", "0.5"))
VOICE_FAST_PROVIDER = os.environ.get("HERMES_VOICE_FAST_PROVIDER", "xai").strip().lower()
OLLAMA_URL = os.environ.get("HERMES_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_VOICE_MODEL = (
    os.environ.get("HERMES_OLLAMA_VOICE_MODEL", "qwen3:4b-instruct").strip()
    or "qwen3:4b-instruct"
)
OLLAMA_KEEP_ALIVE = os.environ.get("HERMES_OLLAMA_KEEP_ALIVE", "30m").strip() or "30m"


_ollama_keepwarm_task: asyncio.Task[Any] | None = None
_ollama_warm_state: dict[str, Any] = {"ready": False, "last_warm_at": None, "last_error": None}


async def _warm_local_voice_model() -> None:
    if VOICE_FAST_PROVIDER != "ollama":
        return
    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_VOICE_MODEL,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": OLLAMA_KEEP_ALIVE,
                },
            )
            response.raise_for_status()
        _ollama_warm_state.update({
            "ready": True,
            "last_warm_at": time.time(),
            "last_warm_seconds": round(time.time() - started, 3),
            "last_error": None,
        })
    except Exception as exc:
        _ollama_warm_state.update({
            "ready": False,
            "last_error": _clean_text(str(exc))[-240:],
        })


async def _ollama_keepwarm_loop() -> None:
    while True:
        await _warm_local_voice_model()
        await asyncio.sleep(20 * 60)


async def start_local_voice_keepwarm() -> None:
    # Keep the local model warm whenever it's the active fast provider OR the
    # xAI fast path's fallback, so a fallback turn isn't cold. Self-disables if
    # Ollama isn't reachable (the warm call just records an error).
    global _ollama_keepwarm_task
    if VOICE_FAST_PROVIDER in ("ollama", "xai"):
        _ollama_keepwarm_task = asyncio.create_task(_ollama_keepwarm_loop())


async def stop_local_voice_keepwarm() -> None:
    global _ollama_keepwarm_task
    if _ollama_keepwarm_task is not None:
        _ollama_keepwarm_task.cancel()
        try:
            await _ollama_keepwarm_task
        except asyncio.CancelledError:
            pass
        _ollama_keepwarm_task = None


@asynccontextmanager
async def voice_room_lifespan(_app: FastAPI):
    await start_local_voice_keepwarm()
    try:
        yield
    finally:
        await stop_local_voice_keepwarm()


app = FastAPI(title="Hermes Voice Room", lifespan=voice_room_lifespan)

# The frontend may be served from Vercel while this backend stays on the Mac;
# the password / device session (not open CORS) is the access gate.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

# Public shell (Vercel) + local HTTPS only. ngrok is the API host, not a browser origin.
_VOICE_CORS_ORIGIN_REGEX = os.environ.get(
    "HERMES_VOICE_CORS_ORIGIN_REGEX",
    r"https://([a-z0-9-]+\.)*vercel\.app|https?://(localhost|127\.0\.0\.1)(:\d+)?",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_VOICE_CORS_ORIGIN_REGEX,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.middleware("http")
async def auth_context_middleware(request: Request, call_next: Any) -> Any:
    """Password OR remembered device OR truly local client.

    Local/LAN UI on the Mac can skip password. Public tunnels (ngrok Host) always
    require password or a device session — even when the TCP peer is 127.0.0.1.
    """
    token = _REQUEST_AUTH_OK.set(False)
    try:
        if request.url.path.startswith("/api/") and request.url.path != "/api/auth/session":
            password = request.headers.get("x-hermes-voice-password")
            session_token = (
                request.headers.get("x-hermes-device-session")
                or request.cookies.get("hermes_device_session")
                or ""
            )
            local_ok = _connection_is_local(request=request)
            _REQUEST_AUTH_OK.set(
                local_ok
                or VOICE_PUBLIC_NO_AUTH
                or _password_ok(password)
                or _device_session_ok(session_token)
            )
        else:
            _REQUEST_AUTH_OK.set(True)
        return await call_next(request)
    finally:
        _REQUEST_AUTH_OK.reset(token)


class TextTurnRequest(BaseModel):
    text: str
    speak: bool = False


class ThreadCreateRequest(BaseModel):
    title: str | None = None
    engine: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    engine_session_id: str | None = None


class ThreadPatchRequest(BaseModel):
    title: str | None = None
    archived: bool | None = None
    mode: str | None = None
    engine: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None


class EngineDefaultRequest(BaseModel):
    engine: str


class RealtimeSessionRequest(BaseModel):
    thread_id: str | None = None


class AuthSessionRequest(BaseModel):
    password: str
    device_id: str | None = None
    device_label: str | None = None


class HermesConfigPatchRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    approvals_mode: str | None = None
    thread_id: str | None = None


class ToolsetsPatchRequest(BaseModel):
    platform: str = "cli"
    toolsets: list[str]
    thread_id: str | None = None


class McpServerRequest(BaseModel):
    name: str
    transport: str = "stdio"
    command: str | None = None
    args: str | list[str] | None = None
    env: str | dict[str, str] | None = None
    url: str | None = None
    headers: str | dict[str, str] | None = None
    timeout: int | None = None
    connect_timeout: int | None = None
    keepalive_interval: int | None = None
    thread_id: str | None = None


class InterruptedTurn(Exception):
    """Raised when the active turn was intentionally cancelled."""


class VoiceRoomStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    hermes_session_id TEXT,
                    goal TEXT NOT NULL DEFAULT '',
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    reply TEXT NOT NULL,
                    latency_seconds REAL NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    interrupted INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES threads(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deleted_threads (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    engine TEXT NOT NULL DEFAULT 'hermes',
                    snapshot_json TEXT,
                    session_id TEXT,
                    deleted_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS turns_thread_created_idx ON turns(thread_id, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS threads_updated_idx ON threads(archived, updated_at)")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(threads)")}
            if "mode" not in columns:
                conn.execute("ALTER TABLE threads ADD COLUMN mode TEXT NOT NULL DEFAULT 'fast'")
            if "external" not in columns:
                # 1 = chat imported from a Hermes desktop/terminal session; its
                # history lives in state.db, not the local turns table.
                conn.execute("ALTER TABLE threads ADD COLUMN external INTEGER NOT NULL DEFAULT 0")
            if "engine" not in columns:
                # which local agent answers this thread (hermes/grok/codex/claude);
                # existing threads keep their historical hermes behavior.
                conn.execute("ALTER TABLE threads ADD COLUMN engine TEXT NOT NULL DEFAULT 'hermes'")
            if "engine_session_id" not in columns:
                # CLI session id of the non-hermes engine, so every turn resumes
                # the same agent conversation (grok -r / codex exec resume / claude --resume).
                conn.execute("ALTER TABLE threads ADD COLUMN engine_session_id TEXT")
            if "model" not in columns:
                # Optional per-thread model override for CLI-backed agents.
                # Hermes keeps its provider/model selection in hermes-config.
                conn.execute("ALTER TABLE threads ADD COLUMN model TEXT")
            if "reasoning_effort" not in columns:
                # Per-thread CLI reasoning setting. It is validated against the
                # selected model's live catalog before being persisted.
                conn.execute("ALTER TABLE threads ADD COLUMN reasoning_effort TEXT")

    def ensure_default_thread(self) -> dict[str, Any]:
        threads = self.list_threads(include_archived=False)
        if threads:
            return threads[0]
        return self.create_thread("Voice Room")

    def create_thread(
        self,
        title: str | None = None,
        engine: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        engine_session_id: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        thread_id = str(uuid.uuid4())
        clean_title = (title or "").strip() or f"Voice Room {time.strftime('%H:%M')}"
        wanted = (engine or "").strip().lower() or _default_engine()
        if wanted not in AGENT_ENGINES:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {wanted}")
        if engine is not None and not _ensure_engine_available(wanted):
            raise HTTPException(status_code=400, detail=f"Engine {wanted} is unavailable: {_engine_unavailable_note(wanted)}")
        if engine is None and not _ensure_engine_available(wanted):
            wanted = "hermes"
        selected_model, selected_effort = _normalize_engine_selection(
            wanted,
            model=model,
            reasoning_effort=reasoning_effort,
            choose_default=True,
        )
        sid = (engine_session_id or "").strip() or None
        with self.lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO threads
                  (id, title, hermes_session_id, goal, archived, engine, model, reasoning_effort, engine_session_id, created_at, updated_at)
                VALUES (?, ?, NULL, '', 0, ?, ?, ?, ?, ?, ?)
                """,
                (thread_id, clean_title, wanted, selected_model, selected_effort, sid, now, now),
            )
        return self.get_thread(thread_id)

    def import_external_thread(self, session_id: str, title: str) -> dict[str, Any]:
        # a chat that originated in the Hermes desktop/terminal app; its history
        # lives in state.db (external=1), and continuing it resumes that session.
        now = _now()
        thread_id = str(uuid.uuid4())
        clean_title = (title or "").strip() or "Imported chat"
        with self.lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO threads (id, title, hermes_session_id, goal, archived, external, created_at, updated_at)
                VALUES (?, ?, ?, '', 0, 1, ?, ?)
                """,
                (thread_id, clean_title, session_id, now, now),
            )
        return self.get_thread(thread_id)

    def get_thread(self, thread_id: str) -> dict[str, Any]:
        with self.lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Voice room thread not found")
        return _thread_row(row)

    def list_threads(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE archived = 0"
        with self.lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM threads {where} ORDER BY updated_at DESC, created_at DESC LIMIT 80"
            ).fetchall()
        return [_thread_row(row) for row in rows]

    def patch_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        archived: bool | None = None,
        mode: str | None = None,
        engine: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        thread = self.get_thread(thread_id)
        new_title = thread["title"] if title is None else (title.strip() or thread["title"])
        new_archived = thread["archived"] if archived is None else bool(archived)
        new_mode = thread["mode"] if mode is None else mode
        if new_mode not in THREAD_MODES:
            raise HTTPException(status_code=400, detail=f"Invalid thread mode: {new_mode}")
        new_engine = thread.get("engine") or "hermes"
        new_engine_session = thread.get("engine_session_id")
        new_model = thread.get("model")
        new_reasoning_effort = thread.get("reasoning_effort")
        if engine is not None:
            wanted = engine.strip().lower()
            if wanted not in AGENT_ENGINES:
                raise HTTPException(status_code=400, detail=f"Unknown engine: {wanted}")
            if not _ensure_engine_available(wanted):
                note = _engine_unavailable_note(wanted)
                raise HTTPException(status_code=400, detail=f"Engine {wanted} is unavailable: {note}")
            if wanted != new_engine:
                # a different agent can't continue another agent's session
                new_engine_session = None
                new_model, new_reasoning_effort = _normalize_engine_selection(
                    wanted,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    choose_default=True,
                )
            new_engine = wanted
        if model is not None and (engine is None or new_engine == thread.get("engine")):
            new_model, default_effort = _normalize_engine_selection(
                new_engine,
                model=model,
                reasoning_effort=reasoning_effort,
                choose_default=True,
            )
            new_reasoning_effort = default_effort
        elif reasoning_effort is not None and (engine is None or new_engine == thread.get("engine")):
            new_model, new_reasoning_effort = _normalize_engine_selection(
                new_engine,
                model=new_model,
                reasoning_effort=reasoning_effort,
            )
        now = _now()
        with self.lock, self._connect() as conn:
            conn.execute(
                "UPDATE threads SET title = ?, archived = ?, mode = ?, engine = ?, model = ?, reasoning_effort = ?, engine_session_id = ?, "
                "updated_at = ? WHERE id = ?",
                (
                    new_title,
                    1 if new_archived else 0,
                    new_mode,
                    new_engine,
                    new_model,
                    new_reasoning_effort,
                    new_engine_session,
                    now,
                    thread_id,
                ),
            )
        return self.get_thread(thread_id)

    def export_thread(self, thread_id: str) -> dict[str, Any] | None:
        # full snapshot (thread row + all turns) so a delete can be undone
        with self.lock, self._connect() as conn:
            trow = conn.execute("SELECT * FROM threads WHERE id = ?", (thread_id,)).fetchone()
            if not trow:
                return None
            turns = conn.execute(
                "SELECT * FROM turns WHERE thread_id = ? ORDER BY created_at", (thread_id,)
            ).fetchall()
        return {"thread": dict(trow), "turns": [dict(r) for r in turns]}

    def restore_thread(self, snap: dict[str, Any]) -> dict[str, Any]:
        t = snap["thread"]
        # column names come from our own schema (trusted); values are bound
        with self.lock, self._connect() as conn:
            cols = ",".join(t.keys())
            ph = ",".join("?" for _ in t)
            conn.execute(f"INSERT OR REPLACE INTO threads ({cols}) VALUES ({ph})", tuple(t.values()))
            for turn in snap.get("turns", []):
                tcols = ",".join(turn.keys())
                tph = ",".join("?" for _ in turn)
                conn.execute(f"INSERT OR REPLACE INTO turns ({tcols}) VALUES ({tph})", tuple(turn.values()))
        return self.get_thread(t["id"])

    def delete_thread(self, thread_id: str) -> None:
        # hard delete: remove the thread and all its turns
        with self.lock, self._connect() as conn:
            conn.execute("DELETE FROM turns WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))

    def add_deleted_thread(
        self,
        *,
        thread_id: str,
        title: str,
        engine: str,
        snapshot: dict[str, Any] | None,
        session_id: str,
        deleted_at: float,
        expires_at: float,
    ) -> None:
        encoded = json.dumps(snapshot, separators=(",", ":")) if snapshot else None
        with self.lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO deleted_threads "
                "(id,title,engine,snapshot_json,session_id,deleted_at,expires_at) VALUES (?,?,?,?,?,?,?)",
                (thread_id, title, engine or "hermes", encoded, session_id or None, deleted_at, expires_at),
            )

    def purge_expired_deleted_threads(self, now: float | None = None) -> None:
        cutoff = time.time() if now is None else now
        with self.lock, self._connect() as conn:
            conn.execute("DELETE FROM deleted_threads WHERE expires_at <= ?", (cutoff,))

    def list_deleted_threads(self) -> list[dict[str, Any]]:
        self.purge_expired_deleted_threads()
        with self.lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id,title,engine,deleted_at,expires_at FROM deleted_threads ORDER BY deleted_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_deleted_thread(self, thread_id: str) -> dict[str, Any]:
        self.purge_expired_deleted_threads()
        with self.lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM deleted_threads WHERE id = ?", (thread_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Deleted chat is no longer recoverable")
        result = dict(row)
        result["snapshot"] = json.loads(result.pop("snapshot_json")) if result.get("snapshot_json") else None
        return result

    def remove_deleted_thread(self, thread_id: str) -> None:
        with self.lock, self._connect() as conn:
            conn.execute("DELETE FROM deleted_threads WHERE id = ?", (thread_id,))

    def set_goal(self, thread_id: str, goal: str) -> dict[str, Any]:
        now = _now()
        with self.lock, self._connect() as conn:
            conn.execute(
                "UPDATE threads SET goal = ?, updated_at = ? WHERE id = ?",
                (goal, now, thread_id),
            )
        return self.get_thread(thread_id)

    def set_hermes_session(self, thread_id: str, hermes_session_id: str) -> dict[str, Any]:
        now = _now()
        with self.lock, self._connect() as conn:
            conn.execute(
                "UPDATE threads SET hermes_session_id = ?, updated_at = ? WHERE id = ?",
                (hermes_session_id, now, thread_id),
            )
        return self.get_thread(thread_id)

    def set_engine_session(self, thread_id: str, engine_session_id: str | None) -> dict[str, Any]:
        now = _now()
        with self.lock, self._connect() as conn:
            conn.execute(
                "UPDATE threads SET engine_session_id = ?, updated_at = ? WHERE id = ?",
                (engine_session_id, now, thread_id),
            )
        return self.get_thread(thread_id)

    def recent_turns(self, thread_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self.lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM turns
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return list(reversed([_turn_row(row) for row in rows]))

    def add_turn(
        self,
        *,
        thread_id: str,
        mode: str,
        user_text: str,
        reply: str,
        started: float,
        metrics: dict[str, Any] | None = None,
        interrupted: bool = False,
        mirror: bool = True,
    ) -> dict[str, Any]:
        now = _now()
        record = {
            "id": str(uuid.uuid4()),
            "thread_id": thread_id,
            "mode": mode,
            "transcript": user_text,
            "reply": reply or "I processed that, but did not get a response.",
            "latency_seconds": round(time.time() - started, 2),
            "metrics": metrics or {},
            "interrupted": interrupted,
            "at": int(started),
        }
        with self.lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO turns
                  (id, thread_id, mode, user_text, reply, latency_seconds, metrics_json, interrupted, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    thread_id,
                    mode,
                    user_text,
                    record["reply"],
                    record["latency_seconds"],
                    json.dumps(record["metrics"], ensure_ascii=False),
                    1 if interrupted else 0,
                    now,
                ),
            )
            conn.execute("UPDATE threads SET updated_at = ? WHERE id = ?", (now, thread_id))
        # Mirror fast-mode turns into the shared Hermes session store so the chat
        # shows up in the desktop/terminal apps too. Agent-mode already persists
        # via the Hermes CLI. Best-effort — never breaks turn saving.
        if mirror and not interrupted:
            try:
                _mirror_web_turn(thread_id, mode, user_text, reply, record["metrics"])
            except Exception as exc:
                print(f"[sync] mirror hook failed: {exc}")
        return record

    def merge_turn_metrics(self, turn_id: str, thread_id: str, values: dict[str, float]) -> bool:
        with self.lock, self._connect() as conn:
            row = conn.execute(
                "SELECT metrics_json FROM turns WHERE id = ? AND thread_id = ?",
                (turn_id, thread_id),
            ).fetchone()
            if not row:
                return False
            try:
                metrics = json.loads(row["metrics_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                metrics = {}
            metrics.update(values)
            metrics["client_metrics_received_at"] = time.time()
            conn.execute(
                "UPDATE turns SET metrics_json = ? WHERE id = ? AND thread_id = ?",
                (json.dumps(metrics, ensure_ascii=False), turn_id, thread_id),
            )
        return True


class RuntimeState:
    def __init__(self) -> None:
        self._thread_locks_guard = threading.Lock()
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._active_guard = threading.RLock()
        self._active_processes: dict[str, subprocess.Popen[str]] = {}
        self._interrupted_at: dict[str, float] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}

    def cancel_event(self, thread_id: str) -> asyncio.Event:
        with self._active_guard:
            event = self._cancel_events.get(thread_id)
            if event is None:
                event = asyncio.Event()
                self._cancel_events[thread_id] = event
            return event

    def reset_cancel(self, thread_id: str) -> asyncio.Event:
        with self._active_guard:
            event = asyncio.Event()
            self._cancel_events[thread_id] = event
            return event

    def trigger_cancel(self, thread_id: str) -> None:
        with self._active_guard:
            event = self._cancel_events.get(thread_id)
        if event is not None:
            event.set()

    def thread_lock(self, thread_id: str) -> asyncio.Lock:
        with self._thread_locks_guard:
            lock = self._thread_locks.get(thread_id)
            if lock is None:
                lock = asyncio.Lock()
                self._thread_locks[thread_id] = lock
            return lock

    def set_active(self, thread_id: str, proc: subprocess.Popen[str]) -> None:
        with self._active_guard:
            self._active_processes[thread_id] = proc

    def clear_active(self, thread_id: str, proc: subprocess.Popen[str]) -> None:
        with self._active_guard:
            if self._active_processes.get(thread_id) is proc:
                self._active_processes.pop(thread_id, None)

    def mark_interrupted(self, thread_id: str) -> None:
        with self._active_guard:
            self._interrupted_at[thread_id] = time.time()

    def was_interrupted_since(self, thread_id: str, started: float) -> bool:
        with self._active_guard:
            return self._interrupted_at.get(thread_id, 0) >= started

    def interrupt(self, thread_id: str) -> dict[str, Any]:
        self.mark_interrupted(thread_id)
        with self._active_guard:
            proc = self._active_processes.get(thread_id)
        killed = False
        if proc is not None and proc.poll() is None:
            killed = _terminate_process_tree(proc)
        return {"ok": True, "thread_id": thread_id, "killed_process": killed}


store = VoiceRoomStore(DB_PATH)
runtime = RuntimeState()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (ROOT / "static/index.html").read_text(encoding="utf-8")


# PWA assets referenced relatively from index.html, so the same markup works
# on this server and on the Vercel-hosted copy (files at site root there).
@app.get("/manifest.webmanifest")
async def manifest() -> Any:
    from fastapi.responses import FileResponse

    return FileResponse(ROOT / "static/manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
async def service_worker() -> Any:
    from fastapi.responses import FileResponse

    return FileResponse(ROOT / "static/sw.js", media_type="text/javascript")


@app.get("/icon-180.png")
async def icon_180() -> Any:
    from fastapi.responses import FileResponse

    return FileResponse(ROOT / "static/icon-180.png", media_type="image/png")


@app.get("/icon-512.png")
async def icon_512() -> Any:
    from fastapi.responses import FileResponse

    return FileResponse(ROOT / "static/icon-512.png", media_type="image/png")


@app.post("/api/auth/session")
async def create_auth_session(payload: AuthSessionRequest) -> JSONResponse:
    if not _password_ok(payload.password):
        raise HTTPException(status_code=401, detail="Login failed. Check the voice room password.")
    session = _issue_device_session(payload.device_id, payload.device_label)
    response = JSONResponse({"ok": True, **session})
    response.set_cookie(
        "hermes_device_session",
        session["session_token"],
        max_age=DEVICE_SESSION_TTL_SECONDS,
        secure=True,
        httponly=True,
        samesite="none",
        path="/",
    )
    return response


class VoiceRuntimePatchRequest(BaseModel):
    fast_provider: str | None = None  # xai | ollama | default
    stt_engine: str | None = None  # whisper_flow | browser | auto
    allow_ollama_fallback: bool | None = None
    stt_language: str | None = None


@app.get("/api/voice-runtime")
async def get_voice_runtime(
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    runtime = _load_voice_runtime()
    return {
        "ok": True,
        "runtime": runtime,
        "effective": {
            "fast_provider": _effective_fast_provider(),
            "stt_engine": _effective_stt_engine(),
            "whisper_flow_reachable": _whisper_server_reachable(),
            "ollama_model": OLLAMA_VOICE_MODEL,
            "env_fast_provider": VOICE_FAST_PROVIDER,
        },
        "options": {
            "fast_provider": [
                {"id": "xai", "label": "Grok (xAI / Hermes config)", "hint": "Uses the model you Apply in Settings (e.g. grok-4.5)."},
                {"id": "ollama", "label": f"Local Ollama ({OLLAMA_VOICE_MODEL})", "hint": "Runs entirely on this Mac. No cloud."},
            ],
            "stt_engine": [
                {"id": "whisper_flow", "label": "Whisper Flow (local, accurate)", "hint": "Uses your local whisper.cpp quality model on :12712."},
                {"id": "browser", "label": "Natural browser speech", "hint": "Device speech recognition — fast conversational, no Whisper upload."},
                {"id": "auto", "label": "Auto", "hint": "Whisper Flow when available, otherwise Hermes STT fallbacks."},
            ],
        },
    }


@app.patch("/api/voice-runtime")
async def patch_voice_runtime(
    payload: VoiceRuntimePatchRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    runtime = _save_voice_runtime(payload.model_dump(exclude_unset=True))
    # Keep local model warm when Ollama is selected.
    if _effective_fast_provider() == "ollama":
        try:
            asyncio.get_running_loop().create_task(_warm_local_voice_model())
        except Exception:
            pass
    return {
        "ok": True,
        "runtime": runtime,
        "effective": {
            "fast_provider": _effective_fast_provider(),
            "stt_engine": _effective_stt_engine(),
            "whisper_flow_reachable": _whisper_server_reachable(),
            "ollama_model": OLLAMA_VOICE_MODEL,
        },
    }


@app.get("/api/agent-activity/fixtures")
async def agent_activity_fixtures(
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    """Deterministic execution streams for UI/dev verification (no Hermes calls)."""
    _require_password(x_hermes_voice_password)
    from engines.agent_activity import mock_stream_scenarios

    return {"ok": True, "scenarios": mock_stream_scenarios()}


@app.get("/api/agent-activity/fixtures/{name}")
async def agent_activity_fixture(
    name: str,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    from engines.agent_activity import apply_events, mock_stream_scenarios

    scenarios = mock_stream_scenarios()
    if name not in scenarios:
        raise HTTPException(status_code=404, detail=f"Unknown fixture: {name}")
    events = scenarios[name]
    return {"ok": True, "name": name, "events": events, "final_state": apply_events(events)}


@app.get("/api/health")
async def health(
    request: Request,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    default_thread = store.ensure_default_thread()
    local_client = _connection_is_local(request=request)
    tunnel_url = await _ngrok_tunnel_url()
    return {
        "ok": True,
        "agent_activity": True,
        "tunnel_url": tunnel_url,
        # Local/LAN only skips password. Public/ngrok always requires auth.
        "auth_required": not local_client and not VOICE_PUBLIC_NO_AUTH,
        "local_client": local_client,
        "password_configured": bool(VOICE_PASSWORD),
        "hermes_python": str(HERMES_PYTHON),
        "hermes_agent": str(HERMES_AGENT),
        "cwd": str(DEFAULT_CWD),
        "voice": DEFAULT_VOICE,
        "tts": {
            "engine": DEFAULT_TTS_ENGINE,
            "kokoro_available": _kokoro_available(),
            "kokoro_service_ready": _kokoro_service_ready(),
            "kokoro_service_url": KOKORO_SERVICE_URL,
            "kokoro_voice": KOKORO_VOICE,
            "macos_fallback_voice": DEFAULT_VOICE,
        },
        "db": str(DB_PATH),
        "default_thread_id": default_thread["id"],
        "stt": {
            "preferred_provider": VOICE_STT_PROVIDER,
            "fallbacks": list(VOICE_STT_FALLBACKS),
            "whisper_server_url": WHISPER_SERVER_URL,
            "local_model": VOICE_STT_LOCAL_MODEL,
            "xai_model": VOICE_STT_XAI_MODEL,
            "openai_model": VOICE_STT_OPENAI_MODEL,
            "engine": _effective_stt_engine(),
            "whisper_flow_reachable": _whisper_server_reachable(),
        },
        "fast_voice": {
            "provider": _effective_fast_provider(),
            "env_default": VOICE_FAST_PROVIDER,
            "ollama_url": OLLAMA_URL,
            "ollama_model": OLLAMA_VOICE_MODEL,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "warm": dict(_ollama_warm_state),
            "allow_ollama_fallback": bool(_load_voice_runtime().get("allow_ollama_fallback")),
        },
        "voice_runtime": _load_voice_runtime(),
        "realtime": {
            "configured": True,
            "enabled": True,
            "provider": "local",
            "pipeline": "whisper.cpp (Whisper Flow) → Fast model → Kokoro",
            "transport": "websocket",
            "continuous_input_streaming": True,
            "endpoint_prefetch": True,
            "rolling_partials": True,
            "rolling_interval_seconds": ROLLING_PARTIAL_INTERVAL_S,
        },
        "agents": _engines_payload(),
    }


@app.get("/api/threads")
async def list_threads(
    include_archived: bool = False,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    default_thread = store.ensure_default_thread()
    threads = store.list_threads(include_archived=include_archived)
    # Unify with Hermes desktop/terminal chats (state.db). A web thread already
    # linked to a session hides its duplicate session entry.
    linked = {t["hermes_session_id"] for t in threads if t.get("hermes_session_id")}
    externals = [s for s in _list_hermes_sessions(include_archived) if s["hermes_session_id"] not in linked]
    externals += _list_cli_sessions()
    merged = threads + externals
    merged.sort(key=lambda t: t.get("updated_at") or 0, reverse=True)
    return {"ok": True, "threads": merged, "default_thread_id": default_thread["id"]}


@app.get("/api/hermes-config")
async def hermes_config(x_hermes_voice_password: str | None = Header(default=None)) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    return _hermes_config_payload()


@app.patch("/api/hermes-config")
async def patch_hermes_config(
    payload: HermesConfigPatchRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    return await asyncio.to_thread(_patch_hermes_config, payload)


@app.get("/api/tools-config")
async def tools_config(x_hermes_voice_password: str | None = Header(default=None)) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    return await asyncio.to_thread(_tools_config_payload)


@app.patch("/api/tools-config/toolsets")
async def patch_toolsets(
    payload: ToolsetsPatchRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)

    def _patch() -> dict[str, Any]:
        platform = (payload.platform or "cli").strip().lower()
        if platform != "cli":
            raise HTTPException(status_code=400, detail="This web panel currently edits the CLI/voice agent platform only")
        clean_toolsets: list[str] = []
        for item in payload.toolsets or []:
            toolset = str(item or "").strip()
            if not toolset:
                continue
            if toolset not in AVAILABLE_TOOLSET_IDS:
                raise HTTPException(status_code=400, detail=f"Unknown toolset: {toolset}")
            if toolset not in clean_toolsets:
                clean_toolsets.append(toolset)
        if not clean_toolsets:
            raise HTTPException(status_code=400, detail="Select at least one toolset")
        yaml, data = _load_hermes_config_document()
        platform_cfg = _config_mapping(data, "platform_toolsets")
        platform_cfg[platform] = clean_toolsets
        _save_hermes_config_document(yaml, data)
        _maybe_reset_thread_session(payload.thread_id)
        result = _tools_config_payload(data)
        result["changed"] = True
        return result

    return await asyncio.to_thread(_patch)


@app.post("/api/mcp-servers")
async def upsert_mcp_server(
    payload: McpServerRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)

    def _upsert() -> dict[str, Any]:
        name = _sanitize_mcp_name(payload.name)
        transport = (payload.transport or "stdio").strip().lower()
        entry: dict[str, Any] = {}
        if transport == "stdio":
            command = str(payload.command or "").strip()
            if not command:
                raise HTTPException(status_code=400, detail="Command is required for stdio MCP servers")
            entry["command"] = command
            args = _parse_args(payload.args)
            if args:
                entry["args"] = args
            env = _parse_key_values(payload.env, "Environment")
            if env:
                entry["env"] = env
        elif transport == "http":
            url = str(payload.url or "").strip()
            if not re.match(r"^https?://", url):
                raise HTTPException(status_code=400, detail="HTTP MCP URL must start with http:// or https://")
            entry["url"] = url
            headers = _parse_key_values(payload.headers, "Headers")
            if headers:
                entry["headers"] = headers
        else:
            raise HTTPException(status_code=400, detail="Transport must be stdio or http")

        for key in ("timeout", "connect_timeout", "keepalive_interval"):
            value = getattr(payload, key)
            if value is None:
                continue
            if int(value) <= 0:
                raise HTTPException(status_code=400, detail=f"{key} must be positive")
            entry[key] = int(value)

        yaml, data = _load_hermes_config_document()
        servers = _config_mapping(data, "mcp_servers")
        servers[name] = entry
        _save_hermes_config_document(yaml, data)
        _maybe_reset_thread_session(payload.thread_id)
        result = _tools_config_payload(data)
        result["changed"] = True
        result["server"] = _mcp_server_summary(name, entry)
        return result

    return await asyncio.to_thread(_upsert)


@app.delete("/api/mcp-servers/{server_name}")
async def delete_mcp_server(
    server_name: str,
    thread_id: str | None = None,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)

    def _delete() -> dict[str, Any]:
        name = _sanitize_mcp_name(server_name)
        yaml, data = _load_hermes_config_document()
        servers = data.get("mcp_servers") if isinstance(data.get("mcp_servers"), dict) else {}
        if name not in servers:
            raise HTTPException(status_code=404, detail=f"MCP server not found: {name}")
        servers.pop(name, None)
        data["mcp_servers"] = servers
        _save_hermes_config_document(yaml, data)
        _maybe_reset_thread_session(thread_id)
        result = _tools_config_payload(data)
        result["changed"] = True
        return result

    return await asyncio.to_thread(_delete)


@app.post("/api/threads")
async def create_thread(
    payload: ThreadCreateRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    thread = store.create_thread(
        payload.title,
        engine=payload.engine,
        model=payload.model,
        reasoning_effort=payload.reasoning_effort,
        engine_session_id=payload.engine_session_id,
    )
    return {"ok": True, "thread": thread}


@app.patch("/api/threads/{thread_id}")
async def patch_thread(
    thread_id: str,
    payload: ThreadPatchRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    thread = store.patch_thread(
        thread_id,
        title=payload.title,
        archived=payload.archived,
        mode=payload.mode,
        engine=payload.engine,
        model=payload.model,
        reasoning_effort=payload.reasoning_effort,
    )
    if payload.title is not None and thread.get("hermes_session_id"):
        _rename_hermes_session_quietly(thread["hermes_session_id"], thread["title"])
    return {"ok": True, "thread": thread}


@app.get("/api/engines")
async def list_engines(
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    return {"ok": True, **await asyncio.to_thread(_engines_payload)}


@app.get("/api/engines/{engine_id}")
async def get_engine_status(
    engine_id: str,
    refresh: bool = False,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    engine_id = engine_id.strip().lower()
    info = AGENT_ENGINES.get(engine_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Unknown engine: {engine_id}")
    probe = await asyncio.to_thread(_engine_probe, engine_id, refresh=refresh)
    return {
        "ok": True,
        "id": engine_id,
        "label": info["label"],
        "detail": info["detail"],
        "available": bool(probe.get("available")),
        "startable": _engine_startable(engine_id),
        "poll": bool(info.get("poll")),
        "note": str(probe.get("note") or ""),
        "checked_at": probe.get("checked_at"),
    }


@app.post("/api/engines/{engine_id}/launch")
async def launch_engine_app(
    engine_id: str,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    """Open the engine's Mac app remotely (phone-initiated). CLI engines have
    nothing to open — they are spawned per turn."""
    _require_password(x_hermes_voice_password)
    engine_id = engine_id.strip().lower()
    if engine_id not in AGENT_ENGINES:
        raise HTTPException(status_code=404, detail=f"Unknown engine: {engine_id}")
    if engine_id == "hermes":
        cmd = ["open", "-a", "Hermes"]
    elif engine_id == "antigravity":
        cmd = ["open", str(Path.home() / "Desktop" / "Antigravity.app")]
    elif engine_id == "codex":
        cmd = ["open", "-a", "ChatGPT"]
    elif engine_id == "grok":
        # A .command file opens Terminal and runs grok — no automation perms needed.
        # Optional ?session_id= on this endpoint resumes that Grok session in Terminal.
        script = DATA_DIR / "launch-grok.command"
        # session_id may be passed as a query param via FastAPI if present on request scope —
        # launch endpoint currently has no query; open plain grok in DEFAULT_CWD.
        cwd_q = str(DEFAULT_CWD).replace('"', '\\"')
        script.write_text(
            f'#!/bin/zsh\ncd "{cwd_q}"\nexec "$HOME/.grok/bin/grok"\n',
            encoding="utf-8",
        )
        script.chmod(0o755)
        cmd = ["open", str(script)]
    else:
        return {"ok": True, "launched": False,
                "note": f"{AGENT_ENGINES[engine_id]['label']} has no launchable Mac app."}
    proc = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=(proc.stderr or "launch failed").strip()[-200:])
    return {"ok": True, "launched": True,
            "note": f"Opening {AGENT_ENGINES[engine_id]['label']} on the Mac."}


@app.get("/api/engines/{engine_id}/models")
async def get_engine_models(
    engine_id: str,
    refresh: bool = False,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    engine_id = engine_id.strip().lower()
    if engine_id not in AGENT_ENGINES:
        raise HTTPException(status_code=404, detail=f"Unknown engine: {engine_id}")
    try:
        payload = await asyncio.to_thread(_engine_model_payload, engine_id, refresh=refresh)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load {engine_id} models: {exc}") from exc
    return {"ok": True, **payload}


@app.post("/api/engines")
async def set_default_engine(
    payload: EngineDefaultRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    wanted = payload.engine.strip().lower()
    if wanted not in AGENT_ENGINES:
        raise HTTPException(status_code=400, detail=f"Unknown engine: {wanted}")
    if not await asyncio.to_thread(_ensure_engine_available, wanted):
        raise HTTPException(status_code=400, detail=f"Engine {wanted} is unavailable: {_engine_unavailable_note(wanted)}")
    _set_default_engine(wanted)
    return {"ok": True, **await asyncio.to_thread(_engines_payload)}


class GrokAttachRequest(BaseModel):
    session_id: str
    title: str | None = None
    thread_id: str | None = None


class GrokNewSessionRequest(BaseModel):
    title: str | None = None


@app.get("/api/engines/grok/sessions")
async def list_grok_sessions(
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    """List Grok Build CLI sessions (including live Terminal) for remote attach.

    Does not use Hermes.app. Phone picks a session and continues with full tools.
    """
    _require_password(x_hermes_voice_password)
    sessions = await asyncio.to_thread(_list_grok_sessions_detailed)
    return {
        "ok": True,
        "engine": "grok",
        "writable": True,
        "cwd": str(DEFAULT_CWD),
        "sessions": sessions,
        "live_count": sum(1 for s in sessions if s.get("live")),
        "bin": GROK_BIN if Path(GROK_BIN).exists() else None,
    }


@app.post("/api/engines/grok/sessions/attach")
async def attach_grok_session(
    payload: GrokAttachRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    """Attach a Grok session for full remote control from the web app (writable)."""
    _require_password(x_hermes_voice_password)
    return await asyncio.to_thread(
        _attach_grok_session,
        payload.session_id,
        title=payload.title,
        thread_id=payload.thread_id,
    )


@app.post("/api/engines/grok/sessions/new")
async def new_grok_session(
    payload: GrokNewSessionRequest | None = None,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    """Start a new Grok Build conversation from the phone (writable, session id after first turn)."""
    _require_password(x_hermes_voice_password)
    body = payload or GrokNewSessionRequest()
    return await asyncio.to_thread(_new_grok_session_thread, body.title)


@app.post("/api/engines/grok/sessions/{session_id}/open-terminal")
async def open_grok_session_in_terminal(
    session_id: str,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    """Open Terminal.app on the Mac resumed to this Grok session (optional companion view)."""
    _require_password(x_hermes_voice_password)
    sid = session_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    if not Path(GROK_BIN).exists():
        raise HTTPException(status_code=503, detail="Grok CLI not found on this Mac")
    script = DATA_DIR / f"launch-grok-{sid[:8]}.command"
    cwd_q = str(DEFAULT_CWD).replace('"', '\"')
    bin_q = GROK_BIN.replace('"', '\"')
    script.write_text(
        f'#!/bin/zsh\ncd "{cwd_q}"\nexec "{bin_q}" --resume "{sid}"\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    proc = await asyncio.to_thread(
        subprocess.run, ["open", str(script)], capture_output=True, text=True, timeout=20
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=(proc.stderr or "open failed").strip()[-200:])
    return {"ok": True, "launched": True, "session_id": sid, "note": "Opening Terminal with Grok resumed."}


@app.post("/api/threads/import")
async def import_hermes_session(
    request: Request,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    """Adopt a Hermes desktop/terminal session as a web thread so it can be
    opened and continued here. Idempotent — returns the existing link if any."""
    _require_password(x_hermes_voice_password)
    body = await request.json()
    sid = str(body.get("session_id", "")).strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    for t in store.list_threads(include_archived=True):
        if t.get("hermes_session_id") == sid:
            return {"ok": True, "thread": t}
    title = "Imported chat"
    try:
        conn = _state_ro(); conn.row_factory = sqlite3.Row
        r = conn.execute("SELECT title, display_name FROM sessions WHERE id=?", (sid,)).fetchone()
        if r:
            title = ((r["title"] or r["display_name"] or "").strip() or _session_preview(conn, sid) or title)[:80]
        conn.close()
    except Exception:
        pass
    thread = store.import_external_thread(sid, title)
    return {"ok": True, "thread": thread}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    """Move a chat to durable Recently Deleted for 24 hours.

    The short undo token is intentionally separate: it expires after 30 seconds,
    while Settings can still restore the durable snapshot for one day.
    """
    _require_password(x_hermes_voice_password)
    session_id = ""
    web_snap = None
    title = "Deleted chat"
    engine = "hermes"
    if thread_id.startswith("hs:"):
        session_id = thread_id[3:]
        external = next((t for t in _list_hermes_sessions(False) if t.get("id") == thread_id), None)
        if external:
            title = external.get("title") or title
            engine = external.get("engine") or engine
    else:
        thread = store.get_thread(thread_id)   # 404 if missing
        title = thread.get("title") or title
        engine = thread.get("engine") or engine
        session_id = (thread.get("hermes_session_id") or "").strip()
        web_snap = store.export_thread(thread_id)
        store.delete_thread(thread_id)
    # hide the shared session so the delete syncs to the desktop app (reversible)
    if session_id:
        sdb = _state_store()
        if sdb:
            try:
                sdb.set_session_archived(session_id, True)
            except Exception as exc:
                print(f"[sync] session archive failed: {exc}")
    now = time.time()
    store.add_deleted_thread(
        thread_id=thread_id,
        title=title,
        engine=engine,
        snapshot=web_snap,
        session_id=session_id,
        deleted_at=now,
        expires_at=now + DELETED_THREAD_TTL,
    )
    token = uuid.uuid4().hex
    with _undo_lock:
        _undo_snapshots[token] = {"thread_id": thread_id, "ts": now}
        for k in [k for k, v in _undo_snapshots.items() if now - v["ts"] > UNDO_TTL]:
            _undo_snapshots.pop(k, None)
    return {"ok": True, "deleted": thread_id, "undo_token": token}


@app.post("/api/threads/undo")
async def undo_delete(
    request: Request,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    body = await request.json()
    token = str(body.get("token", ""))
    with _undo_lock:
        entry = _undo_snapshots.pop(token, None)
    if not entry:
        raise HTTPException(status_code=404, detail="Nothing to undo (it may have expired)")
    return _restore_deleted_chat(str(entry.get("thread_id") or ""))


def _restore_deleted_chat(thread_id: str) -> dict[str, Any]:
    entry = store.get_deleted_thread(thread_id)
    session_id = str(entry.get("session_id") or "")
    if session_id:
        sdb = _state_store()
        if sdb:
            try:
                sdb.set_session_archived(session_id, False)
            except Exception as exc:
                print(f"[sync] session unarchive failed: {exc}")
    thread = store.restore_thread(entry["snapshot"]) if entry.get("snapshot") else {
        "id": thread_id,
        "title": entry.get("title") or "Restored chat",
        "engine": entry.get("engine") or "hermes",
        "source": "desktop",
        "external": True,
        "archived": False,
        "updated_at": time.time(),
    }
    store.remove_deleted_thread(thread_id)
    return {"ok": True, "thread": thread}


@app.get("/api/deleted-threads")
async def list_deleted_threads(
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    return {"ok": True, "threads": store.list_deleted_threads()}


@app.post("/api/threads/{thread_id}/restore")
async def restore_deleted_thread(
    thread_id: str,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    return _restore_deleted_chat(thread_id)


@app.delete("/api/threads/{thread_id}/permanent")
async def permanently_delete_thread(
    thread_id: str,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    # Shared Hermes sessions stay archived in the shared store; removing this
    # durable snapshot makes the deletion permanent from Voice Room/iOS.
    store.get_deleted_thread(thread_id)
    store.remove_deleted_thread(thread_id)
    return {"ok": True, "deleted": thread_id}


@app.get("/api/voice")
async def get_voice(
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    return {"current": _current_voice(), "catalog": _voice_catalog()}


@app.post("/api/voice")
async def set_voice(
    request: Request,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    body = await request.json()
    v = _set_voice(str(body.get("engine", "")), str(body.get("voice", "")), str(body.get("lang", "a")))
    return {"ok": True, "current": v}


@app.post("/api/voice/preview")
async def preview_voice(
    request: Request,
    x_hermes_voice_password: str | None = Header(default=None),
):
    from fastapi.responses import Response

    _require_password(x_hermes_voice_password)
    body = await request.json()
    v = _validate_voice(str(body.get("engine", "")), str(body.get("voice", "")), str(body.get("lang", "a")))
    text = str(body.get("text") or "").strip()
    if not text:
        text = "أهلاً، أنا هيرمس." if v["lang"] == "ar" else "Hi, I'm Hermes. This is how I'll sound."
    try:
        audio = _synth_bytes(text, override=v)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"voice preview failed: {exc}")
    return Response(content=audio, media_type="audio/wav")


@app.get("/api/threads/{thread_id}/turns")
async def thread_turns(
    thread_id: str,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    # CLI-agent sessions (grok / codex / antigravity) live in their own stores.
    if thread_id[:3] in ("gk:", "cx:", "ag:"):
        return {"ok": True, "thread": _cli_session_stub(thread_id),
                "turns": _cli_session_turns(thread_id)}
    thread = store.get_thread(thread_id)
    # external (desktop/terminal) chats keep their history in state.db
    if thread.get("external") and thread.get("hermes_session_id"):
        turns = _session_as_turns(thread["hermes_session_id"])[-60:]
    else:
        turns = store.recent_turns(thread_id, limit=50)
    return {"ok": True, "thread": thread, "turns": turns}


@app.get("/api/turns")
async def turns(
    thread_id: str | None = None,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    thread = store.get_thread(thread_id) if thread_id else store.ensure_default_thread()
    return {
        "ok": True,
        "thread": thread,
        "turns": store.recent_turns(thread["id"], limit=20),
        "goal": thread["goal"],
    }


@app.post("/api/threads/{thread_id}/attachments")
async def upload_thread_attachments(
    thread_id: str,
    files: list[UploadFile] = File(...),
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    thread = store.get_thread(thread_id)
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    safe_thread_id = re.sub(r"[^A-Za-z0-9_.:-]+", "_", thread_id).strip("._-") or "thread"
    thread_dir = ATTACHMENTS_DIR / safe_thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)

    attachments: list[dict[str, Any]] = []
    for upload in files:
        raw = await upload.read()
        size = len(raw)
        if size > MAX_ATTACHMENT_BYTES:
            raise HTTPException(status_code=413, detail=f"Attachment too large: {upload.filename or 'file'}")

        attachment_id = f"{int(time.time())}-{uuid.uuid4().hex[:10]}"
        safe_name = _safe_attachment_name(upload.filename)
        path = thread_dir / f"{attachment_id}-{safe_name}"
        path.write_bytes(raw)

        content_type = (upload.content_type or "application/octet-stream").split(";", 1)[0].strip()
        text_preview = ""
        if _looks_like_text_attachment(content_type, safe_name):
            text_preview = raw[:TEXT_ATTACHMENT_PREVIEW_BYTES].decode("utf-8", errors="replace")

        attachments.append(
            {
                "id": attachment_id,
                "name": safe_name,
                "type": content_type or "application/octet-stream",
                "size": size,
                "path": str(path),
                "text_preview": text_preview,
            }
        )

    return {"ok": True, "thread": thread, "attachments": attachments}


@app.post("/api/realtime/session")
async def realtime_session(
    payload: RealtimeSessionRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> JSONResponse:
    _require_password(x_hermes_voice_password)
    if payload.thread_id:
        store.get_thread(payload.thread_id)
    if not _openai_key_configured():
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "reason": "openai_key_missing",
                "message": "Realtime voice is not configured yet. The local Hermes fallback is active.",
            },
        )
    return JSONResponse(
        status_code=501,
        content={
            "ok": False,
            "reason": "realtime_provider_pending",
            "message": "OpenAI Realtime credentials exist, but the provider bridge is intentionally left for the later integration step.",
        },
    )


@app.post("/api/threads/{thread_id}/interrupt")
async def interrupt_thread(
    thread_id: str,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    store.get_thread(thread_id)
    runtime.trigger_cancel(thread_id)
    return await asyncio.to_thread(runtime.interrupt, thread_id)


@app.post("/api/interrupt")
async def interrupt(
    thread_id: str | None = None,
    x_hermes_voice_password: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_password(x_hermes_voice_password)
    thread = store.get_thread(thread_id) if thread_id else store.ensure_default_thread()
    runtime.trigger_cancel(thread["id"])
    return await asyncio.to_thread(runtime.interrupt, thread["id"])


@app.post("/api/turn")
async def voice_turn(
    audio: UploadFile = File(...),
    thread_id: str | None = Form(default=None),
    x_hermes_voice_password: str | None = Header(default=None),
) -> JSONResponse:
    _require_password(x_hermes_voice_password)
    thread = store.get_thread(thread_id) if thread_id else store.ensure_default_thread()
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Audio upload was empty")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio upload is too large")

    async with runtime.thread_lock(thread["id"]):
        started = time.time()
        transcript = ""
        stage = "received"
        metrics: dict[str, Any] = {
            "received_at": started,
            "audio_bytes": len(raw),
            "fallback_voice": True,
        }
        suffix = _suffix_for_upload(audio)
        try:
            with tempfile.TemporaryDirectory(prefix="hermes-voice-room-") as tmp:
                tmpdir = Path(tmp)
                audio_path = tmpdir / f"input{suffix}"
                audio_path.write_bytes(raw)

                stage = "transcribe"
                transcribe_started = time.time()
                transcript, stt_meta = await asyncio.to_thread(_transcribe_audio, audio_path)
                metrics.update(stt_meta)
                metrics["transcribe_seconds"] = round(time.time() - transcribe_started, 3)
                if not transcript:
                    return JSONResponse(
                        {
                            "ok": False,
                            "reason": "no_speech",
                            "message": "I did not catch speech clearly.",
                            "thread": store.get_thread(thread["id"]),
                        }
                    )

                transcript = _normalize_voice_transcript(transcript, metrics)

                stage = "hermes"
                hermes_started = time.time()
                metrics.update(_turn_config_metrics())
                metrics["thread_mode"] = thread.get("mode", "fast")
                reply, meta = await asyncio.to_thread(_respond_to_text, thread["id"], transcript, "voice")
                for key in ("engine", "model_provider", "model", "reasoning_effort"):
                    if meta and meta.get(key) is not None:
                        metrics[key] = meta[key]
                metrics["hermes_seconds"] = round(time.time() - hermes_started, 3)

                stage = "tts"
                tts_started = time.time()
                audio_reply_path, audio_mime, tts_engine = await asyncio.to_thread(
                    _synthesize_speech, thread["id"], reply, tmpdir
                )
                metrics["tts_seconds"] = round(time.time() - tts_started, 3)
                metrics["tts_engine"] = tts_engine
                audio_b64 = base64.b64encode(audio_reply_path.read_bytes()).decode("ascii")
        except InterruptedTurn:
            record = store.add_turn(
                thread_id=thread["id"],
                mode="voice",
                user_text="[interrupted]",
                reply="Interrupted.",
                started=started,
                metrics=metrics,
                interrupted=True,
            )
            return JSONResponse({"ok": False, "reason": "interrupted", **record})
        except HTTPException as exc:
            return _voice_failure_response(thread["id"], started, metrics, stage, transcript, str(exc.detail), exc.status_code)
        except Exception as exc:
            logger.exception("Voice turn failed during %s", stage)
            return _voice_failure_response(thread["id"], started, metrics, stage, transcript, str(exc), 500)

        record = store.add_turn(
            thread_id=thread["id"],
            mode="voice",
            user_text=transcript,
            reply=reply,
            started=started,
            metrics=metrics,
        )
        current_thread = store.get_thread(thread["id"])

        return JSONResponse(
            {
                "ok": True,
                **record,
                "thread": current_thread,
                "new_thread": meta.get("new_thread") if meta else None,
                "audio_mime": audio_mime,
                "audio_base64": audio_b64,
            }
        )


def _voice_failure_response(
    thread_id: str,
    started: float,
    metrics: dict[str, Any],
    stage: str,
    transcript: str,
    detail: str,
    status_code: int,
) -> JSONResponse:
    clean_detail = _clean_text(detail or "Voice turn failed")[-900:]
    metrics["error_stage"] = stage
    metrics["error"] = clean_detail
    record = store.add_turn(
        thread_id=thread_id,
        mode="voice",
        user_text=transcript or "[voice turn failed]",
        reply=f"Voice turn failed during {stage}: {clean_detail}",
        started=started,
        metrics=metrics,
    )
    logger.warning("Voice turn failed during %s: %s", stage, clean_detail)
    return JSONResponse(
        status_code=200,
        content={
            "ok": False,
            "reason": "voice_turn_failed",
            "stage": stage,
            "message": clean_detail,
            "backend_status_code": status_code,
            "thread": store.get_thread(thread_id),
            **record,
        },
    )


@app.post("/api/threads/{thread_id}/hermes-turn")
async def thread_text_turn(
    thread_id: str,
    payload: TextTurnRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> JSONResponse:
    _require_password(x_hermes_voice_password)
    return await _text_turn_for_thread(thread_id, payload)


@app.post("/api/text")
async def text_turn(
    payload: TextTurnRequest,
    x_hermes_voice_password: str | None = Header(default=None),
) -> JSONResponse:
    _require_password(x_hermes_voice_password)
    thread = store.ensure_default_thread()
    return await _text_turn_for_thread(thread["id"], payload)


async def _text_turn_for_thread(thread_id: str, payload: TextTurnRequest) -> JSONResponse:
    thread = store.get_thread(thread_id)
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    async with runtime.thread_lock(thread["id"]):
        started = time.time()
        metrics: dict[str, Any] = {
            "received_at": started,
            "thread_mode": thread.get("mode", "fast"),
            **_turn_config_metrics(),
        }
        try:
            hermes_started = time.time()
            reply, meta = await asyncio.to_thread(_respond_to_text, thread["id"], text, "text")
            for key in ("engine", "model_provider", "model", "reasoning_effort"):
                if meta and meta.get(key) is not None:
                    metrics[key] = meta[key]
            metrics["hermes_seconds"] = round(time.time() - hermes_started, 3)
            record = store.add_turn(
                thread_id=thread["id"],
                mode="text",
                user_text=text,
                reply=reply,
                started=started,
                metrics=metrics,
            )
            response: dict[str, Any] = {
                "ok": True,
                **record,
                "thread": store.get_thread(thread["id"]),
                "new_thread": meta.get("new_thread") if meta else None,
            }

            if payload.speak:
                with tempfile.TemporaryDirectory(prefix="hermes-voice-room-text-") as tmp:
                    tts_started = time.time()
                    audio_reply_path, audio_mime, tts_engine = await asyncio.to_thread(
                        _synthesize_speech, thread["id"], reply, Path(tmp)
                    )
                    response["metrics"]["tts_seconds"] = round(time.time() - tts_started, 3)
                    response["metrics"]["tts_engine"] = tts_engine
                    response["audio_mime"] = audio_mime
                    response["audio_base64"] = base64.b64encode(audio_reply_path.read_bytes()).decode("ascii")

            return JSONResponse(response)
        except InterruptedTurn:
            record = store.add_turn(
                thread_id=thread["id"],
                mode="text",
                user_text=text,
                reply="Interrupted.",
                started=started,
                metrics=metrics,
                interrupted=True,
            )
            return JSONResponse({"ok": False, "reason": "interrupted", **record})


# =========================================================================
# Streaming voice pipeline (WebSocket)
#
# Fast-mode turns skip the Hermes CLI subprocess entirely. The default path
# streams a local Ollama model; xAI remains an explicit opt-in fallback. Both
# paths cut tokens into clauses, synthesize each clause with Kokoro while the
# rest is still generating, and push audio chunks to the phone as they exist.
# Agent-mode turns keep the full Hermes CLI path (tools, memory, sessions).
# =========================================================================

FAST_VOICE_SYSTEM = (
    "You are Hermes, the user's personal AI assistant, in a live voice conversation "
    "on their phone. Your replies are spoken aloud by a TTS engine: sound natural, "
    "warm and direct, and keep it brief — one to three sentences unless the user "
    "asks for more. Never use markdown, bullet points, headings, code blocks, or "
    "emojis. Plain speakable sentences only. Numbers and abbreviations should be "
    "written the way they are spoken. Always reply in the user's language: if "
    "they speak Arabic, answer in Arabic.\n\n"
    "You have real tools and can actually DO things on the user's Mac, not just talk. "
    "When the user asks you to do something concrete — read or change a file, run a "
    "command, check something, look something up — USE THE TOOL, then tell them what "
    "you did or found in one or two spoken sentences. Rules for using tools well:\n"
    "- Prefer acting over asking. If the request is clear, do it. Only ask a "
    "clarifying question when you genuinely cannot proceed.\n"
    "- To change part of an existing file, read_file first if unsure, then edit_file "
    "with an exact, unique snippet. Use write_file only for new files or full rewrites.\n"
    "- Use run_command for real actions (git, tests, ls, builds). Never guess a "
    "command's output — run it and report what actually came back.\n"
    "- Use web_search only when the answer needs current or external facts; answer "
    "from your own knowledge otherwise.\n"
    "- Only start a background Composer build when the user explicitly asks to build "
    "something, work in the background, or use Composer. For quick edits, just do them "
    "directly with the file and command tools.\n"
    "- If you call a tool, say a short filler first like 'One sec' or 'Let me check' "
    "so the user hears you working. After the tool returns, confirm the real result — "
    "never claim success you didn't verify, and never invent file contents, command "
    "output, or build status.\n"
    "- Keep spoken confirmations tight: what you did and the outcome, not a play-by-play."
)

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?…]['\")\]]?(?:\s+|$)")
_CLAUSE_BOUNDARY_RE = re.compile(r"[,;:—–](?:\s+|$)")
_TTS_STRIP_RE = re.compile(r"[*_`#>|]+")
MAX_SENTENCE_BUFFER = 260
MIN_STREAM_CLAUSE_CHARS = 10
STREAM_HISTORY_TURNS = 12

LOCAL_VOICE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current local date, time, and timezone on this Mac.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_voice_engine_status",
            "description": "Get the active local STT, language model, TTS, and transport status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_voice_latency",
            "description": "Get measured latency for recent successful voice turns in this conversation.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _local_voice_tools_for(user_text: str) -> list[dict[str, Any]]:
    low = user_text.lower()
    asks_time = bool(re.search(r"\b(?:what(?:'s| is) the time|current time|time is it|date is it|today's date)\b", low))
    mentions_voice_engine = any(
        term in low for term in ("voice engine", "voice pipeline", "local voice", "speech engine")
    )
    asks_voice_status = mentions_voice_engine and (
        "status" in low
        or bool(re.search(r"\b(?:what|which)\b.*\b(?:model|provider|using|running|active)\b", low))
        or bool(re.search(
            r"\b(?:is|are)\s+(?:the\s+)?(?:local\s+)?(?:voice engine|voice pipeline|speech engine)\s+"
            r"(?:running|active|working|local)\b",
            low,
        ))
        or bool(re.search(r"\b(?:check|show|tell)\b.*\b(?:engine|pipeline|model|provider)\b", low))
    )
    asks_latency = (
        any(term in low for term in ("voice latency", "response latency", "how fast", "response time"))
        and bool(re.search(r"\b(?:what|how|check|measure|show|tell|average|latest)\b", low))
    )
    if asks_time:
        return [LOCAL_VOICE_TOOLS[0]]
    if asks_voice_status:
        return [LOCAL_VOICE_TOOLS[1]]
    if asks_latency:
        return [LOCAL_VOICE_TOOLS[2]]
    return []


def _run_local_voice_tool(name: str, thread: dict[str, Any]) -> str:
    if name == "get_current_time":
        return json.dumps({
            "local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": time.tzname[0] if time.tzname else "local",
        })
    if name == "get_voice_engine_status":
        voice = _current_voice()
        return json.dumps({
            "stt": "whisper.cpp large-v3-turbo q8, local",
            "model": OLLAMA_VOICE_MODEL,
            "model_provider": "Ollama, local",
            "tts": f"{voice['engine']}:{voice['voice']}, local",
            "transport": "WebSocket",
            "thread_mode": thread.get("mode", "fast"),
        })
    if name == "get_recent_voice_latency":
        turns = [
            turn for turn in store.recent_turns(thread["id"], limit=12)
            if not turn.get("interrupted") and (turn.get("metrics") or {}).get("first_audio_seconds") is not None
        ]
        samples = [float(turn["metrics"]["first_audio_seconds"]) for turn in turns]
        return json.dumps({
            "sample_count": len(samples),
            "average_first_audio_seconds": round(sum(samples) / len(samples), 3) if samples else None,
            "latest_first_audio_seconds": round(samples[-1], 3) if samples else None,
        })
    return json.dumps({"error": f"Unknown local voice tool: {name}"})


# ───────────────────────────────────────────────────────────────────────────
# Fast-voice REAL tools: files, terminal, web. These run locally in DEFAULT_CWD
# so the direct-xAI fast path can actually DO things (not just chat) while
# staying at ~1.3s first token. Descriptions are written for precision — a
# low-effort model should still pick the right tool and arguments.
# ───────────────────────────────────────────────────────────────────────────
FAST_TOOL_TIMEOUT = float(os.environ.get("HERMES_VOICE_TOOL_TIMEOUT", "45"))
FAST_TOOL_READ_LIMIT = 15000
FAST_TOOL_OUTPUT_LIMIT = 6000

FAST_AGENT_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file and return its contents. Use whenever the user asks what is in a file, to review code, or before editing. Path is relative to the working folder unless absolute.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path, e.g. 'server.py' or 'src/app.js'"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create a new file or completely overwrite an existing one with the given content. Use for brand-new files or full rewrites. For a small change to an existing file, prefer edit_file so you don't lose the rest.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "The full file content"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace one exact snippet of text in an existing file with new text, leaving the rest untouched. Use for targeted edits. The 'find' text must match exactly and appear only once — read_file first if unsure.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path to edit"},
            "find": {"type": "string", "description": "Exact text to replace (must be unique in the file)"},
            "replace": {"type": "string", "description": "New text to put in its place"}},
            "required": ["path", "find", "replace"]}}},
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files and folders in a directory (defaults to the working folder). Use to see what exists before reading or to answer 'what files are here'.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path, empty for the working folder"}}}}},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "Search the working folder for files whose contents contain a phrase or symbol. Use to find where something is defined or mentioned. Returns matching files and lines.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Text or symbol to search for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command in the working folder and return its output. Use for real actions: git status, running tests, ls, npm run, moving files, etc. Prefer non-interactive commands. Avoid destructive commands (rm -rf, force push) unless the user clearly asked.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The exact shell command to run"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live web for current information (news, prices, docs, facts you are unsure about). Returns a concise sourced answer. Use only when the answer needs up-to-date or external info, not for things you already know.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch a specific web page and return its readable text. Use when the user gives a URL or you found one via web_search and need its content.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "The full URL to fetch"}},
            "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current local date, time, and timezone on this Mac.",
        "parameters": {"type": "object", "properties": {}}}},
]

COMPOSER_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "tell_composer",
        "description": "Hand a larger build or multi-file coding job to Composer, which works in the background while you keep talking. Use ONLY when the user explicitly asks you to build something, work in the background, or use Composer. For quick edits use edit_file/write_file/run_command instead.",
        "parameters": {"type": "object", "properties": {
            "instruction": {"type": "string", "description": "The exact build instructions"}},
            "required": ["instruction"]}}},
    {"type": "function", "function": {
        "name": "check_composer",
        "description": "Check the status of the currently running background Composer build. Use when the user asks how the build is going.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _fast_resolve_path(path: str) -> Path:
    path = (path or "").strip()
    p = Path(path)
    if not p.is_absolute():
        p = DEFAULT_CWD / p
    return p.resolve()


def _exec_read_file(args: dict[str, Any]) -> str:
    p = _fast_resolve_path(str(args.get("path") or ""))
    if not p.exists():
        return f"No file at {args.get('path')}."
    if not p.is_file():
        return f"{args.get('path')} is a folder, not a file. Use list_directory."
    try:
        data = p.read_text(errors="replace")
    except Exception as exc:
        return f"Could not read {args.get('path')}: {exc}"
    if len(data) > FAST_TOOL_READ_LIMIT:
        return data[:FAST_TOOL_READ_LIMIT] + f"\n… (truncated, file is {len(data)} chars)"
    return data or "(the file is empty)"


def _exec_write_file(args: dict[str, Any]) -> str:
    p = _fast_resolve_path(str(args.get("path") or ""))
    content = str(args.get("content") or "")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    except Exception as exc:
        return f"Could not write {args.get('path')}: {exc}"
    return f"Wrote {len(content)} characters to {args.get('path')}."


def _exec_edit_file(args: dict[str, Any]) -> str:
    p = _fast_resolve_path(str(args.get("path") or ""))
    find = str(args.get("find") or "")
    replace = str(args.get("replace") or "")
    if not p.exists() or not p.is_file():
        return f"No file at {args.get('path')} to edit."
    if not find:
        return "The 'find' text was empty — nothing to replace."
    try:
        text = p.read_text(errors="replace")
    except Exception as exc:
        return f"Could not read {args.get('path')}: {exc}"
    count = text.count(find)
    if count == 0:
        return f"That exact text is not in {args.get('path')}. Read the file first to get the exact snippet."
    if count > 1:
        return f"That text appears {count} times in {args.get('path')}; include more surrounding text so it's unique."
    try:
        p.write_text(text.replace(find, replace, 1))
    except Exception as exc:
        return f"Could not write {args.get('path')}: {exc}"
    return f"Edited {args.get('path')}."


def _exec_list_directory(args: dict[str, Any]) -> str:
    raw = str(args.get("path") or "").strip()
    p = _fast_resolve_path(raw) if raw else DEFAULT_CWD
    if not p.exists() or not p.is_dir():
        return f"No folder at {raw or '.'}."
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    except Exception as exc:
        return f"Could not list {raw or '.'}: {exc}"
    lines = []
    for e in entries[:200]:
        if e.name.startswith("."):
            continue
        lines.append(f"{e.name}/" if e.is_dir() else e.name)
    return "\n".join(lines) or "(empty folder)"


def _exec_search_files(args: dict[str, Any]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "No search query given."
    cmd = ["grep", "-rniI", "--max-count=3", "--exclude-dir=.git", "--exclude-dir=node_modules",
           "--exclude-dir=venv", "--exclude-dir=.venv", query, "."]
    try:
        proc = subprocess.run(cmd, cwd=str(DEFAULT_CWD), capture_output=True, text=True, timeout=20)
    except Exception as exc:
        return f"Search failed: {exc}"
    out = (proc.stdout or "").strip()
    if not out:
        return f"No files contain '{query}'."
    lines = out.splitlines()[:40]
    return "\n".join(lines)


def _exec_run_command(args: dict[str, Any]) -> str:
    command = str(args.get("command") or "").strip()
    if not command:
        return "No command given."
    try:
        proc = subprocess.run(
            command, shell=True, cwd=str(DEFAULT_CWD), capture_output=True, text=True,
            timeout=FAST_TOOL_TIMEOUT, env=_hermes_env(),
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {int(FAST_TOOL_TIMEOUT)}s: {command}"
    except Exception as exc:
        return f"Could not run command: {exc}"
    out = (proc.stdout or "")
    if proc.stderr:
        out += ("\n[stderr]\n" + proc.stderr)
    out = out.strip()
    if len(out) > FAST_TOOL_OUTPUT_LIMIT:
        out = out[:FAST_TOOL_OUTPUT_LIMIT] + "\n… (output truncated)"
    if not out:
        return f"Command finished with exit code {proc.returncode} and no output."
    return f"(exit {proc.returncode})\n{out}"


def _exec_web_search(args: dict[str, Any], creds: dict[str, str]) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "No search query given."
    body = {
        "model": "grok-4.5",
        "messages": [{"role": "user", "content": f"Search the web and answer concisely with key facts: {query}"}],
        "search_parameters": {"mode": "on"},
        "stream": False,
    }
    try:
        with httpx.Client(timeout=httpx.Timeout(40.0, connect=8.0)) as client:
            r = client.post(f"{creds['base_url']}/chat/completions", json=body,
                            headers={"Authorization": f"Bearer {creds['api_key']}"})
            if r.status_code != 200:
                return f"Web search failed (HTTP {r.status_code})."
            data = r.json()
            return (data.get("choices") or [{}])[0].get("message", {}).get("content") or "No result."
    except Exception as exc:
        return f"Web search failed: {exc}"


def _exec_fetch_url(args: dict[str, Any]) -> str:
    url = str(args.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return "That doesn't look like a valid URL."
    try:
        with httpx.Client(timeout=httpx.Timeout(25.0, connect=8.0), follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "Mozilla/5.0 HermesVoice"})
            html = r.text
    except Exception as exc:
        return f"Could not fetch that page: {exc}"
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:FAST_TOOL_READ_LIMIT] or "(no readable text on that page)"


def _fast_tools_for(user_text: str, thread: dict[str, Any]) -> list[dict[str, Any]]:
    """Direct action tools are always available on the fast path; Composer's
    background-build tools appear only when the user explicitly asks to build /
    use Composer (per the user's rule — never auto-spawn a background build)."""
    tools = list(FAST_AGENT_TOOLS)
    low = (user_text or "").lower()
    build_intent = False
    try:
        build_intent = _detect_build_intent(thread["id"], user_text) is not None
    except Exception:
        build_intent = False
    if build_intent or "composer" in low or re.search(r"\bin the background\b", low):
        tools += COMPOSER_TOOLS
    return tools


async def _exec_fast_tool(name: str, args: dict[str, Any], thread: dict[str, Any],
                          creds: dict[str, str]) -> str:
    """Dispatch a fast-path tool call to its executor (sync ones in a thread)."""
    try:
        if name == "read_file":
            return await asyncio.to_thread(_exec_read_file, args)
        if name == "write_file":
            return await asyncio.to_thread(_exec_write_file, args)
        if name == "edit_file":
            return await asyncio.to_thread(_exec_edit_file, args)
        if name == "list_directory":
            return await asyncio.to_thread(_exec_list_directory, args)
        if name == "search_files":
            return await asyncio.to_thread(_exec_search_files, args)
        if name == "run_command":
            return await asyncio.to_thread(_exec_run_command, args)
        if name == "web_search":
            return await asyncio.to_thread(_exec_web_search, args, creds)
        if name == "fetch_url":
            return await asyncio.to_thread(_exec_fetch_url, args)
        if name in ("get_current_time", "get_voice_engine_status", "get_recent_voice_latency"):
            return await asyncio.to_thread(_run_local_voice_tool, name, thread)
        if name == "tell_composer":
            result = await asyncio.to_thread(_start_background_build, thread["id"], str(args.get("instruction") or ""))
            with _builds_guard:
                _b = _builds.get(thread["id"])
            if _b and _b["proc"].poll() is None:
                asyncio.create_task(_monitor_build(thread["id"]))
            return result
        if name == "check_composer":
            return await asyncio.to_thread(_background_build_status, thread["id"])
    except Exception as exc:
        return f"Tool {name} failed: {_clean_text(str(exc))[-200:]}"
    return f"Unknown tool: {name}"


# Short, spoken label for the activity chip while a tool runs.
_FAST_TOOL_LABELS = {
    "read_file": "Reading a file", "write_file": "Writing a file",
    "edit_file": "Editing a file", "list_directory": "Listing files",
    "search_files": "Searching files", "run_command": "Running a command",
    "web_search": "Searching the web", "fetch_url": "Reading a page",
    "get_current_time": "Checking the time", "tell_composer": "Starting a build",
    "check_composer": "Checking the build",
}


async def _emit_execution_tool(
    ws: "WebSocket",
    *,
    tool_name: str,
    tool_call_id: str,
    label: str,
    phase: str,
    preview: str = "",
    request_id: str | None = None,
) -> None:
    """Emit normalized execution events for non-agent streaming paths."""
    try:
        from engines.agent_activity import make_event, safe_preview

        rid = request_id or f"req_fast_{tool_call_id}"
        safe = safe_preview(preview)
        if phase == "started":
            events = [
                make_event("status_changed", rid, status="using_tool", label=label),
                make_event(
                    "tool_started",
                    rid,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    label=label,
                    preview=safe,
                    title=label,
                ),
                make_event(
                    "task_started",
                    rid,
                    task_id=tool_call_id,
                    title=label,
                    detail=safe or None,
                ),
            ]
        elif phase == "failed":
            events = [
                make_event(
                    "tool_failed",
                    rid,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    label=label,
                    ok=False,
                    preview=safe,
                    error=safe,
                ),
                make_event(
                    "task_failed",
                    rid,
                    task_id=tool_call_id,
                    title=label,
                    detail=safe or None,
                    ok=False,
                ),
            ]
        else:
            events = [
                make_event(
                    "tool_completed",
                    rid,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    label=label,
                    ok=True,
                    preview=safe,
                ),
                make_event(
                    "task_completed",
                    rid,
                    task_id=tool_call_id,
                    title=label,
                    detail=safe or None,
                    ok=True,
                ),
            ]
        for event in events:
            await _ws_send(ws, {"type": "execution", "event": event})
    except Exception:
        logger.debug("execution event emit failed", exc_info=True)


def _pop_sentences(buffer: str, *, flush: bool = False) -> tuple[list[str], str]:
    """Split completed sentences off the front of the token buffer."""
    sentences: list[str] = []
    rest = buffer
    while True:
        match = _SENTENCE_BOUNDARY_RE.search(rest)
        if match:
            cut = match.end()
            sentence = rest[:cut].strip()
            rest = rest[cut:]
            if sentence:
                sentences.append(sentence)
            continue
        # A short opening clause gets speech underway substantially sooner
        # than waiting for the model's final period. Keep a minimum length so
        # tiny discourse markers do not become choppy one-word WAV files.
        clause = _CLAUSE_BOUNDARY_RE.search(rest)
        if clause and clause.end() >= MIN_STREAM_CLAUSE_CHARS:
            cut = clause.end()
            sentence = rest[:cut].strip()
            rest = rest[cut:]
            if sentence:
                sentences.append(sentence)
            continue
        if len(rest) > MAX_SENTENCE_BUFFER:
            # No boundary in an over-long run: cut at the last comma/space so
            # TTS can start instead of waiting for a distant period.
            cut_at = max(rest.rfind(", ", 0, MAX_SENTENCE_BUFFER), rest.rfind(" ", 0, MAX_SENTENCE_BUFFER))
            if cut_at > 40:
                sentence = rest[: cut_at + 1].strip()
                rest = rest[cut_at + 1 :]
                if sentence:
                    sentences.append(sentence)
                continue
        break
    if flush:
        tail = rest.strip()
        if tail:
            sentences.append(tail)
        rest = ""
    return sentences, rest


def _tts_normalize(text: str) -> str:
    return _TTS_STRIP_RE.sub("", text).strip()


def _resolve_xai_stream_credentials(provider: str) -> dict[str, str]:
    import sys

    if str(HERMES_AGENT) not in sys.path:
        sys.path.insert(0, str(HERMES_AGENT))
    if provider == "xai-oauth":
        from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

        creds = resolve_xai_oauth_runtime_credentials()
        return {
            "base_url": str(creds.get("base_url") or "https://api.x.ai/v1").rstrip("/"),
            "api_key": str(creds.get("api_key") or ""),
        }
    if provider == "xai":
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY") or ""
        if not api_key:
            raise RuntimeError("XAI_API_KEY is not set; use the xAI OAuth provider instead.")
        return {"base_url": "https://api.x.ai/v1", "api_key": api_key}
    raise RuntimeError(f"Streaming is only wired for xAI providers, got: {provider}")


def _grok_reasoning_body(model: str, effort: str) -> dict[str, Any] | None:
    if not effort or effort in {"", "none"}:
        return None
    import sys

    if str(HERMES_AGENT) not in sys.path:
        sys.path.insert(0, str(HERMES_AGENT))
    try:
        from agent.model_metadata import grok_supports_reasoning_effort

        if not grok_supports_reasoning_effort(model):
            return None
    except Exception:
        return None
    return {"enabled": True, "effort": effort}


def _build_fast_messages(thread: dict[str, Any], user_text: str) -> list[dict[str, str]]:
    system = FAST_VOICE_SYSTEM
    if thread.get("goal"):
        system += f"\nCurrent conversation goal: {thread['goal']}"
    build_note = _active_build_note(thread["id"])
    if build_note:
        system += (
            "\nYou are the voice narrator while a separate builder model (composer) "
            "works in the background. " + build_note
        )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in store.recent_turns(thread["id"], limit=STREAM_HISTORY_TURNS):
        transcript = (turn.get("transcript") or "").strip()
        reply = (turn.get("reply") or "").strip()
        if turn.get("interrupted") or transcript.startswith("["):
            continue
        if transcript:
            messages.append({"role": "user", "content": transcript})
        if reply:
            messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": user_text})
    return messages


_ARABIC_RE = re.compile(r"[؀-ۿ]")


def _macos_say_synth_bytes(text: str, voice: str = "Majed") -> bytes:
    """macOS `say` synthesis for any installed voice (Arabic via Majed, plus
    accents Kokoro lacks). `voice` is regex-validated before reaching here.
    Fully local, same wav container the client already decodes.
    """
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "t.aiff"
        wav = Path(td) / "t.wav"
        subprocess.run(["/usr/bin/say", "-v", voice, "-o", str(aiff), text],
                       check=True, timeout=45)
        subprocess.run(["/opt/homebrew/bin/ffmpeg", "-hide_banner", "-loglevel", "quiet",
                        "-i", str(aiff), "-ar", "24000", "-ac", "1", str(wav)],
                       check=True, timeout=45)
        return wav.read_bytes()


def _kokoro_service_synth(text: str, voice: str, lang: str) -> bytes:
    request = urllib.request.Request(
        f"{KOKORO_SERVICE_URL}/synthesize",
        data=json.dumps({"text": text, "voice": voice, "lang": lang}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 200:
            raise RuntimeError(f"Kokoro service returned HTTP {response.status}")
        return response.read()


def _synth_bytes(text: str, override: dict[str, str] | None = None) -> bytes:
    """Synthesize with the user's selected voice (or an explicit override for
    previews). Arabic text always routes to macOS Majed — Kokoro has no Arabic.
    """
    v = override or _current_voice()
    if _ARABIC_RE.search(text) and v.get("engine") != "say":
        return _macos_say_synth_bytes(text, "Majed")
    if v.get("engine") == "say":
        return _macos_say_synth_bytes(text, v.get("voice", "Samantha"))
    return _kokoro_service_synth(text, v.get("voice", KOKORO_VOICE), v.get("lang", KOKORO_LANG))


def _kokoro_synth_bytes(text: str) -> bytes:
    """Blocking single-sentence synthesis via the live selected voice."""
    return _synth_bytes(text)


async def _ws_send(ws: WebSocket, payload: dict[str, Any]) -> None:
    try:
        turn_id = _WS_TURN_ID.get("")
        outgoing = payload
        if turn_id and payload.get("type") in _TURN_EVENT_TYPES and "turn_id" not in payload:
            outgoing = {**payload, "turn_id": turn_id}
        await ws.send_text(json.dumps(outgoing, ensure_ascii=False))
    except Exception:
        pass


async def _run_ws_turn_context(turn_id: str, coroutine: Any) -> None:
    token = _WS_TURN_ID.set(turn_id)
    try:
        await coroutine
    finally:
        _WS_TURN_ID.reset(token)


async def _stream_ollama_reply(
    ws: WebSocket,
    thread: dict[str, Any],
    user_text: str,
    cancel: asyncio.Event,
    metrics: dict[str, Any],
    *,
    speak: bool = True,
) -> tuple[str, bool]:
    """Stream a fully local Ollama reply into sentence-level Kokoro audio."""
    started = time.time()
    turn_started = float(metrics.get("received_at") or started)
    reply_parts: list[str] = []
    buffer = ""
    seq = 0
    tts_total = 0.0
    first_audio_sent = False
    interrupted = False
    kokoro_ok = speak and _kokoro_available() and _kokoro_service_ready()
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def tts_consumer() -> None:
        nonlocal seq, tts_total, first_audio_sent
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                return
            if cancel.is_set():
                continue
            speakable = _tts_normalize(sentence)
            if not speakable or not kokoro_ok:
                continue
            try:
                tts_started = time.time()
                wav = await asyncio.to_thread(_kokoro_synth_bytes, speakable)
                tts_total += time.time() - tts_started
            except Exception as exc:
                logger.warning("Local sentence TTS failed: %s", exc)
                continue
            if cancel.is_set():
                continue
            if not first_audio_sent:
                first_audio_sent = True
                metrics["model_first_audio_seconds"] = round(time.time() - started, 3)
                metrics["first_audio_seconds"] = round(time.time() - turn_started, 3)
            seq += 1
            await _ws_send(ws, {
                "type": "audio",
                "seq": seq,
                "mime": "audio/wav",
                "b64": base64.b64encode(wav).decode("ascii"),
                "text": sentence,
            })

    body = {
        "model": OLLAMA_VOICE_MODEL,
        "messages": _build_fast_messages(thread, user_text),
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0.35,
            "num_ctx": 4096,
        },
    }
    available_tools = _local_voice_tools_for(user_text)
    if available_tools:
        body["tools"] = available_tools
    consumer_task = asyncio.create_task(tts_consumer())
    try:
        timeout = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)
        prompt_tokens = 0
        completion_tokens = 0
        eval_duration_ns = 0
        async with httpx.AsyncClient(timeout=timeout) as client:
            for _tool_round in range(3):
                tool_calls: list[dict[str, Any]] = []
                round_content = ""
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=body) as response:
                    if response.status_code != 200:
                        detail = (await response.aread()).decode("utf-8", "replace")[:600]
                        raise RuntimeError(f"Ollama returned HTTP {response.status_code}: {detail}")
                    async for line in response.aiter_lines():
                        if cancel.is_set():
                            interrupted = True
                            break
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("error"):
                            raise RuntimeError(str(chunk["error"]))
                        message = chunk.get("message") or {}
                        token = str(message.get("content") or "")
                        if token:
                            if "first_token_seconds" not in metrics:
                                metrics["model_first_token_seconds"] = round(time.time() - started, 3)
                                metrics["first_token_seconds"] = round(time.time() - turn_started, 3)
                            round_content += token
                            reply_parts.append(token)
                            buffer += token
                            await _ws_send(ws, {"type": "token", "text": token})
                            sentences, buffer = _pop_sentences(buffer)
                            for sentence in sentences:
                                sentence_queue.put_nowait(sentence)
                        if message.get("tool_calls"):
                            tool_calls.extend(message["tool_calls"])
                        if chunk.get("done"):
                            prompt_tokens += int(chunk.get("prompt_eval_count") or 0)
                            completion_tokens += int(chunk.get("eval_count") or 0)
                            eval_duration_ns += int(chunk.get("eval_duration") or 0)
                if interrupted or not tool_calls:
                    break

                metrics["local_tool_calls"] = int(metrics.get("local_tool_calls") or 0) + len(tool_calls)
                body["messages"].append({
                    "role": "assistant",
                    "content": round_content,
                    "tool_calls": tool_calls,
                })
                for call in tool_calls:
                    function = call.get("function") or {}
                    name = str(function.get("name") or "")
                    await _ws_send(ws, {"type": "activity", "engine": "ollama", "label": f"Using {name}"})
                    result = await asyncio.to_thread(_run_local_voice_tool, name, thread)
                    body["messages"].append({
                        "role": "tool",
                        "tool_name": name,
                        "content": result,
                    })

            metrics["prompt_tokens"] = prompt_tokens
            metrics["completion_tokens"] = completion_tokens
            metrics["total_tokens"] = prompt_tokens + completion_tokens
            if eval_duration_ns > 0:
                metrics["tokens_per_second"] = round(
                    completion_tokens / (eval_duration_ns / 1_000_000_000), 2
                )
        if not interrupted:
            for sentence in _pop_sentences(buffer, flush=True)[0]:
                sentence_queue.put_nowait(sentence)
    finally:
        sentence_queue.put_nowait(None)
        try:
            await consumer_task
        except Exception:
            pass

    metrics["model"] = OLLAMA_VOICE_MODEL
    metrics["model_provider"] = "ollama"
    metrics["reasoning_effort"] = "none"
    metrics["reasoning_effort_supported"] = False
    metrics["stream_seconds"] = round(time.time() - started, 3)
    metrics["tts_seconds"] = round(tts_total, 3)
    metrics["tts_engine"] = f"kokoro:{_current_voice()['voice']}" if kokoro_ok else "none"
    metrics["transport"] = "ollama-local"
    return "".join(reply_parts).strip(), interrupted or cancel.is_set()


async def _stream_fast_reply(
    ws: WebSocket,
    thread: dict[str, Any],
    user_text: str,
    cancel: asyncio.Event,
    metrics: dict[str, Any],
    *,
    speak: bool = True,
) -> tuple[str, bool]:
    """Stream a fast-mode reply: xAI tokens -> sentences -> Kokoro -> WS audio.

    Returns (full_reply, interrupted).

    Backend is chosen from voice-runtime (UI) first, then env default.
    Selecting Grok/xAI in the panel must NOT silently stay on Ollama.
    """
    fast_provider = _effective_fast_provider()
    metrics["fast_provider"] = fast_provider
    runtime = _load_voice_runtime()
    allow_fallback = bool(runtime.get("allow_ollama_fallback"))

    if fast_provider == "ollama":
        await _ws_send(ws, {
            "type": "activity",
            "engine": "ollama",
            "label": f"Local model · {OLLAMA_VOICE_MODEL}",
        })
        return await _stream_ollama_reply(ws, thread, user_text, cancel, metrics, speak=speak)
    if fast_provider != "xai":
        raise RuntimeError(f"Unsupported fast voice provider: {fast_provider}")

    provider = ""
    model = ""
    effort = ""
    try:
        _yaml, data = _load_hermes_config_document()
        model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
        agent_cfg = data.get("agent") if isinstance(data.get("agent"), dict) else {}
        provider = str(model_cfg.get("provider") or "").strip()
        model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
        effort = str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    except Exception as exc:
        raise RuntimeError(f"Could not read Hermes config: {exc}") from exc
    if provider not in XAI_PROVIDERS or not model:
        raise RuntimeError(f"Fast streaming needs an xAI model in Hermes config (got {provider}/{model}).")

    await _ws_send(ws, {
        "type": "activity",
        "engine": "hermes",
        "label": f"Using {model}",
    })

    # If xAI auth is unavailable (revoked/expired OAuth):
    # - default: surface the error so UI "Grok" selection is honest
    # - only fall back to Ollama when the user explicitly enabled it
    try:
        creds = await asyncio.to_thread(_resolve_xai_stream_credentials, provider)
    except Exception as exc:
        err = _clean_text(str(exc))[-200:]
        logger.warning("xAI auth failed (%s)", err)
        metrics["fast_fallback"] = "xai_auth_failed"
        if allow_fallback:
            await _ws_send(ws, {
                "type": "activity",
                "engine": "ollama",
                "label": f"xAI unavailable — local {OLLAMA_VOICE_MODEL}",
            })
            return await _stream_ollama_reply(ws, thread, user_text, cancel, metrics, speak=speak)
        raise RuntimeError(
            f"Could not use {model} (xAI auth failed). "
            f"Re-auth xAI in Hermes, or switch Chat backend to Local Ollama. Detail: {err}"
        ) from exc
    body: dict[str, Any] = {
        "model": model,
        "messages": _build_fast_messages(thread, user_text),
        "stream": True,
    }
    
    # Real action tools (files/terminal/web/status), always available so the
    # fast path can actually get things done; Composer added only on build intent.
    body["tools"] = _fast_tools_for(user_text, thread)

    reasoning = _grok_reasoning_body(model, effort)
    if reasoning:
        body["reasoning"] = reasoning

    started = time.time()
    first_token_at: float | None = None
    first_audio_sent = False
    reply_parts: list[str] = []
    buffer = ""
    seq = 0
    tts_total = 0.0
    kokoro_ok = speak and _kokoro_available() and _kokoro_service_ready()

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def tts_consumer() -> None:
        nonlocal seq, tts_total, first_audio_sent
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                return
            if cancel.is_set():
                continue
            speakable = _tts_normalize(sentence)
            if not speakable or not kokoro_ok:
                continue
            try:
                tts_started = time.time()
                wav = await asyncio.to_thread(_kokoro_synth_bytes, speakable)
                tts_total += time.time() - tts_started
            except Exception as exc:
                logger.warning("Sentence TTS failed: %s", exc)
                continue
            if cancel.is_set():
                continue
            if not first_audio_sent:
                first_audio_sent = True
                metrics["first_audio_seconds"] = round(time.time() - started, 3)
            seq += 1
            await _ws_send(ws, {
                "type": "audio",
                "seq": seq,
                "mime": "audio/wav",
                "b64": base64.b64encode(wav).decode("ascii"),
                "text": sentence,
            })

    consumer_task = asyncio.create_task(tts_consumer())
    interrupted = False
    # Shared execution lifecycle for chat + voice activity UI.
    _fast_req: str | None = None
    try:
        from engines.agent_activity import STATUS_LABELS, make_event

        _fast_req = f"req_fast_{uuid.uuid4().hex[:10]}"
        await _ws_send(
            ws,
            {
                "type": "execution",
                "event": make_event(
                    "request_started",
                    _fast_req,
                    label=STATUS_LABELS["thinking"],
                    status="thinking",
                ),
            },
        )
    except Exception:
        _fast_req = None
    try:
        timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            while True:
                tool_calls_acc = {}
                async with client.stream(
                    "POST",
                    f"{creds['base_url']}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {creds['api_key']}"},
                ) as response:
                    if response.status_code != 200:
                        detail = (await response.aread()).decode("utf-8", "replace")[:600]
                        raise RuntimeError(f"xAI API returned HTTP {response.status_code}: {detail}")
                    async for line in response.aiter_lines():
                        if cancel.is_set():
                            interrupted = True
                            break
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(payload)
                            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                            token = delta.get("content") or ""
                            
                            tcs = delta.get("tool_calls")
                            if tcs:
                                for tc in tcs:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_acc:
                                        tool_calls_acc[idx] = {"id": tc.get("id", ""), "name": tc.get("function", {}).get("name", ""), "arguments": ""}
                                    if tc.get("function", {}).get("arguments"):
                                        tool_calls_acc[idx]["arguments"] += tc["function"]["arguments"]
                        except (json.JSONDecodeError, AttributeError, IndexError):
                            continue
                        
                        if token:
                            if first_token_at is None:
                                first_token_at = time.time()
                                metrics["first_token_seconds"] = round(first_token_at - started, 3)
                            reply_parts.append(token)
                            buffer += token
                            await _ws_send(ws, {"type": "token", "text": token})
                            sentences, buffer = _pop_sentences(buffer)
                            for sentence in sentences:
                                sentence_queue.put_nowait(sentence)
                
                if not interrupted and tool_calls_acc:
                    assistant_msg = {
                        "role": "assistant",
                        "content": "".join(reply_parts) or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]}
                            } for tc in tool_calls_acc.values()
                        ]
                    }
                    body["messages"].append(assistant_msg)
                    
                    for tc in tool_calls_acc.values():
                        name = tc["name"]
                        try:
                            args = json.loads(tc["arguments"])
                        except Exception:
                            args = {}
                        metrics["fast_tool_calls"] = int(metrics.get("fast_tool_calls") or 0) + 1
                        label = _FAST_TOOL_LABELS.get(name, f"Using {name}")
                        call_id = tc.get("id") or f"fast_{name}"
                        await _ws_send(ws, {"type": "activity", "engine": "hermes",
                                            "label": label, "tool": name, "tool_call_id": call_id})
                        await _emit_execution_tool(
                            ws,
                            tool_name=name,
                            tool_call_id=call_id,
                            label=label,
                            phase="started",
                            preview=str(args)[:120] if args else "",
                        )
                        result_text = await _exec_fast_tool(name, args, thread, creds)
                        await _emit_execution_tool(
                            ws,
                            tool_name=name,
                            tool_call_id=call_id,
                            label=label,
                            phase="completed",
                            preview=(result_text or "Done.")[:160],
                        )
                        body["messages"].append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result_text or "Done."
                        })

                    reply_parts.clear()
                    buffer = ""
                    continue
                
                break

        if not interrupted:
            for sentence in _pop_sentences(buffer, flush=True)[0]:
                sentence_queue.put_nowait(sentence)
    finally:
        sentence_queue.put_nowait(None)
        try:
            await consumer_task
        except Exception:
            pass

    metrics["stream_seconds"] = round(time.time() - started, 3)
    metrics["tts_seconds"] = round(tts_total, 3)
    metrics["tts_engine"] = f"kokoro:{KOKORO_VOICE}" if kokoro_ok else "none"
    metrics["transport"] = "stream"
    try:
        from engines.agent_activity import make_event

        if _fast_req:
            if interrupted or cancel.is_set():
                await _ws_send(
                    ws,
                    {
                        "type": "execution",
                        "event": make_event("response_cancelled", _fast_req),
                    },
                )
            else:
                await _ws_send(
                    ws,
                    {
                        "type": "execution",
                        "event": make_event("response_completed", _fast_req),
                    },
                )
    except Exception:
        pass
    return "".join(reply_parts).strip(), interrupted or cancel.is_set()


# ---------------------------------------------------------------------------
# CLI agent engines (grok / codex / claude): headless turns with tools.
# Each turn spawns the engine's CLI, resumes the thread's session, parses its
# JSONL stdout into text tokens + tool activity, and feeds the same
# sentence -> Kokoro -> WS audio pipeline the fast path uses.
# Access level: FULL (user-approved 2026-07-09) — engines run unsandboxed with
# auto-approved tools, matching the existing Hermes approvals.mode=off setup.
# ---------------------------------------------------------------------------

def _engine_env(engine_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TERM", "dumb")
    if engine_id == "claude":
        # never inherit another Claude session's scoped auth/proxy vars
        for key in list(env):
            if key.startswith(("ANTHROPIC_", "CLAUDE")):
                env.pop(key, None)
    return env


def _engine_argv(
    engine_id: str,
    prompt: str,
    session_id: str | None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    *,
    speak: bool = False,
) -> list[str]:
    # speak=True (hands-free voice) → short spoken style.
    # speak=False (phone text remote) → full agent capability.
    rules = ENGINE_VOICE_RULES if speak else ENGINE_REMOTE_RULES
    note = ENGINE_VOICE_NOTE if speak else ENGINE_REMOTE_NOTE
    if engine_id == "grok":
        argv = [
            GROK_BIN,
            "--output-format", "streaming-json",
            "--always-approve",
            "--cwd", str(DEFAULT_CWD),
            "--rules", rules,
        ]
        if model:
            argv += ["--model", model]
        if reasoning_effort:
            argv += ["--reasoning-effort", reasoning_effort]
        argv += ["-p", prompt]
        if session_id:
            argv += ["--resume", session_id]
        return argv
    if engine_id == "codex":
        voiced = f"{prompt}\n\n{note}"
        base = [CODEX_BIN]
        if reasoning_effort:
            base += ["-c", f'model_reasoning_effort="{reasoning_effort}"']
        if session_id:
            argv = base + [
                "exec", "resume", session_id, "--json",
                "--dangerously-bypass-approvals-and-sandbox", voiced,
            ]
            if model:
                argv.insert(-1, "--model")
                argv.insert(-1, model)
            return argv
        argv = base + [
            "exec", "--json", "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", str(DEFAULT_CWD),
        ]
        if model:
            argv += ["--model", model]
        argv.append(voiced)
        return argv
    if engine_id == "claude":
        argv = [
            CLAUDE_BIN, "-p", prompt,
            "--output-format", "stream-json", "--verbose",
            "--dangerously-skip-permissions",
            "--append-system-prompt", rules,
        ]
        if session_id:
            argv += ["--resume", session_id]
        return argv
    raise RuntimeError(f"No CLI runner for engine: {engine_id}")


_CODEX_ACTIVITY_LABELS = {
    "command_execution": "running a command",
    "file_change": "editing files",
    "mcp_tool_call": "using a tool",
    "web_search": "searching the web",
    "todo_list": "planning",
}


def _parse_engine_line(engine_id: str, line: str) -> list[tuple[str, str]]:
    """Normalize one JSONL stdout line into (kind, value) events.

    Kinds: token (speakable text), activity (tool-use label), session (id to
    resume next turn), error (fatal message reported by the engine itself).
    """
    line = line.strip()
    if not line or not line.startswith("{"):
        return []
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return []
    etype = str(event.get("type") or "")

    if engine_id == "grok":
        if etype == "text":
            return [("token", str(event.get("data") or ""))]
        if etype == "thought":
            return [("activity", "thinking")]
        if etype == "end":
            sid = str(event.get("sessionId") or "")
            return [("session", sid)] if sid else []
        return []

    if engine_id == "codex":
        if etype == "thread.started":
            sid = str(event.get("thread_id") or "")
            return [("session", sid)] if sid else []
        item = event.get("item") or {}
        itype = str(item.get("type") or "")
        if etype == "item.completed" and itype == "agent_message":
            text = str(item.get("text") or "")
            return [("token", text + "\n\n")] if text else []
        if etype == "item.started" and itype in _CODEX_ACTIVITY_LABELS:
            return [("activity", _CODEX_ACTIVITY_LABELS[itype])]
        if etype == "error":
            return [("error", str(event.get("message") or "Codex reported an error"))]
        return []

    if engine_id == "claude":
        if etype == "system" and event.get("subtype") == "init":
            sid = str(event.get("session_id") or "")
            return [("session", sid)] if sid else []
        if etype == "assistant":
            out: list[tuple[str, str]] = []
            message = event.get("message") or {}
            content = message.get("content") or []
            synthetic = str(message.get("model") or "") == "<synthetic>"
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text"):
                        if synthetic:
                            # CLI-generated failure notice (auth, crash) — never speak it raw
                            out.append(("error", str(block["text"])))
                        else:
                            out.append(("token", str(block["text"]) + "\n\n"))
                    elif block.get("type") == "tool_use":
                        out.append(("activity", f"using {block.get('name') or 'a tool'}"))
            return out
        if etype == "result":
            if event.get("is_error"):
                return [("error", str(event.get("result") or "Claude reported an error"))]
            sid = str(event.get("session_id") or "")
            return [("session", sid)] if sid else []
        return []

    return []


def _engine_error_hint(engine_id: str, detail: str) -> str:
    low = detail.lower()
    if engine_id == "claude" and ("401" in low or "authenticat" in low or "log in" in low or "login" in low):
        return "Claude Code needs a login on the Mac — run claude and /login once, then try again."
    if "login" in low or "auth" in low or "credential" in low or "401" in low:
        label = AGENT_ENGINES.get(engine_id, {}).get("label", engine_id)
        return f"{label} looks signed out on the Mac — log its CLI in once, then try again."
    return ""


def _iter_engine_events(engine_id: str, stdout_text: str) -> tuple[str, str | None, str | None]:
    """Parse a finished engine run's whole stdout. Returns (reply, session_id, error)."""
    parts: list[str] = []
    session_id: str | None = None
    error: str | None = None
    for line in stdout_text.splitlines():
        for kind, value in _parse_engine_line(engine_id, line):
            if kind == "token":
                parts.append(value)
            elif kind == "session":
                session_id = value
            elif kind == "error":
                error = value
    return "".join(parts).strip(), session_id, error


def _ask_antigravity_blocking(thread: dict[str, Any], user_text: str) -> str:
    """Dispatch to Antigravity and block until its reply lands (poll-based)."""
    if _antigravity is None or not _antigravity.is_available():
        raise HTTPException(status_code=502, detail=_antigravity_note())
    try:
        reply, cid = _antigravity.ask(
            user_text, thread.get("engine_session_id"), str(DEFAULT_CWD),
            title=thread.get("title"), timeout=ANTIGRAVITY_TURN_TIMEOUT,
        )
    except _antigravity.AntigravityUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Antigravity: {_clean_text(str(exc))[-300:]}") from exc
    if cid and cid != thread.get("engine_session_id"):
        store.set_engine_session(thread["id"], cid)
    return reply


def _ask_engine_blocking(thread: dict[str, Any], user_text: str) -> str:
    """Non-streaming engine turn for the HTTP fallback path (interrupt-safe)."""
    engine_id = thread.get("engine") or "hermes"
    if engine_id == "antigravity":
        return _ask_antigravity_blocking(thread, user_text)
    argv = _engine_argv(
        engine_id,
        user_text,
        thread.get("engine_session_id"),
        thread.get("model"),
        thread.get("reasoning_effort"),
        speak=False,
    )
    started = time.time()
    live = _grok_live_info(thread.get("engine_session_id")) if engine_id == "grok" else None
    if live:
        logger.warning(
            "grok session %s also live in terminal pid=%s — phone turn may race",
            thread.get("engine_session_id"), live.get("pid"),
        )
    proc = subprocess.Popen(
        argv,
        cwd=str(DEFAULT_CWD),
        env=_engine_env(engine_id),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    runtime.set_active(thread["id"], proc)
    try:
        stdout, stderr = proc.communicate(timeout=ENGINE_TURN_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(proc)
        if runtime.was_interrupted_since(thread["id"], started):
            raise InterruptedTurn() from exc
        raise HTTPException(status_code=504, detail=f"{engine_id} turn timed out") from exc
    finally:
        runtime.clear_active(thread["id"], proc)
    if runtime.was_interrupted_since(thread["id"], started):
        raise InterruptedTurn()

    reply, session_id, error = _iter_engine_events(engine_id, stdout or "")
    if error:
        if thread.get("engine_session_id"):
            store.set_engine_session(thread["id"], None)
        hint = _engine_error_hint(engine_id, f"{error} {reply}")
        if hint:
            return hint
        if not reply:
            raise HTTPException(status_code=502, detail=f"{engine_id} failed: {_clean_text(error)[-400:]}")
        return reply
    if session_id and session_id != thread.get("engine_session_id"):
        store.set_engine_session(thread["id"], session_id)
    if proc.returncode != 0 and not reply:
        detail = _clean_text(stderr or stdout or f"{engine_id} failed")[-400:]
        hint = _engine_error_hint(engine_id, detail)
        raise HTTPException(status_code=502, detail=hint or f"{engine_id} failed: {detail}")
    if not reply:
        raise HTTPException(status_code=502, detail=f"{engine_id} returned no text.")
    return reply


async def _stream_antigravity_reply(
    ws: WebSocket,
    thread: dict[str, Any],
    user_text: str,
    cancel: asyncio.Event,
    metrics: dict[str, Any],
    *,
    speak: bool = True,
) -> tuple[str, bool]:
    """Run an Antigravity turn: dispatch a background agent, poll for its reply,
    then speak the whole reply (Antigravity has no token stream)."""
    if _antigravity is None or not _antigravity.is_available():
        raise RuntimeError(_antigravity_note())
    started = time.time()
    metrics["engine"] = "antigravity"
    metrics["transport"] = "antigravity-poll"

    await _ws_send(ws, {"type": "activity", "engine": "antigravity",
                        "label": "dispatching to Antigravity agents"})

    def _run() -> tuple[str, str]:
        return _antigravity.ask(
            user_text, thread.get("engine_session_id"), str(DEFAULT_CWD),
            title=thread.get("title"), timeout=ANTIGRAVITY_TURN_TIMEOUT,
        )

    task = asyncio.create_task(asyncio.to_thread(_run))
    # Let the user interrupt while Antigravity works in the background.
    while not task.done():
        if cancel.is_set():
            # detach: the agent keeps running in the IDE, we just stop waiting
            task.add_done_callback(_consume_background_task_result)
            metrics["stream_seconds"] = round(time.time() - started, 3)
            return "", True
        await asyncio.sleep(0.2)

    try:
        reply, cid = task.result()
    except _antigravity.AntigravityUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    if cid and cid != thread.get("engine_session_id"):
        store.set_engine_session(thread["id"], cid)
        metrics["engine_session_id"] = cid

    metrics["stream_seconds"] = round(time.time() - started, 3)
    reply = (reply or "").strip()
    if not reply:
        raise RuntimeError("Antigravity ran but returned no reply text.")

    await _ws_send(ws, {"type": "token", "text": reply})
    kokoro_ok = speak and _kokoro_available() and _kokoro_service_ready()
    if kokoro_ok and not cancel.is_set():
        tts_started = time.time()
        seq = 0
        for sentence in _pop_sentences(reply, flush=True)[0] or [reply]:
            if cancel.is_set():
                break
            speakable = _tts_normalize(sentence)
            if not speakable:
                continue
            try:
                wav = await asyncio.to_thread(_kokoro_synth_bytes, speakable)
            except Exception as exc:
                logger.warning("Antigravity TTS failed: %s", exc)
                continue
            seq += 1
            if seq == 1:
                metrics["first_audio_seconds"] = round(time.time() - started, 3)
            await _ws_send(ws, {"type": "audio", "seq": seq, "mime": "audio/wav",
                                "b64": base64.b64encode(wav).decode("ascii"), "text": sentence})
        metrics["tts_seconds"] = round(time.time() - tts_started, 3)
    metrics["tts_engine"] = f"kokoro:{KOKORO_VOICE}" if kokoro_ok else "none"
    return reply, cancel.is_set()


async def _stream_engine_reply(
    ws: WebSocket,
    thread: dict[str, Any],
    user_text: str,
    cancel: asyncio.Event,
    metrics: dict[str, Any],
    *,
    speak: bool = True,
) -> tuple[str, bool]:
    """Stream a CLI-engine reply: JSONL stdout -> tokens -> sentences -> Kokoro -> WS audio.

    Returns (full_reply, interrupted). Mirrors _stream_fast_reply's contract.
    """
    engine_id = thread.get("engine") or "hermes"
    label = AGENT_ENGINES.get(engine_id, {}).get("label", engine_id)

    if engine_id == "antigravity":
        return await _stream_antigravity_reply(ws, thread, user_text, cancel, metrics, speak=speak)

    argv = _engine_argv(
        engine_id,
        user_text,
        thread.get("engine_session_id"),
        thread.get("model"),
        thread.get("reasoning_effort"),
        speak=speak,
    )

    started = time.time()
    if engine_id == "grok":
        live = _grok_live_info(thread.get("engine_session_id"))
        if live:
            metrics["grok_live_warning"] = True
            metrics["grok_live_pid"] = live.get("pid")
            try:
                await _ws_send(ws, {
                    "type": "activity",
                    "engine": "grok",
                    "label": f"session also open in Terminal (pid {live.get('pid')}) — one writer at a time",
                })
            except Exception:
                pass
    first_token_at: float | None = None
    first_audio_sent = False
    reply_parts: list[str] = []
    buffer = ""
    seq = 0
    tts_total = 0.0
    kokoro_ok = speak and _kokoro_available() and _kokoro_service_ready()
    session_id: str | None = None
    engine_error: str | None = None
    last_activity = ""
    timed_out = False

    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def tts_consumer() -> None:
        nonlocal seq, tts_total, first_audio_sent
        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                return
            if cancel.is_set():
                continue
            speakable = _tts_normalize(sentence)
            if not speakable or not kokoro_ok:
                continue
            try:
                tts_started = time.time()
                wav = await asyncio.to_thread(_kokoro_synth_bytes, speakable)
                tts_total += time.time() - tts_started
            except Exception as exc:
                logger.warning("Engine sentence TTS failed: %s", exc)
                continue
            if cancel.is_set():
                continue
            if not first_audio_sent:
                first_audio_sent = True
                metrics["first_audio_seconds"] = round(time.time() - started, 3)
            seq += 1
            await _ws_send(ws, {
                "type": "audio",
                "seq": seq,
                "mime": "audio/wav",
                "b64": base64.b64encode(wav).decode("ascii"),
                "text": sentence,
            })

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(DEFAULT_CWD),
        env=_engine_env(engine_id),
        start_new_session=True,
        limit=8 * 1024 * 1024,
    )

    def _kill_proc() -> None:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

    async def watchdog() -> None:
        nonlocal timed_out
        try:
            await asyncio.wait_for(cancel.wait(), timeout=ENGINE_TURN_TIMEOUT)
        except asyncio.TimeoutError:
            timed_out = True
        _kill_proc()

    consumer_task = asyncio.create_task(tts_consumer())
    watchdog_task = asyncio.create_task(watchdog())
    stderr_task = asyncio.create_task(proc.stderr.read())
    interrupted = False
    try:
        while True:
            try:
                line_bytes = await proc.stdout.readline()
            except (ValueError, asyncio.LimitOverrunError):
                continue  # over-long non-JSON noise; skip
            if not line_bytes:
                break
            if cancel.is_set():
                interrupted = True
                break
            for kind, value in _parse_engine_line(engine_id, line_bytes.decode("utf-8", "replace")):
                if kind == "token" and value:
                    if first_token_at is None:
                        first_token_at = time.time()
                        metrics["first_token_seconds"] = round(first_token_at - started, 3)
                    reply_parts.append(value)
                    buffer += value
                    await _ws_send(ws, {"type": "token", "text": value})
                    sentences, buffer = _pop_sentences(buffer)
                    for sentence in sentences:
                        sentence_queue.put_nowait(sentence)
                elif kind == "activity" and value != last_activity:
                    last_activity = value
                    await _ws_send(ws, {"type": "activity", "engine": engine_id, "label": value})
                    # Structured execution bridge for shared chat/voice UI.
                    tool_key = value.lower().replace(" ", "_")[:48]
                    call_id = f"{engine_id}_{tool_key}_{int(time.time() * 1000) % 100000}"
                    await _emit_execution_tool(
                        ws,
                        tool_name=tool_key,
                        tool_call_id=call_id,
                        label=value,
                        phase="started",
                    )
                elif kind == "session":
                    session_id = value
                elif kind == "error":
                    engine_error = value

        await proc.wait()
        if not interrupted and not cancel.is_set():
            for sentence in _pop_sentences(buffer, flush=True)[0]:
                sentence_queue.put_nowait(sentence)
            if engine_error and not "".join(reply_parts).strip():
                # nothing real was said — speak the actionable hint if we have one
                hint = _engine_error_hint(engine_id, engine_error)
                if hint:
                    reply_parts.append(hint)
                    await _ws_send(ws, {"type": "token", "text": hint})
                    for sentence in _pop_sentences(hint, flush=True)[0]:
                        sentence_queue.put_nowait(sentence)
    finally:
        watchdog_task.cancel()
        _kill_proc()
        sentence_queue.put_nowait(None)
        try:
            await consumer_task
        except Exception:
            pass

    stderr_text = ""
    try:
        stderr_text = (await stderr_task).decode("utf-8", "replace")
    except Exception:
        pass

    reply = "".join(reply_parts).strip()
    interrupted = interrupted or cancel.is_set()

    if engine_error:
        # engines can surface fatal errors as a normal-looking message (e.g.
        # claude's synthetic "Failed to authenticate"); speak the actionable
        # hint instead, and don't resume this broken session next turn.
        hint = _engine_error_hint(engine_id, f"{engine_error} {reply}")
        if hint:
            reply = hint
        session_id = None
        if thread.get("engine_session_id"):
            # a dead session must never wedge the thread — start fresh next turn
            store.set_engine_session(thread["id"], None)

    if session_id and session_id != thread.get("engine_session_id"):
        store.set_engine_session(thread["id"], session_id)
        metrics["engine_session_id"] = session_id

    metrics["engine"] = engine_id
    metrics["stream_seconds"] = round(time.time() - started, 3)
    metrics["tts_seconds"] = round(tts_total, 3)
    metrics["tts_engine"] = f"kokoro:{KOKORO_VOICE}" if kokoro_ok else "none"
    metrics["transport"] = "engine-cli"

    if not interrupted and not reply:
        if timed_out:
            raise RuntimeError(f"{label} timed out after {int(ENGINE_TURN_TIMEOUT)} seconds.")
        detail = _clean_text(engine_error or stderr_text or f"{label} returned no text")[-400:]
        hint = _engine_error_hint(engine_id, detail)
        raise RuntimeError(hint or f"{label} engine failed: {detail}")
    return reply, interrupted


async def _stream_hermes_agent_ws(
    ws: WebSocket,
    thread: dict[str, Any],
    user_text: str,
    cancel: asyncio.Event,
    metrics: dict[str, Any],
    *,
    speak: bool = True,
) -> tuple[str, bool]:
    """Stream agent-mode Hermes with structured execution events over the WS."""
    from engines.hermes_stream import stream_hermes_agent_turn

    session_id = (thread.get("hermes_session_id") or "").strip() or f"voice-{thread['id']}"
    if not thread.get("hermes_session_id"):
        try:
            store.set_hermes_session(thread["id"], session_id)
        except Exception:
            pass

    goal_line = f"Current voice-room goal: {thread['goal']}\n" if thread.get("goal") else ""
    ephemeral = (
        f"{VOICE_AGENT_SCOPE}\n\n"
        "You are Hermes in a live phone-controlled voice room. "
        "Replies may be spoken aloud: be clear and concrete. "
        f"{goal_line}"
    )

    # Build a light history for session continuity when the Hermes session is new.
    history: list[dict[str, str]] = []
    if not thread.get("hermes_session_id"):
        for turn in store.recent_turns(thread["id"], limit=8):
            if turn.get("transcript"):
                history.append({"role": "user", "content": str(turn["transcript"])})
            if turn.get("reply"):
                history.append({"role": "assistant", "content": str(turn["reply"])})

    audio_seq = 0
    tts_total = 0.0
    kokoro_ok = speak and _kokoro_available() and _kokoro_service_ready()

    async def _send(payload: dict[str, Any]) -> None:
        await _ws_send(ws, payload)

    async def _speak_sentence(sentence: str) -> None:
        nonlocal audio_seq, tts_total
        if not kokoro_ok or cancel.is_set():
            return
        speakable = _tts_normalize(sentence)
        if not speakable:
            return
        try:
            t0 = time.time()
            wav = await asyncio.to_thread(_kokoro_synth_bytes, speakable)
            tts_total += time.time() - t0
            audio_seq += 1
            await _ws_send(
                ws,
                {
                    "type": "audio",
                    "seq": audio_seq,
                    "mime": "audio/wav",
                    "b64": base64.b64encode(wav).decode("ascii"),
                    "text": sentence,
                },
            )
        except Exception as exc:
            logger.warning("Agent-stream TTS failed: %s", exc)

    reply, interrupted, stream_metrics = await stream_hermes_agent_turn(
        agent_dir=HERMES_AGENT,
        user_text=user_text,
        session_id=session_id,
        conversation_history=history,
        ephemeral_system_prompt=ephemeral,
        cancel_event=cancel,
        send=_send,
        speak_sentence=_speak_sentence if speak else None,
        max_iterations=30,
        cwd=str(DEFAULT_CWD),
    )
    metrics.update(stream_metrics)
    if tts_total:
        metrics["tts_seconds"] = round(tts_total, 3)
        metrics["tts_engine"] = f"kokoro:{KOKORO_VOICE}" if kokoro_ok else "none"
    return reply, interrupted


async def _ws_run_turn(
    ws: WebSocket,
    thread_id: str,
    user_text: str,
    mode: str,
    metrics: dict[str, Any],
    started: float,
    *,
    speak: bool = True,
) -> None:
    """Run one voice-room turn over the socket, fast-streamed when possible."""
    thread = store.get_thread(thread_id)
    cancel = runtime.reset_cancel(thread_id)
    metrics.update(_turn_config_metrics())
    metrics["thread_mode"] = thread.get("mode", "fast")
    engine = thread.get("engine") or "hermes"
    turn_id = _WS_TURN_ID.get("")
    logger.info(
        "voice_agent start turn=%s thread=%s engine=%s mode=%s speak=%s",
        turn_id or "-", thread_id, engine, mode, speak,
    )
    if not _engine_available(engine):
        raise RuntimeError(f"{AGENT_ENGINES.get(engine, {}).get('label', engine)} is unavailable: {_engine_unavailable_note(engine)}")
    metrics["engine"] = engine
    if engine != "hermes":
        metrics["model_provider"] = engine
        metrics["model"] = thread.get("model")
        metrics["reasoning_effort"] = thread.get("reasoning_effort") or "default"

    local = _handle_local_command(thread_id, user_text) or _handle_voice_room_status_intent(thread_id, user_text)
    if local is None:
        local = _detect_engine_switch(thread_id, user_text)
    is_slash = user_text.strip().startswith("/")
    build_intent = (
        None if (local is not None or engine != "hermes") else _detect_build_intent(thread_id, user_text)
    )
    save_kind = None if (local is not None or build_intent is not None) else _detect_save_intent(user_text)
    use_engine = engine != "hermes" and local is None and save_kind is None
    use_stream = (
        engine == "hermes"
        and thread.get("mode", "fast") == "fast"
        and not is_slash
        and save_kind is None
        and build_intent is None
    )

    reply = ""
    interrupted = False
    meta: dict[str, Any] = {}
    try:
        if local is not None:
            reply, meta = local
            await _ws_send(ws, {"type": "token", "text": reply})
            if speak and _kokoro_available() and _kokoro_service_ready():
                try:
                    wav = await asyncio.to_thread(_kokoro_synth_bytes, _tts_normalize(reply))
                    await _ws_send(ws, {"type": "audio", "seq": 1, "mime": "audio/wav",
                                        "b64": base64.b64encode(wav).decode("ascii"), "text": reply})
                except Exception:
                    pass
        elif build_intent is not None:
            reply = await asyncio.to_thread(_handle_build_intent, thread_id, user_text)
            reply = reply or "I couldn't act on that build request."
            meta = {"build": True}
            await _ws_send(ws, {"type": "token", "text": reply})
            if speak and _kokoro_available() and _kokoro_service_ready():
                try:
                    wav = await asyncio.to_thread(_kokoro_synth_bytes, _tts_normalize(reply))
                    await _ws_send(ws, {"type": "audio", "seq": 1, "mime": "audio/wav",
                                        "b64": base64.b64encode(wav).decode("ascii"), "text": reply})
                except Exception:
                    pass
            # If a build is now running, watch it and proactively announce
            # completion / heartbeats without the user having to ask.
            with _builds_guard:
                _b = _builds.get(thread_id)
            if _b and _b["proc"].poll() is None:
                asyncio.create_task(_monitor_build(thread_id))
        elif save_kind is not None:
            await _ws_send(ws, {"type": "state", "state": "saving"})
            hermes_started = time.time()
            reply = await asyncio.to_thread(_run_save, thread_id, user_text, save_kind)
            metrics["hermes_seconds"] = round(time.time() - hermes_started, 3)
            metrics["saved"] = save_kind
            meta = {"saved": save_kind}
            await _ws_send(ws, {"type": "token", "text": reply})
            if speak and not cancel.is_set() and _kokoro_available() and _kokoro_service_ready():
                try:
                    wav = await asyncio.to_thread(_kokoro_synth_bytes, _tts_normalize(reply))
                    await _ws_send(ws, {"type": "audio", "seq": 1, "mime": "audio/wav",
                                        "b64": base64.b64encode(wav).decode("ascii"), "text": reply})
                except Exception:
                    pass
        elif use_engine:
            await _ws_send(ws, {"type": "state", "state": "thinking"})
            engine_started = time.time()
            reply, interrupted = await _stream_engine_reply(ws, thread, user_text, cancel, metrics, speak=speak)
            metrics["hermes_seconds"] = round(time.time() - engine_started, 3)
        elif use_stream:
            await _ws_send(ws, {"type": "state", "state": "thinking"})
            hermes_started = time.time()
            reply, interrupted = await _stream_fast_reply(ws, thread, user_text, cancel, metrics, speak=speak)
            metrics["hermes_seconds"] = round(time.time() - hermes_started, 3)
        else:
            # Agent mode: stream real Hermes tool/progress events when possible.
            # Falls back to the legacy CLI path if the in-process agent stream fails.
            await _ws_send(ws, {"type": "state", "state": "thinking"})
            hermes_started = time.time()
            try:
                reply, interrupted = await _stream_hermes_agent_ws(
                    ws, thread, user_text, cancel, metrics, speak=speak,
                )
            except Exception as stream_exc:
                logger.warning(
                    "Hermes agent stream failed (%s); falling back to CLI",
                    _clean_text(str(stream_exc))[-200:],
                )
                metrics["agent_stream_fallback"] = "cli"
                reply = await asyncio.to_thread(_run_hermes_prompt_for_thread, thread, user_text)
                interrupted = cancel.is_set()
                if reply and not interrupted:
                    await _ws_send(ws, {"type": "token", "text": reply})
                    if speak and not cancel.is_set() and _kokoro_available() and _kokoro_service_ready():
                        try:
                            tts_started = time.time()
                            wav = await asyncio.to_thread(_kokoro_synth_bytes, _tts_normalize(reply)[:4000])
                            metrics["tts_seconds"] = round(time.time() - tts_started, 3)
                            await _ws_send(ws, {"type": "audio", "seq": 1, "mime": "audio/wav",
                                                "b64": base64.b64encode(wav).decode("ascii"), "text": reply})
                        except Exception:
                            pass
            metrics["hermes_seconds"] = round(time.time() - hermes_started, 3)
    except InterruptedTurn:
        interrupted = True
    except Exception as exc:
        logger.exception("WS turn failed")
        detail = _clean_text(str(exc))[-600:]
        record = store.add_turn(
            thread_id=thread_id, mode=mode, user_text=user_text,
            reply=f"Turn failed: {detail}", started=started, metrics=metrics,
        )
        await _ws_send(ws, {"type": "error", "message": detail, "turn": record})
        logger.warning("voice_agent failed turn=%s engine=%s detail=%s", turn_id or "-", engine, detail)
        return

    if interrupted or cancel.is_set():
        record = store.add_turn(
            thread_id=thread_id, mode=mode, user_text=user_text,
            reply=(reply + " …" if reply else "Interrupted."), started=started,
            metrics=metrics, interrupted=True,
        )
        await _ws_send(ws, {"type": "done", "ok": False, "reason": "interrupted", "turn": record,
                            "thread": store.get_thread(thread_id)})
        logger.info("voice_agent interrupted turn=%s engine=%s", turn_id or "-", engine)
        return

    record = store.add_turn(
        thread_id=thread_id, mode=mode, user_text=user_text,
        reply=reply, started=started, metrics=metrics,
    )
    await _ws_send(ws, {
        "type": "done", "ok": True, "turn": record,
        "thread": store.get_thread(thread_id),
        "new_thread": meta.get("new_thread") if meta else None,
    })
    logger.info(
        "voice_agent done turn=%s engine=%s reply_chars=%s latency=%.2f",
        turn_id or "-", engine, len(reply), float(record.get("latency_seconds") or 0),
    )


def _run_hermes_prompt_for_thread(thread: dict[str, Any], user_text: str) -> str:
    """Agent-mode WS turns reuse the full CLI path (tools, memory, sessions)."""
    return _ask_hermes(thread["id"], user_text, "voice")


# thread_id -> set of live sockets, so a background build can push an
# unsolicited "it's done" to whoever is watching that thread.
_thread_sockets: dict[str, set[WebSocket]] = {}
_thread_sockets_guard = threading.Lock()
_build_monitors: set[str] = set()


def _subscribe_socket(thread_id: str, ws: WebSocket) -> None:
    if not thread_id:
        return
    with _thread_sockets_guard:
        _thread_sockets.setdefault(thread_id, set()).add(ws)


def _unsubscribe_socket(ws: WebSocket) -> None:
    with _thread_sockets_guard:
        for subs in _thread_sockets.values():
            subs.discard(ws)


async def _push_to_thread(thread_id: str, payload: dict[str, Any]) -> None:
    with _thread_sockets_guard:
        targets = list(_thread_sockets.get(thread_id, ()))
    for ws in targets:
        await _ws_send(ws, payload)


async def _announce(thread_id: str, kind: str, text: str, *, speak: bool = True) -> None:
    """Push an unsolicited spoken message (e.g. build finished) to the thread."""
    payload: dict[str, Any] = {"type": kind, "text": text, "thread_id": thread_id}
    if speak and _kokoro_available() and _kokoro_service_ready():
        try:
            wav = await asyncio.to_thread(_kokoro_synth_bytes, _tts_normalize(text))
            payload["b64"] = base64.b64encode(wav).decode("ascii")
            payload["mime"] = "audio/wav"
        except Exception:
            pass
    await _push_to_thread(thread_id, payload)


async def _monitor_build(thread_id: str) -> None:
    """Watch a background build; announce completion (and periodic heartbeats)
    to the thread's sockets without the user having to ask."""
    if thread_id in _build_monitors:
        return
    _build_monitors.add(thread_id)
    last_heartbeat = time.time()
    try:
        while True:
            await asyncio.sleep(4)
            with _builds_guard:
                b = _builds.get(thread_id)
            if not b:
                return
            done = b["proc"].poll() is not None
            if done:
                rc = b["proc"].returncode
                if rc == 0:
                    msg = f"Heads up — Composer just finished building {b['goal'][:90]}. It's ready whenever you want to review it."
                else:
                    msg = f"Heads up — the background build of {b['goal'][:80]} stopped with an error (exit code {rc}). Want me to look at what went wrong?"
                await _announce(thread_id, "build_done", msg)
                return
            # Optional spoken heartbeat every 5 minutes for long builds.
            if time.time() - last_heartbeat >= 300:
                last_heartbeat = time.time()
                mins = max(1, int((time.time() - b["started"]) // 60))
                note = f"Quick update — Composer's still working on {b['goal'][:70]}, about {mins} minutes in. Latest step: {_build_last_activity(b['log_path'])[:100]}."
                await _announce(thread_id, "build_heartbeat", note)
    finally:
        _build_monitors.discard(thread_id)


@app.websocket("/ws/voice")
async def voice_socket(ws: WebSocket) -> None:
    peer = ws.client.host if ws.client else ""
    await ws.accept()
    try:
        hello_raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        hello = json.loads(hello_raw)
    except WebSocketDisconnect:
        logger.info("voice_ws disconnected_before_hello peer=%s", peer)
        return
    except Exception as exc:
        logger.warning(
            "voice_ws hello_failed peer=%s kind=%s error=%s",
            peer,
            type(exc).__name__,
            _clean_text(str(exc))[-160:],
        )
        try:
            await ws.close(code=4401)
        except Exception:
            pass
        return
    host_header = ""
    try:
        host_header = ws.headers.get("host") or ""
    except Exception:
        host_header = ""
    # ngrok terminates on loopback — never treat tunnel WS as "local".
    local_ok = _connection_is_local(host=peer, host_header=host_header)
    if not (
        local_ok
        or VOICE_PUBLIC_NO_AUTH
        or _password_ok(str(hello.get("password") or ""))
        or _device_session_ok(str(hello.get("device_session") or ""))
    ):
        # Fail closed: public tunnel without password/session cannot reach the Mac.
        if not VOICE_PASSWORD and not local_ok:
            await _ws_send(ws, {"type": "error", "message": "Voice room is locked. Set HERMES_VOICE_PASSWORD."})
        else:
            await _ws_send(ws, {"type": "error", "message": "Bad password or expired device session"})
        await ws.close(code=4401)
        return

    await _ws_send(ws, {
        "type": "ready",
        "protocols": ["batch-v1", "pcm16-v1", "pcm16-prefetch-v1", "pcm16-rolling-v1"],
    })
    hello_thread = str(hello.get("thread_id") or "")
    logger.info("voice_ws ready peer=%s thread=%s", peer, hello_thread or "-")
    if hello_thread:
        _subscribe_socket(hello_thread, ws)

    pending_turn: dict[str, Any] | None = None
    pcm_turn: dict[str, Any] | None = None
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if pcm_turn is not None:
                    chunk = message["bytes"]
                    pcm_bytes = pcm_turn["audio"]
                    if len(pcm_bytes) + len(chunk) <= MAX_AUDIO_BYTES:
                        pcm_bytes.extend(chunk)
                        pcm_turn["chunk_count"] = int(pcm_turn.get("chunk_count") or 0) + 1
                        if pcm_turn["chunk_count"] == 1:
                            logger.info(
                                "voice_pcm first_chunk peer=%s turn=%s bytes=%s",
                                peer,
                                str(pcm_turn["header"].get("turn_id") or "-"),
                                len(chunk),
                            )
                    else:
                        await _ws_send(ws, {"type": "error", "message": "PCM audio stream is too large"})
                        _stop_pcm_stream(pcm_turn)
                        pcm_turn = None
                    continue
                if not pending_turn:
                    continue
                header = pending_turn
                pending_turn = None
                audio_bytes = message["bytes"]
                _subscribe_socket(str(header.get("thread_id") or ""), ws)
                asyncio.create_task(_run_ws_turn_context(
                    str(header.get("turn_id") or ""),
                    _ws_voice_turn(ws, header, audio_bytes),
                ))
                continue
            raw = message.get("text")
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            kind = data.get("type")
            if kind == "turn":
                pending_turn = data
                _subscribe_socket(str(data.get("thread_id") or ""), ws)
            elif kind == "pcm_start":
                try:
                    sample_rate = int(data.get("sample_rate") or 16000)
                except (TypeError, ValueError):
                    sample_rate = 0
                if sample_rate not in {8000, 16000, 24000, 32000, 44100, 48000}:
                    await _ws_send(ws, {"type": "error", "message": "Unsupported PCM sample rate"})
                    continue
                if pcm_turn is not None:
                    _stop_pcm_stream(pcm_turn)
                pcm_turn = {
                    "header": data,
                    "sample_rate": sample_rate,
                    "audio": bytearray(),
                    "started_at": time.time(),
                    "chunk_count": 0,
                    "generation": 0,
                    "prefetch_task": None,
                    "prefetch_generation": -1,
                    "active": True,
                    "rolling_task": None,
                    "rolling_loop": None,
                    "rolling_result": None,
                    "partial_count": 0,
                    "last_rolling_started_at": 0.0,
                }
                pcm_turn["rolling_loop"] = asyncio.create_task(
                    _rolling_partial_loop(ws, pcm_turn)
                )
                pcm_turn["rolling_loop"].add_done_callback(_consume_background_task_result)
                _subscribe_socket(str(data.get("thread_id") or ""), ws)
                logger.info(
                    "voice_pcm start peer=%s turn=%s thread=%s rate=%s",
                    peer,
                    str(data.get("turn_id") or "-"),
                    str(data.get("thread_id") or "-"),
                    sample_rate,
                )
            elif kind == "voice_capture_probe":
                logger.info(
                    "voice_capture probe peer=%s turn=%s rms=%.6f peak=%.6f threshold=%.6f",
                    peer,
                    str(data.get("turn_id") or "-"),
                    float(data.get("rms") or 0.0),
                    float(data.get("peak") or 0.0),
                    float(data.get("threshold") or 0.0),
                )
            elif kind == "voice_playback_event":
                logger.info(
                    "voice_playback event=%s peer=%s turn=%s bytes=%s detail=%s",
                    _clean_text(str(data.get("event") or "unknown"))[:40],
                    peer,
                    str(data.get("turn_id") or "-"),
                    int(data.get("bytes") or 0),
                    _clean_text(str(data.get("detail") or "-"))[:160],
                )
            elif kind == "pcm_endpoint_candidate":
                stream = pcm_turn
                if stream is None or str(data.get("turn_id") or "") != str(stream["header"].get("turn_id") or ""):
                    continue
                task = stream.get("prefetch_task")
                if task is None or task.done():
                    snapshot = bytes(stream["audio"])
                    logger.info(
                        "voice_pcm candidate peer=%s turn=%s bytes=%s chunks=%s",
                        peer,
                        str(stream["header"].get("turn_id") or "-"),
                        len(snapshot),
                        int(stream.get("chunk_count") or 0),
                    )
                    stream["prefetch_generation"] = stream["generation"]
                    stream["prefetch_audio_bytes"] = len(snapshot)
                    prefetch_task = asyncio.create_task(
                        _transcribe_pcm_snapshot(snapshot, stream["sample_rate"])
                    )
                    prefetch_task.add_done_callback(_consume_background_task_result)
                    stream["prefetch_task"] = prefetch_task
            elif kind == "pcm_resume":
                stream = pcm_turn
                if stream is not None and str(data.get("turn_id") or "") == str(stream["header"].get("turn_id") or ""):
                    stream["generation"] += 1
                    logger.info(
                        "voice_pcm resume peer=%s turn=%s generation=%s",
                        peer,
                        str(data.get("turn_id") or "-"),
                        stream["generation"],
                    )
            elif kind == "pcm_end":
                stream = pcm_turn
                pcm_turn = None
                if stream is None or str(data.get("turn_id") or "") != str(stream["header"].get("turn_id") or ""):
                    if stream is not None:
                        _stop_pcm_stream(stream)
                    continue
                stream["active"] = False
                header = stream["header"]
                pcm = bytes(stream["audio"])
                logger.info(
                    "voice_pcm end peer=%s turn=%s bytes=%s chunks=%s seconds=%.3f",
                    peer,
                    str(header.get("turn_id") or "-"),
                    len(pcm),
                    int(stream.get("chunk_count") or 0),
                    time.time() - float(stream.get("started_at") or time.time()),
                )
                if len(pcm) < int(stream["sample_rate"] * 0.1) * 2:
                    _stop_pcm_stream(stream)
                    await _ws_send(ws, {"type": "done", "ok": False, "reason": "no_speech"})
                    continue
                asyncio.create_task(_run_ws_turn_context(
                    str(header.get("turn_id") or ""),
                    _finish_pcm_stream(ws, stream, pcm),
                ))
            elif kind == "pcm_cancel":
                if pcm_turn is not None:
                    logger.info(
                        "voice_pcm cancel peer=%s turn=%s bytes=%s chunks=%s",
                        peer,
                        str(pcm_turn["header"].get("turn_id") or "-"),
                        len(pcm_turn.get("audio") or b""),
                        int(pcm_turn.get("chunk_count") or 0),
                    )
                    _stop_pcm_stream(pcm_turn)
                pcm_turn = None
            elif kind == "text_turn":
                _subscribe_socket(str(data.get("thread_id") or ""), ws)
                asyncio.create_task(_run_ws_turn_context(
                    str(data.get("turn_id") or ""),
                    _ws_text_turn(ws, data),
                ))
            elif kind == "subscribe":
                _subscribe_socket(str(data.get("thread_id") or ""), ws)
            elif kind == "finalize":
                _subscribe_socket(str(data.get("thread_id") or ""), ws)
                asyncio.create_task(_run_ws_turn_context(
                    str(data.get("turn_id") or ""),
                    _ws_finalize_turn(ws, data),
                ))
            elif kind == "interrupt":
                thread_id = str(data.get("thread_id") or "")
                if thread_id:
                    runtime.trigger_cancel(thread_id)
                    await asyncio.to_thread(runtime.interrupt, thread_id)
            elif kind == "client_metrics":
                thread_id = str(data.get("thread_id") or "")
                record_id = str(data.get("record_id") or "")
                clean_metrics: dict[str, float] = {}
                for key in ("client_upload_to_playback_seconds", "client_speech_end_to_playback_seconds"):
                    try:
                        value = float(data.get(key))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= value <= 300:
                        clean_metrics[key] = round(value, 3)
                if thread_id and record_id and clean_metrics:
                    await asyncio.to_thread(store.merge_turn_metrics, record_id, thread_id, clean_metrics)
            elif kind == "ping":
                await _ws_send(ws, {"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Voice socket crashed")
    finally:
        if pcm_turn is not None:
            logger.info(
                "voice_pcm socket_closed peer=%s turn=%s bytes=%s chunks=%s",
                peer,
                str(pcm_turn["header"].get("turn_id") or "-"),
                len(pcm_turn.get("audio") or b""),
                int(pcm_turn.get("chunk_count") or 0),
            )
            _stop_pcm_stream(pcm_turn)
        _unsubscribe_socket(ws)


# Semantic turn-taking: when the user pauses but the transcript so far ends
# mid-thought, we ask the client to keep listening and stitch the continuation
# rather than answering a half-finished sentence. Whisper reliably terminates
# finished utterances with . ! ?, so the absence of it plus a dangling
# function/opener word is a strong "they're not done" signal.
MAX_CONTINUATIONS = 4
# Words that almost never end a *complete* spoken sentence: articles,
# conjunctions, prepositions, possessives, and clear openers. Deliberately
# EXCLUDES pronouns ("thank you", "it's me") and auxiliaries/verbs ("yes I
# can", "that's all I need") which legitimately end sentences — including them
# caused false "keep listening" extensions.
_INCOMPLETE_TAIL_WORDS = frozenset({
    "and", "or", "but", "so", "because", "then", "if", "when", "while", "that",
    "which", "to", "of", "for", "with", "into", "onto", "from", "about", "than",
    "at", "by", "as", "in", "on", "the", "a", "an", "my", "your", "our",
    "their", "his", "her", "its", "please", "lets", "also", "plus", "gonna",
    "wanna", "um", "uh", "er", "hmm",
})


def _looks_incomplete(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith("...") or t.endswith("…"):
        return True
    # Questions / exclamations are reliably finished thoughts.
    if t.endswith("?") or t.endswith("!"):
        return False
    # A trailing comma/dash mid-utterance means more is coming.
    if t[-1] in ",;:-":
        return True
    # Whisper punctuates aggressively, so DON'T trust a trailing period —
    # strip it and judge by the last real word. "I want you to." still ends
    # on a dangling "to" and is clearly unfinished.
    core = t.rstrip(" .,;:-–—")
    words = re.findall(r"[a-zA-Z']+", core.lower())
    if not words:
        return False
    return words[-1] in _INCOMPLETE_TAIL_WORDS


async def _ws_voice_turn(ws: WebSocket, header: dict[str, Any], audio_bytes: bytes) -> None:
    thread_id = str(header.get("thread_id") or "")
    prev_partial = str(header.get("prev_partial") or "").strip()
    try:
        cont_count = int(header.get("cont_count") or 0)
    except (TypeError, ValueError):
        cont_count = 0
    try:
        started = float(header.pop("_received_at", time.time()))
    except (TypeError, ValueError):
        started = time.time()
    prefetched = header.pop("_stt_prefetch", None)
    metrics: dict[str, Any] = {"received_at": started, "audio_bytes": len(audio_bytes)}
    try:
        thread = store.get_thread(thread_id) if thread_id else store.ensure_default_thread()
        thread_id = thread["id"]
        if not audio_bytes or len(audio_bytes) > MAX_AUDIO_BYTES:
            await _ws_send(ws, {"type": "error", "message": "Bad audio upload"})
            return
        async with runtime.thread_lock(thread_id):
            await _ws_send(ws, {"type": "state", "state": "transcribing"})
            if isinstance(prefetched, dict) and prefetched.get("transcript"):
                transcript = str(prefetched["transcript"])
                stt_meta = prefetched.get("meta") if isinstance(prefetched.get("meta"), dict) else {}
                metrics.update(stt_meta)
                metrics["transcribe_seconds"] = float(prefetched.get("endpoint_wait_seconds") or 0)
                metrics["stt_prefetch_reused"] = True
                metrics["stt_prefetch_total_seconds"] = float(prefetched.get("seconds") or 0)
                metrics["stt_prefetch_audio_seconds"] = float(prefetched.get("audio_seconds") or 0)
                if prefetched.get("stt_source"):
                    metrics["stt_source"] = prefetched.get("stt_source")
                if prefetched.get("stt_partial_count") is not None:
                    metrics["stt_partial_count"] = prefetched.get("stt_partial_count")
                if prefetched.get("stt_rolling_age_seconds") is not None:
                    metrics["stt_rolling_age_seconds"] = prefetched.get("stt_rolling_age_seconds")
            else:
                suffix = _suffix_from_mime(str(header.get("mime") or ""))
                with tempfile.TemporaryDirectory(prefix="hermes-voice-ws-") as tmp:
                    audio_path = Path(tmp) / f"input{suffix}"
                    audio_path.write_bytes(audio_bytes)
                    transcribe_started = time.time()
                    transcript, stt_meta = await asyncio.to_thread(_transcribe_audio, audio_path)
                    metrics.update(stt_meta)
                    metrics["transcribe_seconds"] = round(time.time() - transcribe_started, 3)
            if not transcript:
                # Silence after a partial: don't drop it — finalize what we have.
                if prev_partial:
                    await _ws_send(ws, {"type": "transcript", "text": prev_partial})
                    await _ws_run_turn(ws, thread_id, prev_partial, "voice", metrics, started)
                else:
                    await _ws_send(ws, {"type": "done", "ok": False, "reason": "no_speech"})
                return
            transcript = _normalize_voice_transcript(transcript, metrics)
            full = f"{prev_partial} {transcript}".strip() if prev_partial else transcript
            # Slash / build / save commands must resolve immediately, never wait.
            special = full.startswith("/") or _detect_build_intent(thread_id, full) or _detect_save_intent(full)
            if not special and _looks_incomplete(full) and cont_count < MAX_CONTINUATIONS:
                await _ws_send(ws, {"type": "continue", "partial": full, "cont_count": cont_count + 1})
                return
            await _ws_send(ws, {"type": "transcript", "text": full})
            await _ws_run_turn(ws, thread_id, full, "voice", metrics, started)
    except HTTPException as exc:
        await _ws_send(ws, {"type": "error", "message": str(exc.detail)})
    except Exception as exc:
        logger.exception("WS voice turn failed")
        await _ws_send(ws, {"type": "error", "message": _clean_text(str(exc))[-400:]})


async def _ws_finalize_turn(ws: WebSocket, data: dict[str, Any]) -> None:
    """Process a stitched partial as a full turn (user trailed off and stopped)."""
    thread_id = str(data.get("thread_id") or "")
    text = str(data.get("partial") or "").strip()
    started = time.time()
    metrics: dict[str, Any] = {"received_at": started, "finalized": True}
    if not text:
        await _ws_send(ws, {"type": "done", "ok": False, "reason": "no_speech"})
        return
    try:
        thread = store.get_thread(thread_id) if thread_id else store.ensure_default_thread()
        async with runtime.thread_lock(thread["id"]):
            await _ws_send(ws, {"type": "transcript", "text": text})
            await _ws_run_turn(ws, thread["id"], text, "voice", metrics, started)
    except HTTPException as exc:
        await _ws_send(ws, {"type": "error", "message": str(exc.detail)})
    except Exception as exc:
        logger.exception("WS finalize turn failed")
        await _ws_send(ws, {"type": "error", "message": _clean_text(str(exc))[-400:]})


async def _ws_text_turn(ws: WebSocket, data: dict[str, Any]) -> None:
    thread_id = str(data.get("thread_id") or "")
    text = str(data.get("text") or "").strip()
    started = time.time()
    metrics: dict[str, Any] = {"received_at": started}
    if not text:
        return
    try:
        thread = store.get_thread(thread_id) if thread_id else store.ensure_default_thread()
        speak = bool(data.get("speak", False))
        async with runtime.thread_lock(thread["id"]):
            await _ws_run_turn(ws, thread["id"], text, "text", metrics, started, speak=speak)
    except HTTPException as exc:
        await _ws_send(ws, {"type": "error", "message": str(exc.detail)})
    except Exception as exc:
        logger.exception("WS text turn failed")
        await _ws_send(ws, {"type": "error", "message": _clean_text(str(exc))[-400:]})


def _suffix_from_mime(mime: str) -> str:
    mime = mime.split(";", 1)[0].strip().lower()
    if mime == "audio/mp4":
        return ".m4a"
    if mime in {"audio/ogg", "audio/opus"}:
        return ".ogg"
    if mime == "audio/wav":
        return ".wav"
    return ".webm"


def _pcm16_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


async def _transcribe_pcm_snapshot(pcm: bytes, sample_rate: int) -> dict[str, Any]:
    started = time.time()
    wav = _pcm16_to_wav(pcm, sample_rate)
    with tempfile.TemporaryDirectory(prefix="hermes-pcm-prefetch-") as tmpdir:
        audio_path = Path(tmpdir) / "speech.wav"
        audio_path.write_bytes(wav)
        transcript, meta = await asyncio.to_thread(_transcribe_audio, audio_path)
    return {
        "transcript": transcript,
        "meta": meta,
        "seconds": round(time.time() - started, 3),
    }


def _consume_background_task_result(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def _stop_pcm_stream(stream: dict[str, Any] | None) -> None:
    if not stream:
        return
    stream["active"] = False
    loop = stream.get("rolling_loop")
    if loop is not None and not loop.done():
        loop.cancel()
    task = stream.get("rolling_task")
    if task is not None and not task.done():
        task.cancel()


def _pcm_window_bytes(stream: dict[str, Any]) -> bytes:
    """Recent PCM window for rolling STT (full buffer when short)."""
    sample_rate = int(stream.get("sample_rate") or 16000)
    max_bytes = max(1, int(sample_rate * ROLLING_PARTIAL_WINDOW_S) * 2)
    audio = stream.get("audio") or bytearray()
    if len(audio) <= max_bytes:
        return bytes(audio)
    return bytes(audio[-max_bytes:])


async def _rolling_partial_loop(ws: WebSocket, stream: dict[str, Any]) -> None:
    """While speech is active, run at most one Whisper job ~every 0.8s."""
    try:
        while stream.get("active"):
            await asyncio.sleep(0.12)
            if not stream.get("active"):
                break
            await _maybe_schedule_rolling_partial(ws, stream)
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Rolling partial loop failed")


async def _maybe_schedule_rolling_partial(ws: WebSocket, stream: dict[str, Any]) -> None:
    if not stream.get("active"):
        return
    task = stream.get("rolling_task")
    if task is not None and not task.done():
        return
    now = time.time()
    last = float(stream.get("last_rolling_started_at") or 0.0)
    if now - last < ROLLING_PARTIAL_INTERVAL_S:
        return
    sample_rate = int(stream.get("sample_rate") or 16000)
    min_bytes = int(sample_rate * ROLLING_PARTIAL_MIN_AUDIO_S) * 2
    if len(stream.get("audio") or b"") < min_bytes:
        return
    snapshot = _pcm_window_bytes(stream)
    if len(snapshot) < min_bytes:
        return
    generation = int(stream.get("generation") or 0)
    turn_id = str((stream.get("header") or {}).get("turn_id") or "")
    stream["last_rolling_started_at"] = now
    audio_bytes = len(snapshot)

    async def _run() -> None:
        try:
            result = await _transcribe_pcm_snapshot(snapshot, sample_rate)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Rolling partial Whisper failed")
            return
        if not stream.get("active") and stream.get("rolling_result") is None:
            # Endpoint may still consume a just-finished result.
            pass
        stream["partial_count"] = int(stream.get("partial_count") or 0) + 1
        stream["rolling_result"] = {
            **result,
            "at": time.time(),
            "audio_bytes": audio_bytes,
            "generation": generation,
            "source": "rolling",
        }
        text = str(result.get("transcript") or "").strip()
        if text and stream.get("active"):
            try:
                await _ws_send(
                    ws,
                    {
                        "type": "partial_transcript",
                        "text": text,
                        "turn_id": turn_id,
                        "partial_count": stream["partial_count"],
                        "seconds": result.get("seconds"),
                    },
                )
            except Exception:
                pass

    rolling_task = asyncio.create_task(_run())
    rolling_task.add_done_callback(_consume_background_task_result)
    stream["rolling_task"] = rolling_task


async def _resolve_pcm_stt_prefetch(
    stream: dict[str, Any],
    endpoint_received: float,
) -> dict[str, Any] | None:
    """Pick the best ready STT result without stacking Whisper jobs."""
    sample_rate = int(stream.get("sample_rate") or 16000)
    total_audio_bytes = len(stream.get("audio") or b"")
    partial_count = int(stream.get("partial_count") or 0)

    def _pack(source: str, result: dict[str, Any], audio_bytes: float, **extra: Any) -> dict[str, Any]:
        return {
            **result,
            "endpoint_wait_seconds": round(time.time() - endpoint_received, 3),
            "audio_seconds": round(float(audio_bytes) / (sample_rate * 2), 3),
            "stt_source": source,
            "stt_partial_count": partial_count,
            **extra,
        }

    # 1) Endpoint silence prefetch (full buffer) — best accuracy when present.
    task = stream.get("prefetch_task")
    if task is not None and stream.get("prefetch_generation") == stream.get("generation"):
        try:
            prefetch = await asyncio.wait_for(asyncio.shield(task), timeout=45)
        except Exception:
            prefetch = None
        else:
            if prefetch and str(prefetch.get("transcript") or "").strip():
                return _pack(
                    "endpoint_prefetch",
                    prefetch,
                    float(stream.get("prefetch_audio_bytes") or total_audio_bytes),
                )

    # 2) Finish an in-flight rolling job (started while user was still talking).
    rolling_task = stream.get("rolling_task")
    if rolling_task is not None and not rolling_task.done():
        try:
            await asyncio.wait_for(asyncio.shield(rolling_task), timeout=1.0)
        except Exception:
            pass

    rolling = stream.get("rolling_result")
    if isinstance(rolling, dict) and str(rolling.get("transcript") or "").strip():
        age = endpoint_received - float(rolling.get("at") or 0.0)
        gen_ok = int(rolling.get("generation") or 0) == int(stream.get("generation") or 0)
        roll_bytes = float(rolling.get("audio_bytes") or 0)
        # Rolling windows are last N seconds; only reuse if it covers ~all audio
        # or the utterance is short enough that the window is the whole thing.
        coverage_ok = total_audio_bytes <= 0 or roll_bytes >= total_audio_bytes * 0.85
        if gen_ok and coverage_ok and 0 <= age <= ROLLING_PARTIAL_MAX_AGE_S:
            return _pack(
                "rolling",
                rolling,
                roll_bytes,
                stt_rolling_age_seconds=round(age, 3),
            )
        if gen_ok and coverage_ok and 0 <= age <= (ROLLING_PARTIAL_MAX_AGE_S + 0.4):
            return _pack(
                "rolling_soft",
                rolling,
                roll_bytes,
                stt_rolling_age_seconds=round(age, 3),
            )

    # 3) Final full-buffer snapshot (single Whisper pass at endpoint).
    try:
        final = await _transcribe_pcm_snapshot(bytes(stream.get("audio") or b""), sample_rate)
    except Exception:
        return None
    if not str(final.get("transcript") or "").strip():
        return None
    return _pack("final_snapshot", final, float(total_audio_bytes))


async def _finish_pcm_stream(ws: WebSocket, stream: dict[str, Any], pcm: bytes) -> None:
    endpoint_received = time.time()
    header = stream["header"]
    turn_id = str(header.get("turn_id") or "-")
    stream["active"] = False
    try:
        stt = await _resolve_pcm_stt_prefetch(stream, endpoint_received)
    finally:
        _stop_pcm_stream(stream)
    header["mime"] = "audio/wav"
    header["_received_at"] = endpoint_received
    if stt and stt.get("transcript"):
        header["_stt_prefetch"] = stt
    logger.info(
        "voice_pcm stt_ready turn=%s source=%s transcript_chars=%s audio_bytes=%s",
        turn_id,
        str((stt or {}).get("stt_source") or "final"),
        len(str((stt or {}).get("transcript") or "")),
        len(pcm),
    )
    wav = _pcm16_to_wav(pcm, stream["sample_rate"])
    await _ws_voice_turn(ws, header, wav)


def _load_hermes_config_document() -> tuple[Any, Any]:
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    data = yaml.load(HERMES_CONFIG_PATH.read_text(encoding="utf-8")) if HERMES_CONFIG_PATH.exists() else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Hermes config is not a YAML mapping")
    return yaml, data


def _save_hermes_config_document(yaml: Any, data: Any) -> None:
    HERMES_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = HERMES_CONFIG_PATH.with_suffix(".yaml.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)
    tmp_path.replace(HERMES_CONFIG_PATH)


def _config_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        value = {}
        data[key] = value
    return value


def _parse_key_values(value: str | dict[str, str] | None, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k).strip(): str(v) for k, v in value.items() if str(k).strip()}
    parsed: dict[str, str] = {}
    for raw in str(value or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
        elif ":" in line:
            key, _, val = line.partition(":")
        else:
            raise HTTPException(status_code=400, detail=f"{field} lines must be KEY=value")
        clean_key = key.strip()
        if not clean_key:
            raise HTTPException(status_code=400, detail=f"{field} contains an empty key")
        parsed[clean_key] = val.strip()
    return parsed


def _parse_args(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    return shlex.split(text) if text else []


def _masked_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(k): "configured" for k in value.keys()}


def _sanitize_mcp_name(name: str) -> str:
    clean = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", clean):
        raise HTTPException(status_code=400, detail="MCP server name must use letters, numbers, dots, underscores, or dashes")
    return clean


def _mcp_server_summary(name: str, cfg: Any) -> dict[str, Any]:
    cfg = cfg if isinstance(cfg, dict) else {}
    transport = "http" if cfg.get("url") else "stdio"
    return {
        "name": name,
        "transport": transport,
        "command": str(cfg.get("command") or ""),
        "args": [str(item) for item in (cfg.get("args") or [])] if isinstance(cfg.get("args"), list) else [],
        "url": str(cfg.get("url") or ""),
        "env": _masked_mapping(cfg.get("env")),
        "headers": _masked_mapping(cfg.get("headers")),
        "timeout": cfg.get("timeout"),
        "connect_timeout": cfg.get("connect_timeout"),
        "keepalive_interval": cfg.get("keepalive_interval"),
    }


def _tools_config_payload(data: dict[str, Any] | None = None) -> dict[str, Any]:
    if data is None:
        _yaml, loaded = _load_hermes_config_document()
        data = loaded
    platform_cfg = data.get("platform_toolsets") if isinstance(data.get("platform_toolsets"), dict) else {}
    cli_toolsets = platform_cfg.get("cli") if isinstance(platform_cfg.get("cli"), list) else ["hermes-cli"]
    mcp_servers = data.get("mcp_servers") if isinstance(data.get("mcp_servers"), dict) else {}
    return {
        "ok": True,
        "config_path": str(HERMES_CONFIG_PATH),
        "platform": "cli",
        "current_toolsets": [str(item) for item in cli_toolsets],
        "available_toolsets": list(AVAILABLE_TOOLSETS),
        "mcp_servers": [_mcp_server_summary(str(name), cfg) for name, cfg in sorted(mcp_servers.items())],
    }


def _maybe_reset_thread_session(thread_id: str | None) -> None:
    if thread_id:
        store.get_thread(thread_id)
        store.set_hermes_session(thread_id, "")


def _xai_model_options() -> dict[str, list[str]]:
    import sys

    if str(HERMES_AGENT) not in sys.path:
        sys.path.insert(0, str(HERMES_AGENT))
    catalog: dict[str, list[str]] = {}
    try:
        from hermes_cli.models import provider_model_ids

        catalog = {provider: list(provider_model_ids(provider)) for provider in XAI_PROVIDERS}
    except Exception:
        logger.exception("Could not load Hermes xAI model catalog")

    # Always surface the currently configured chat model even if catalog lags.
    try:
        _yaml, data = _load_hermes_config_document()
        model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
        current_provider = str(model_cfg.get("provider") or "").strip()
        current = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    except Exception:
        current_provider = ""
        current = ""

    models: dict[str, list[str]] = {}
    for provider in XAI_PROVIDERS:
        seen: set[str] = set()
        ordered: list[str] = []
        provider_models = list(catalog.get(provider, []))
        if current and provider == current_provider:
            provider_models.insert(0, current)
        for model in provider_models:
            if not model or model in seen:
                continue
            seen.add(model)
            ordered.append(model)
        models[provider] = ordered
    return models


def _xai_reasoning_support(models: dict[str, list[str]]) -> dict[str, dict[str, bool]]:
    import sys

    if str(HERMES_AGENT) not in sys.path:
        sys.path.insert(0, str(HERMES_AGENT))
    try:
        from agent.model_metadata import grok_supports_reasoning_effort
    except Exception:
        logger.exception("Could not load Hermes xAI reasoning-effort metadata")

        def grok_supports_reasoning_effort(_model: str) -> bool:
            return False

    return {
        provider: {model: bool(grok_supports_reasoning_effort(model)) for model in model_list}
        for provider, model_list in models.items()
    }


def _hermes_config_payload() -> dict[str, Any]:
    _yaml, data = _load_hermes_config_document()
    model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
    agent_cfg = data.get("agent") if isinstance(data.get("agent"), dict) else {}
    provider = str(model_cfg.get("provider") or "auto").strip()
    model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    reasoning_effort = str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    if reasoning_effort not in REASONING_EFFORTS:
        reasoning_effort = ""
    models = _xai_model_options()
    return {
        "ok": True,
        "config_path": str(HERMES_CONFIG_PATH),
        "current": {
            "provider": provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "approvals_mode": _read_approvals_mode(data),
        },
        "approvals_modes": [
            {"value": "ask", "label": "Ask first", "hint": "Prompts before every risky command. Safest."},
            {"value": "smart", "label": "Smart", "hint": "Auto-runs safe commands, asks on risky ones."},
            {"value": "full", "label": "Full access", "hint": "Runs everything with no prompts. Trust only."},
        ],
        "providers": [
            {"id": "xai-oauth", "label": "xAI OAuth"},
            {"id": "xai", "label": "xAI API"},
        ],
        "models": models,
        "reasoning_support": _xai_reasoning_support(models),
        "reasoning_efforts": [
            {"value": "", "label": "Default"},
            {"value": "none", "label": "None"},
            {"value": "minimal", "label": "Minimal"},
            {"value": "low", "label": "Low"},
            {"value": "medium", "label": "Medium"},
            {"value": "high", "label": "High"},
            {"value": "xhigh", "label": "XHigh"},
        ],
    }


def _patch_hermes_config(payload: HermesConfigPatchRequest) -> dict[str, Any]:
    yaml, data = _load_hermes_config_document()
    models = _xai_model_options()
    model_cfg = data.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        data["model"] = model_cfg
    agent_cfg = data.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
        data["agent"] = agent_cfg

    provider = (payload.provider or str(model_cfg.get("provider") or "xai-oauth")).strip()
    model = (payload.model or str(model_cfg.get("default") or model_cfg.get("model") or "")).strip()
    reasoning_effort = (
        payload.reasoning_effort
        if payload.reasoning_effort is not None
        else str(agent_cfg.get("reasoning_effort") or "")
    )
    reasoning_effort = str(reasoning_effort or "").strip().lower()

    if provider not in XAI_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider for this panel: {provider}")
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")
    if model not in models.get(provider, []):
        raise HTTPException(status_code=400, detail=f"Model is not in Hermes' {provider} catalog: {model}")
    if reasoning_effort not in REASONING_EFFORTS:
        raise HTTPException(status_code=400, detail=f"Invalid reasoning effort: {reasoning_effort}")

    old_provider = str(model_cfg.get("provider") or "")
    old_model = str(model_cfg.get("default") or model_cfg.get("model") or "")
    old_reasoning = str(agent_cfg.get("reasoning_effort") or "")

    model_cfg["provider"] = provider
    model_cfg["default"] = model
    if provider == "xai":
        model_cfg["base_url"] = "https://api.x.ai/v1"
    elif provider == "xai-oauth":
        # OAuth path still hits api.x.ai; keep base_url consistent for streamers.
        model_cfg.setdefault("base_url", "https://api.x.ai/v1")
    agent_cfg["reasoning_effort"] = reasoning_effort

    # Selecting an xAI Grok model in the panel must also switch Fast-mode
    # traffic off the silent Ollama path (that was the "Apply doesn't change
    # anything" bug after a local Qwen fallback was forced).
    if provider in XAI_PROVIDERS:
        try:
            _save_voice_runtime({"fast_provider": "xai"})
        except Exception:
            logger.exception("Failed to pin voice runtime fast_provider=xai")

    approvals_changed = False
    if payload.approvals_mode is not None:
        wanted = payload.approvals_mode.strip().lower()
        if wanted not in APPROVAL_MODES:
            raise HTTPException(status_code=400, detail=f"Invalid access mode: {wanted}")
        approvals_cfg = data.get("approvals")
        if not isinstance(approvals_cfg, dict):
            approvals_cfg = {}
            data["approvals"] = approvals_cfg
        old_mode = _read_approvals_mode(data)
        # "manual"/"smart"/"off" are Hermes' native values; write "off" as a
        # real string so re-reads stay clear rather than YAML's bool False.
        approvals_cfg["mode"] = {"ask": "manual", "smart": "smart", "full": "off"}[wanted]
        approvals_changed = old_mode != wanted

    changed = (
        old_provider != provider
        or old_model != model
        or old_reasoning.strip().lower() != reasoning_effort
        or approvals_changed
    )
    if changed:
        _save_hermes_config_document(yaml, data)

    thread_session_reset = False
    if payload.thread_id:
        store.get_thread(payload.thread_id)
        store.set_hermes_session(payload.thread_id, "")
        thread_session_reset = True

    result = _hermes_config_payload()
    result["changed"] = changed
    result["thread_session_reset"] = thread_session_reset
    return result


def _turn_config_metrics() -> dict[str, Any]:
    try:
        _yaml, data = _load_hermes_config_document()
        model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
        agent_cfg = data.get("agent") if isinstance(data.get("agent"), dict) else {}
        provider = str(model_cfg.get("provider") or "").strip()
        model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
        reasoning_effort = str(agent_cfg.get("reasoning_effort") or "").strip().lower()
        supported = False
        if provider in XAI_PROVIDERS and model:
            supported = _xai_reasoning_support({provider: [model]}).get(provider, {}).get(model, False)
        return {
            "model_provider": provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "reasoning_effort_supported": supported,
        }
    except Exception as exc:
        logger.warning("Could not collect turn config metrics: %s", exc)
        return {}


_SAVE_SKILL_TRIGGERS = (
    "save this command", "save that command", "save this as a skill",
    "save that as a skill", "make a skill", "make this a skill",
    "create a skill", "add a skill", "turn this into a skill",
    "save the command", "save this workflow", "remember this command",
)
# Memory triggers are matched at the START of the (prefix-stripped) transcript
# so conversational "remember when we…" doesn't fire a save.
_SAVE_MEMORY_TRIGGERS = (
    "remember that ", "remember this", "remember to ", "remember my ",
    "remember i ", "remember im ", "remember i'm ", "note that ",
    "make a note", "save to memory", "save this to memory",
    "add to memory", "add this to memory", "keep in mind that",
    "don't forget that ", "dont forget that ", "don't forget to ",
    "dont forget to ", "memorize that", "memorize this",
)
_LEAD_PREFIX_RE = re.compile(r"^(?:hey |ok |okay |yo |hermes[,: ]+)+", re.IGNORECASE)


def _detect_save_intent(user_text: str) -> str | None:
    """Return 'memory' | 'skill' | None for a voice/text turn.

    Recognises explicit /remember and /skill slash commands plus a tight set
    of natural-language triggers, so a normal conversation is never
    accidentally persisted.
    """
    raw = user_text.strip()
    low = raw.lower()
    if low.startswith(("/remember", "/memorize", "/memory ")):
        return "memory"
    if low.startswith(("/skill", "/saveskill")):
        return "skill"
    stripped = _LEAD_PREFIX_RE.sub("", low).strip()
    if any(trig in stripped for trig in _SAVE_SKILL_TRIGGERS):
        return "skill"
    if any(stripped.startswith(trig) for trig in _SAVE_MEMORY_TRIGGERS):
        return "memory"
    if "to memory" in stripped and any(v in stripped for v in ("save", "add", "store", "commit")):
        return "memory"
    return None


def _strip_save_prefix(user_text: str) -> str:
    """Drop a leading /remember or /skill token so the instruction is clean."""
    raw = user_text.strip()
    for cmd in ("/remember", "/memorize", "/memory", "/saveskill", "/skill"):
        if raw.lower().startswith(cmd):
            return raw[len(cmd):].strip() or raw
    return raw


# =========================================================================
# Two-model split: a background BUILDER (composer) runs a long build in a
# detached process while the NARRATOR (the selected voice model, ideally
# grok-4.3) keeps talking to the user and reports the builder's progress.
# This is what lets Hermes "build code and talk at the same time" — the
# blocking single-model turn was the reason it couldn't before.
# =========================================================================

VOICE_BUILDER_MODEL = os.environ.get("HERMES_VOICE_BUILDER_MODEL", "grok-composer-2.5-fast")
BUILD_DIR = DATA_DIR / "builds"
_builds_guard = threading.Lock()
_builds: dict[str, dict[str, Any]] = {}


def _build_log_tail(log_path: Path, lines: int = 40) -> str:
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(_clean_text(raw).splitlines()[-lines:])


_BUILD_LOG_NOISE = re.compile(
    r"^\d{2}:\d{2}:\d{2}\s*-\s*[\w.]+\s*-\s*(?:INFO|DEBUG|WARNING)\s*(?:\[[^\]]*\])?\s*-\s*",
)


def _build_last_activity(log_path: Path) -> str:
    """Best-effort human-readable 'what composer is doing now' from the log.

    Prefers tool/file/command activity over Hermes' internal INFO logging so
    the spoken status sounds like progress, not log noise.
    """
    lines = [l.strip() for l in _build_log_tail(log_path, 80).splitlines() if l.strip()]
    interesting = ("Running", "Writing", "Editing", "Creating", "Reading", "Executing",
                   "$ ", "npm ", "python", "git ", "mkdir", "touch", "Installing", "Tool:")
    for line in reversed(lines):
        if line.startswith("session_id:"):
            continue
        cleaned = _BUILD_LOG_NOISE.sub("", line)
        if any(k in cleaned for k in interesting):
            return cleaned[:160]
    # Fall back to the last non-logging message we can find.
    for line in reversed(lines):
        cleaned = _BUILD_LOG_NOISE.sub("", line)
        if cleaned and not cleaned.startswith("session_id:") and "API call" not in cleaned:
            return cleaned[:160]
    return "working through the task"


def _start_background_build(thread_id: str, goal: str) -> str:
    goal = goal.strip()
    if not goal:
        return "Tell me what to build, for example: build the login backend in the background."
    with _builds_guard:
        existing = _builds.get(thread_id)
        if existing and existing["proc"].poll() is None:
            return f"Composer is already building: {existing['goal'][:100]}. Ask me for a status update, or say stop the build first."
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    log_path = BUILD_DIR / f"{thread_id}.log"
    scoped_goal = f"{VOICE_AGENT_SCOPE}\n\nUser request:\n{goal}"
    cmd = [
        str(HERMES_PYTHON), "-m", "hermes_cli.main", "--cli", "chat",
        "-q", scoped_goal, "-m", VOICE_BUILDER_MODEL,
        "--source", f"{THREAD_SOURCE}-build", "--yolo", "-v",
    ]
    log_handle = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=str(DEFAULT_CWD), env=_hermes_env(), text=True,
        stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
    )
    with _builds_guard:
        _builds[thread_id] = {
            "goal": goal, "proc": proc, "log_path": log_path, "started": time.time(),
        }
    return (
        f"Composer is now building that in the background: {goal[:120]}. "
        "Keep talking to me — I'll track it and let you know how it's going."
    )


def _external_composer_status() -> str:
    """Best-effort status for Composer processes not owned by this server.

    This deliberately avoids inventing a task. It only reports observable OS
    process facts when the in-memory voice-room build registry is empty, which
    happens after backend restarts or when the desktop Hermes app started the
    builder separately.
    """
    try:
        proc = subprocess.run(
            ["ps", "axo", "pid=,etime=,command="],
            text=True,
            capture_output=True,
            timeout=4,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    rows: list[tuple[str, str, str, str]] = []
    for line in proc.stdout.splitlines():
        raw = line.strip()
        if not raw or "grok-composer" not in raw:
            continue
        if "voice_room/server.py" in raw or " rg " in raw:
            continue
        parts = raw.split(None, 2)
        if len(parts) < 3:
            continue
        pid, elapsed, command = parts
        model_match = re.search(r"--model\s+(\S+)", command) or re.search(
            r"(?:^|\s)-m\s+(grok-composer[^\s]*)", command
        )
        session_match = re.search(r"--session-key\s+(\S+)|--resume\s+(\S+)", command)
        model = next((g for g in (model_match.groups() if model_match else ()) if g), "grok-composer")
        session = next((g for g in (session_match.groups() if session_match else ()) if g), "")
        rows.append((pid, elapsed, model, session))
    if not rows:
        return ""
    pid, elapsed, model, session = rows[0]
    session_text = f", session {session}" if session else ""
    return (
        f"I do see a separate Composer process running: {model}, pid {pid}, "
        f"alive for {elapsed}{session_text}. It was not started by this voice-room "
        "thread, so I can confirm it is running but not safely name its task from here."
    )


def _background_build_status(thread_id: str) -> str:
    with _builds_guard:
        b = _builds.get(thread_id)
    if not b:
        external = _external_composer_status()
        if external:
            return external
        return "There's no background build running right now."
    mins = max(0, int((time.time() - b["started"]) // 60))
    dur = f"{mins} minute{'s' if mins != 1 else ''}" if mins else "under a minute"
    last = _build_last_activity(b["log_path"])
    if b["proc"].poll() is None:
        return f"Composer has been building '{b['goal'][:80]}' for about {dur}. Right now it's on: {last}"
    rc = b["proc"].returncode
    outcome = "finished successfully" if rc == 0 else f"stopped with exit code {rc}"
    return f"The background build of '{b['goal'][:80]}' {outcome} after about {dur}. Last step: {last}"


def _stop_background_build(thread_id: str) -> str:
    with _builds_guard:
        b = _builds.get(thread_id)
    if not b or b["proc"].poll() is not None:
        return "There's no active background build to stop."
    _terminate_process_tree(b["proc"])
    return "Stopped the background build."


def _active_build_note(thread_id: str) -> str:
    """A short line injected into the narrator's context so it can proactively
    mention the build. Empty when nothing is running."""
    with _builds_guard:
        b = _builds.get(thread_id)
    if not b:
        return ""
    if b["proc"].poll() is None:
        mins = max(0, int((time.time() - b["started"]) // 60))
        return (
            f"[Background builder status: composer is actively building "
            f"'{b['goal'][:80]}' (~{mins} min in). Latest step: "
            f"{_build_last_activity(b['log_path'])}. If the user asks about the "
            f"build or 'how's it going', report this naturally.]"
        )
    rc = b["proc"].returncode
    state = "finished successfully" if rc == 0 else f"stopped (exit {rc})"
    return (
        f"[Background builder status: composer's build of '{b['goal'][:80]}' has "
        f"{state}. Mention this to the user if relevant.]"
    )


_BUILD_START_TRIGGERS = (
    "build this in the background", "build that in the background",
    "in the background build", "in the background, build",
    "start building", "work on this in the background",
    "work on that in the background", "build me", "go build",
)
_BUILD_STATUS_TRIGGERS = (
    "build status", "how's the build", "how is the build", "hows the build",
    "how's it going with the build", "progress on the build", "status of the build",
    "how's the builder", "check the build", "where's the build",
    "what is composer doing", "what's composer doing", "whats composer doing",
    "what is composer 2 doing", "what is composer 2.5 doing",
    "what's composer 2 doing", "what's composer 2.5 doing",
    "what is the composer doing", "what's the composer doing",
    "is composer building", "is the composer building",
    "is composer still building", "is the composer still building",
    "composer status", "composer progress", "composer doing",
    "builder status", "builder progress",
)
_BUILD_STOP_TRIGGERS = (
    "stop the build", "cancel the build", "kill the build", "stop building",
    "abort the build",
)


_BUILD_VERBS = (
    "build", "create", "make", "implement", "write", "code", "develop",
    "set up", "scaffold", "generate", "add", "refactor", "fix",
)
_BUILD_BACKGROUND_RE = re.compile(r"\bin(?:\s+the)?\s+background\b", re.IGNORECASE)
_BUILD_COMMAND_RE = re.compile(
    r"^(?:hey\s+|yo\s+|hermes[,\s]+|okay\s+|ok\s+|please\s+)*"
    r"(?:(?:i\s+want\s+you\s+to|can\s+you|could\s+you|please|go|start)\s+)?"
    r"(build(?:ing)?|create|make|implement|write|code|develop|set\s+up|scaffold|generate|add|refactor|fix)\b",
    re.IGNORECASE,
)
_BUILD_META_QUESTION_RE = re.compile(
    r"\b("
    r"is\s+that\s+true|right\??|am\s+i|are\s+you|will\s+you|would\s+you|"
    r"when\s+i|if\s+i|i\s+want\s+to\s+ask|two\s+models|basically\s+two\s+models|"
    r"you're\s+gonna|you\s+are\s+gonna|you're\s+going\s+to|you\s+are\s+going\s+to"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_build_meta_question(low: str) -> bool:
    if _BUILD_META_QUESTION_RE.search(low):
        # "Can you build X?" is a command, not a meta-question. Keep it valid
        # unless the utterance is clearly asking about how the two-model setup works.
        if re.search(r"\b(?:can|could)\s+you\s+(?:please\s+)?(?:build|create|make|implement|write|code|develop|set\s+up|scaffold|generate|add|refactor|fix)\b", low):
            return False
        return True
    return False


def _detect_build_intent(thread_id: str, user_text: str) -> tuple[str, str] | None:
    """Return ('start', goal) | ('status', '') | ('stop', '') | None."""
    raw = user_text.strip()
    low = raw.lower()
    normalized_low = re.sub(r"\b(?:composure|compose her|composter)\b", "composer", low)
    if low.startswith("/build"):
        arg = raw[len("/build"):].strip()
        if arg.lower() in {"", "status"}:
            return ("status", "")
        if arg.lower() in {"stop", "cancel", "kill"}:
            return ("stop", "")
        return ("start", arg)
    if any(t in normalized_low for t in _BUILD_STOP_TRIGGERS):
        return ("stop", "")
    if any(t in normalized_low for t in _BUILD_STATUS_TRIGGERS):
        return ("status", "")
    if "composer" in normalized_low and any(word in normalized_low for word in ("doing", "active", "actively", "stopped", "still", "monitor", "working", "progress")):
        return ("status", "")
    # "how's it going" / bare update requests only count when a build is active.
    with _builds_guard:
        has_build = thread_id in _builds
    if has_build and low in {"how's it going", "hows it going", "how is it going", "any update", "any updates"}:
        return ("status", "")
    if _looks_like_build_meta_question(normalized_low):
        return None
    # Primary signal: an explicit "in the background" anywhere in a request
    # that also reads like a build/work task. Robust to natural phrasing like
    # "build a script that … in the background" (the brittle exact-phrase
    # matching missed these and fell through to a blocking inline build).
    if _BUILD_BACKGROUND_RE.search(normalized_low) and any(v in normalized_low for v in _BUILD_VERBS) and _BUILD_COMMAND_RE.search(normalized_low):
        goal = _BUILD_BACKGROUND_RE.sub(" ", raw).strip(" ,.-")
        goal = re.sub(r"\s{2,}", " ", goal)
        return ("start", goal or raw)
    # Convenience openers that imply a background build without saying so.
    for trig in _BUILD_START_TRIGGERS:
        if trig in normalized_low and _BUILD_COMMAND_RE.search(normalized_low):
            idx = normalized_low.find(trig) + len(trig)
            goal = raw[idx:].strip(" :,-") or raw
            return ("start", goal)
    return None


def _handle_build_intent(thread_id: str, user_text: str) -> str | None:
    intent = _detect_build_intent(thread_id, user_text)
    if intent is None:
        return None
    kind, goal = intent
    if kind == "start":
        return _start_background_build(thread_id, goal)
    if kind == "status":
        return _background_build_status(thread_id)
    if kind == "stop":
        return _stop_background_build(thread_id)
    return None


def _run_save(thread_id: str, user_text: str, kind: str) -> str:
    """Persist to Hermes memory or skills via a focused, tool-scoped turn.

    Uses the real Hermes memory/skill tools (shared with the desktop app),
    with recent conversation as context so 'remember that' can resolve
    references to what was just said.
    """
    recent = store.recent_turns(thread_id, limit=6)
    context = "\n".join(
        f"User: {t.get('transcript', '')}\nHermes: {t.get('reply', '')}" for t in recent
    ).strip()
    payload = _strip_save_prefix(user_text)
    if kind == "skill":
        instruction = (
            "You are in a live voice conversation. The user wants to save a reusable "
            "command or workflow as a skill. Use the skill_manage tool to create or "
            "update a concise, well-named skill that captures it. Then reply with ONE "
            "short spoken sentence confirming the skill name you saved — no markdown.\n\n"
            f"Recent conversation:\n{context or '(none)'}\n\nUser's request: {payload}"
        )
        toolset = "skills"
    else:
        instruction = (
            "You are in a live voice conversation. Persist what the user wants "
            "remembered using your memory tool, so it is available in future sessions. "
            "Keep the stored note concise and factual. Then reply with ONE short spoken "
            "sentence confirming what you saved — no markdown.\n\n"
            f"Recent conversation:\n{context or '(none)'}\n\nUser's request: {payload}"
        )
        toolset = "memory"
    return _run_hermes_prompt(thread_id, instruction, "", toolsets=toolset, max_turns=6)


def _handle_voice_room_status_intent(thread_id: str, user_text: str) -> tuple[str, dict[str, Any]] | None:
    low = user_text.strip().lower()
    if (
        any(phrase in low for phrase in ("which model", "what model", "model is this", "who is speaking"))
        and any(word in low for word in ("model", "this", "speaking", "voice"))
    ):
        thread = store.get_thread(thread_id)
        mode = thread.get("mode", "fast")
        engine = thread.get("engine") or "hermes"
        if mode == "fast" and engine == "hermes" and _effective_fast_provider() == "ollama":
            provider = "local Ollama"
            model = OLLAMA_VOICE_MODEL
            effort = "none"
        elif engine != "hermes":
            provider = engine
            model = thread.get("model") or "default"
            effort = thread.get("reasoning_effort") or "default"
        else:
            cfg = _turn_config_metrics()
            provider = cfg.get("model_provider") or "unknown provider"
            model = cfg.get("model") or "unknown model"
            effort = cfg.get("reasoning_effort") or "default"
        return (
            f"This voice room is using {model} through {provider}, with thinking effort {effort}, in {mode} mode.",
            {"status": "model"},
        )
    return None


def _respond_to_text(thread_id: str, user_text: str, mode: str) -> tuple[str, dict[str, Any]]:
    local = _handle_local_command(thread_id, user_text)
    if local is not None:
        return local
    switch = _detect_engine_switch(thread_id, user_text)
    if switch is not None:
        return switch
    status = _handle_voice_room_status_intent(thread_id, user_text)
    if status is not None:
        return status
    thread = store.get_thread(thread_id)
    engine = thread.get("engine") or "hermes"
    if not _engine_available(engine):
        raise HTTPException(
            status_code=503,
            detail=f"{AGENT_ENGINES.get(engine, {}).get('label', engine)} is unavailable: {_engine_unavailable_note(engine)}",
        )
    save_kind = _detect_save_intent(user_text)
    if save_kind is not None:
        return _run_save(thread_id, user_text, save_kind), {"saved": save_kind}
    if engine != "hermes":
        return _ask_engine_blocking(thread, user_text), {
            "engine": engine,
            "model_provider": engine,
            "model": thread.get("model"),
            "reasoning_effort": thread.get("reasoning_effort") or "default",
        }
    build_reply = _handle_build_intent(thread_id, user_text)
    if build_reply is not None:
        return build_reply, {"build": True}
    return _ask_hermes(thread_id, user_text, mode), {}


def _normalize_voice_transcript(transcript: str, metrics: dict[str, Any]) -> str:
    text = transcript.strip()
    compact = re.sub(r"\s+", " ", text).strip()
    corrected = compact
    corrected = re.sub(r"\bcompose her\b", "Composer", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\bcomposure\b", "Composer", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\bcomposter\b", "Composer", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\bcomposer\s+two\s+point\s+five\b", "Composer 2.5", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\bcomposer\s+two\b", "Composer 2", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\bgrok\s+four\s+point\s+three\b", "Grok 4.3", corrected, flags=re.IGNORECASE)
    corrected = re.sub(r"\bgrok\s+for\s+point\s+three\b", "Grok 4.3", corrected, flags=re.IGNORECASE)
    if corrected != compact:
        metrics["raw_transcript"] = transcript
        metrics["normalized_transcript"] = corrected
        compact = corrected
    lowered = compact.lower().rstrip(".!?")
    prefixes = ("slash ", "forward slash ", "forward-slash ")
    for prefix in prefixes:
        if lowered.startswith(prefix):
            command_text = compact[len(prefix) :].strip().rstrip(".!?")
            if command_text:
                normalized = "/" + command_text
                metrics["raw_transcript"] = transcript
                metrics["normalized_transcript"] = normalized
                return normalized
    return text


_ENGINE_SWITCH_RE = re.compile(
    r"^(?:please\s+)?(?:(?:switch|swap|change)(?:\s+(?:this\s+)?(?:chat|thread|conversation))?"
    r"(?:\s+(?:over|back))?\s+to|use|talk\s+to)\s+(?:the\s+)?(?P<name>[a-z][a-z\- ]*?)"
    r"(?:\s+(?:engine|agent|backend))?(?:\s+(?:from\s+now\s+on|for\s+this\s+(?:chat|thread)))?\s*$",
    re.IGNORECASE,
)


def _detect_engine_switch(thread_id: str, user_text: str) -> tuple[str, dict[str, Any]] | None:
    """Voice-friendly engine switching: 'switch to codex', 'use grok build', …

    Full-message match against a whitelist of engine names only, so ordinary
    sentences that merely contain 'use'/'switch' never trigger it.
    """
    text = user_text.strip().lower().rstrip(".!?").strip()
    match = _ENGINE_SWITCH_RE.match(text)
    if not match:
        return None
    engine_id = _ENGINE_ALIASES.get(match.group("name").strip())
    if not engine_id:
        return None
    info = AGENT_ENGINES[engine_id]
    if not _ensure_engine_available(engine_id):
        return (f"I can't switch to {info['label']} — {_engine_unavailable_note(engine_id)}", {})
    thread = store.get_thread(thread_id)
    if (thread.get("engine") or "hermes") == engine_id:
        return (f"This chat is already on {info['label']}.", {})
    store.patch_thread(thread_id, engine=engine_id)
    if engine_id == "hermes":
        return ("Switched this chat back to Hermes.", {})
    if engine_id == "antigravity":
        return ("Switched this chat to Antigravity — it'll run your tasks as background agents in the app.", {})
    return (f"Switched this chat to {info['label']} — full tools, with its own running session.", {})


def _handle_local_command(thread_id: str, user_text: str) -> tuple[str, dict[str, Any]] | None:
    text = user_text.strip()
    if not text.startswith("/"):
        return None
    command, _, rest = text[1:].partition(" ")
    command = command.strip().lower()
    arg = rest.strip()

    if command in {"help", "commands"}:
        return (
            "Voice room commands: /new, /goal <text>, /goal clear, /model, "
            "/model <model>, /model <provider> <model>, /mode fast, /mode agent, "
            "/engine, /engine <grok|codex|claude|antigravity|hermes>, /context, /sessions.",
            {},
        )
    if command in {"engine", "engines"}:
        thread = store.get_thread(thread_id)
        current = thread.get("engine") or "hermes"
        if not arg:
            available = ", ".join(
                info["label"] for eid, info in AGENT_ENGINES.items() if _engine_available(eid)
            )
            return (
                f"Current engine: {AGENT_ENGINES[current]['label']}. Available: {available}. "
                "Say /engine grok, codex, claude, or hermes to switch.",
                {},
            )
        engine_id = _ENGINE_ALIASES.get(arg.lower().strip())
        if not engine_id:
            return (f"Unknown engine: {arg}. Options: hermes, grok, codex, claude, antigravity.", {})
        info = AGENT_ENGINES[engine_id]
        if not _engine_available(engine_id):
            return (f"{info['label']} is unavailable: {_engine_unavailable_note(engine_id)}", {})
        if engine_id == current:
            return (f"This chat is already on {info['label']}.", {})
        store.patch_thread(thread_id, engine=engine_id)
        return (f"Switched this thread to {info['label']}.", {})
    if command == "mode":
        thread = store.get_thread(thread_id)
        if not arg:
            return (f"Current response mode: {thread['mode']}. Say /mode fast or /mode agent to switch.", {})
        wanted = arg.lower().strip()
        if wanted not in THREAD_MODES:
            return (f"Unknown mode: {wanted}. Use fast or agent.", {})
        store.patch_thread(thread_id, mode=wanted)
        label = "fast (no tools, quicker replies)" if wanted == "fast" else "agent (full Hermes tools)"
        return (f"Switched this thread to {label}.", {})
    if command in {"new", "newchat", "reset"}:
        title = arg or None
        new_thread = store.create_thread(title)
        return ("Started a new voice-room thread.", {"new_thread": new_thread})
    if command == "goal":
        thread = store.get_thread(thread_id)
        if not arg:
            return (f"Current goal: {thread['goal'] or 'none'}", {})
        if arg.lower() in {"clear", "off", "none"}:
            store.set_goal(thread_id, "")
            return ("Cleared the voice-room goal.", {})
        thread = store.set_goal(thread_id, arg)
        return (f"Set voice-room goal: {thread['goal']}", {})
    if command in {"context", "ctx"}:
        thread = store.get_thread(thread_id)
        model = _load_model_config()
        engine_id = thread.get("engine") or "hermes"
        return (
            f"Thread: {thread['title']}\n"
            f"Thread ID: {thread['id']}\n"
            f"Engine: {AGENT_ENGINES.get(engine_id, {}).get('label', engine_id)}\n"
            f"Engine session: {thread.get('engine_session_id') or 'not created yet'}\n"
            f"Hermes session: {thread.get('hermes_session_id') or 'not created yet'}\n"
            f"CWD: {DEFAULT_CWD}\n"
            f"TTS: {DEFAULT_TTS_ENGINE} ({KOKORO_VOICE if DEFAULT_TTS_ENGINE == 'kokoro' else DEFAULT_VOICE})\n"
            f"macOS fallback voice: {DEFAULT_VOICE}\n"
            f"Goal: {thread['goal'] or 'none'}\n"
            f"Model provider: {model.get('provider') or 'default'}\n"
            f"Model: {model.get('default') or model.get('model') or 'default'}\n"
            f"Thinking: {_load_reasoning_effort() or 'default'}\n"
            f"Remembered turns: {len(store.recent_turns(thread_id, limit=50))}",
            {},
        )
    if command == "model":
        if not arg:
            model = _load_model_config()
            return (
                f"Current model provider: {model.get('provider') or 'default'}\n"
                f"Current model: {model.get('default') or model.get('model') or 'default'}",
                {},
            )
        return (_set_model_from_command(arg), {})
    if command in {"session", "sessions"}:
        return (_run_sessions_command(arg), {})

    return None


def _ask_hermes(thread_id: str, user_text: str, mode: str) -> str:
    thread = store.get_thread(thread_id)
    hermes_session_id = thread.get("hermes_session_id") or ""
    fast = thread.get("mode", "fast") == "fast"
    goal_line = f"Current voice-room goal: {thread['goal']}\n" if thread["goal"] else ""
    raw_slash = user_text.strip().startswith("/")
    if raw_slash:
        prompt = user_text.strip()
    elif hermes_session_id:
        # The resumed Hermes session already holds the whole conversation and
        # the voice-room primer; re-sending history only bloats the prompt.
        prompt = user_text
    else:
        recent = store.recent_turns(thread_id, limit=8)
        recent_context = "\n".join(
            f"User: {turn.get('transcript', '')}\nHermes: {turn.get('reply', '')}"
            for turn in recent
        )
        prompt = (
            "You are Hermes in a live phone-controlled voice room. Your replies are "
            "spoken aloud: answer naturally and briefly (a few sentences unless asked "
            "for more), no markdown, no lists unless essential. If the user asks for "
            "work, do the useful first step and say what happened. "
            "Return only your assistant reply; never prefix with 'User:' or 'Hermes:'.\n\n"
            f"Voice-room thread: {thread['title']}\n"
            f"{goal_line}"
            f"Recent conversation:\n{recent_context or '(none yet)'}\n\n"
            f"Input mode: {mode}\n"
            f"User: {user_text}"
        )
    return _run_hermes_prompt(thread_id, prompt, hermes_session_id, fast=fast)


def _run_hermes_prompt(
    thread_id: str,
    prompt: str,
    hermes_session_id: str = "",
    *,
    allow_session_retry: bool = True,
    fast: bool = False,
    toolsets: str | None = None,
    max_turns: int = 15,
) -> str:
    scoped_prompt = f"{VOICE_AGENT_SCOPE}\n\n{prompt}"
    cmd = [
        str(HERMES_PYTHON),
        "-m",
        "hermes_cli.main",
        "--cli",
        "chat",
        "-q",
        scoped_prompt,
        "--source",
        THREAD_SOURCE,
        "--quiet",
    ]
    if toolsets:
        # Focused toolset (e.g. "memory" / "skills") for save actions. Note:
        # NO --ignore-rules here — that flag disables the memory/skill tools
        # ("Memory tool isn't available here"), so saves would silently no-op.
        cmd.extend(["-t", toolsets, "--max-turns", str(max_turns)])
    elif fast:
        # Fast phone turns must not expose an interactive tool. In particular,
        # clarify blocks a headless request for 120 seconds waiting for input.
        # An unknown toolset disables tools; its warning is filtered below.
        cmd.extend(["-t", "none", "--ignore-rules", "--max-turns", "1"])
    if hermes_session_id:
        cmd.extend(["--resume", hermes_session_id])

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(DEFAULT_CWD),
        env=_hermes_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    runtime.set_active(thread_id, proc)
    try:
        stdout, stderr = proc.communicate(timeout=240)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(proc)
        if runtime.was_interrupted_since(thread_id, started):
            raise InterruptedTurn() from exc
        raise HTTPException(status_code=504, detail="Hermes turn timed out") from exc
    finally:
        runtime.clear_active(thread_id, proc)

    if runtime.was_interrupted_since(thread_id, started):
        raise InterruptedTurn()

    combined = f"{stderr or ''}\n{stdout or ''}"
    session_match = HERMES_SESSION_RE.search(combined)
    if session_match:
        store.set_hermes_session(thread_id, session_match.group(1).strip())

    if proc.returncode != 0:
        detail = _clean_text(stderr or stdout or "Hermes failed")[-1200:]
        if hermes_session_id and allow_session_retry and "Session not found" in detail:
            store.set_hermes_session(thread_id, "")
            logger.warning("Cleared stale Hermes session %s for thread %s and retrying once", hermes_session_id, thread_id)
            return _run_hermes_prompt(
                thread_id, prompt, "", allow_session_retry=False, fast=fast,
                toolsets=toolsets, max_turns=max_turns,
            )
        raise HTTPException(status_code=500, detail=detail)

    reply = _clean_text(stdout or "")
    reply = "\n".join(
        line
        for line in reply.splitlines()
        if not line.strip().startswith("session_id:") and not line.strip().startswith("Warning:")
    ).strip()
    return reply or "Hermes finished, but did not return text."


def _kokoro_available() -> bool:
    return KOKORO_PYTHON.exists() and KOKORO_SCRIPT.exists()


async def _ngrok_tunnel_url() -> str | None:
    """Return the public HTTPS tunnel that fronts THIS voice room (port 8765).

    Multiple ngrok agents can run at once (phone access, other projects). Their
    local inspect APIs land on 4040, 4041, … — never assume 4040 is ours.
    Prefer a tunnel whose backend addr targets 8765; fall back only if unique.
    """
    timeout = httpx.Timeout(0.75, connect=0.25)
    # Cover the default inspect port plus the common overflow ports when another
    # ngrok already claimed 4040.
    api_bases = [
        "http://127.0.0.1:4040/api/tunnels",
        "http://127.0.0.1:4041/api/tunnels",
        "http://127.0.0.1:4042/api/tunnels",
        "http://127.0.0.1:4045/api/tunnels",
    ]
    preferred: list[str] = []
    fallback: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for api in api_bases:
                try:
                    response = await client.get(api)
                    response.raise_for_status()
                    payload = response.json()
                except (httpx.HTTPError, ValueError, TypeError):
                    continue
                tunnels = payload.get("tunnels", []) if isinstance(payload, dict) else []
                for tunnel in tunnels:
                    if not isinstance(tunnel, dict):
                        continue
                    public_url = str(tunnel.get("public_url") or "").rstrip("/")
                    if not (public_url.startswith("https://") and len(public_url) <= 2048):
                        continue
                    cfg = tunnel.get("config") if isinstance(tunnel.get("config"), dict) else {}
                    addr = str(cfg.get("addr") or tunnel.get("addr") or "").lower()
                    # Voice room is always 8765 (https or http local).
                    if "8765" in addr:
                        if public_url not in preferred:
                            preferred.append(public_url)
                    else:
                        if public_url not in fallback:
                            fallback.append(public_url)
    except (httpx.HTTPError, ValueError, TypeError):
        pass
    if preferred:
        return preferred[0]
    # Never advertise a sibling ngrok (other local projects). Better empty than
    # poisoning the phone with a URL that is not the voice room.
    return None


def _kokoro_service_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{KOKORO_SERVICE_URL}/health", timeout=0.5) as response:
            return response.status == 200
    except Exception:
        return False


def _synthesize_speech(thread_id: str, text: str, tmpdir: Path) -> tuple[Path, str, str]:
    if DEFAULT_TTS_ENGINE == "kokoro" and _kokoro_available():
        output_path = tmpdir / "reply.wav"
        try:
            _speak_with_kokoro(thread_id, text, output_path)
            return output_path, "audio/wav", f"kokoro:{KOKORO_VOICE}"
        except InterruptedTurn:
            raise
        except Exception:
            # Keep the room usable if Kokoro fails or a model download stalls.
            pass

    output_path = tmpdir / "reply.m4a"
    _speak_with_macos(thread_id, text, output_path)
    return output_path, "audio/mp4", f"macos-say:{DEFAULT_VOICE}"


def _speak_with_kokoro(thread_id: str, text: str, output_path: Path) -> None:
    if _kokoro_service_ready():
        request = urllib.request.Request(
            f"{KOKORO_SERVICE_URL}/synthesize",
            data=json.dumps({"text": text, "voice": KOKORO_VOICE, "lang": KOKORO_LANG}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise RuntimeError(f"Kokoro service returned HTTP {response.status}")
                output_path.write_bytes(response.read())
                return
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            pass

    _run_registered_command(
        thread_id,
        [
            str(KOKORO_PYTHON),
            str(KOKORO_SCRIPT),
            "--text",
            text,
            "--output",
            str(output_path),
            "--voice",
            KOKORO_VOICE,
            "--lang",
            KOKORO_LANG,
        ],
        timeout=120,
    )


def _speak_with_macos(thread_id: str, text: str, output_path: Path) -> None:
    aiff_path = output_path.with_suffix(".aiff")
    _run_registered_command(
        thread_id,
        ["/usr/bin/say", "-v", DEFAULT_VOICE, "-o", str(aiff_path), text],
        timeout=90,
    )
    _run_registered_command(
        thread_id,
        ["/usr/bin/afconvert", "-f", "m4af", "-d", "aac", str(aiff_path), str(output_path)],
        timeout=90,
    )


def _run_registered_command(thread_id: str, cmd: list[str], *, timeout: float) -> tuple[str, str]:
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(DEFAULT_CWD),
        env=_hermes_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    runtime.set_active(thread_id, proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(proc)
        if runtime.was_interrupted_since(thread_id, started):
            raise InterruptedTurn() from exc
        raise HTTPException(status_code=504, detail=f"Command timed out: {Path(cmd[0]).name}") from exc
    finally:
        runtime.clear_active(thread_id, proc)

    if runtime.was_interrupted_since(thread_id, started):
        raise InterruptedTurn()
    if proc.returncode != 0:
        detail = _clean_text(stderr or stdout or f"{Path(cmd[0]).name} failed")[-1200:]
        raise HTTPException(status_code=500, detail=detail)
    return stdout or "", stderr or ""


def _terminate_process_tree(proc: subprocess.Popen[str]) -> bool:
    if proc.poll() is not None:
        return False
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return False
    deadline = time.time() + 0.8
    while time.time() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    return True


def _load_device_sessions() -> dict[str, Any]:
    try:
        data = json.loads(DEVICE_SESSIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    sessions = data.get("sessions")
    return sessions if isinstance(sessions, dict) else {}


def _save_device_sessions(sessions: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DEVICE_SESSIONS_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps({"sessions": sessions}, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(DEVICE_SESSIONS_PATH)
    try:
        DEVICE_SESSIONS_PATH.chmod(0o600)
    except Exception:
        pass


def _pruned_device_sessions(sessions: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    ts = now if now is not None else time.time()
    return {
        token: session
        for token, session in sessions.items()
        if isinstance(session, dict) and float(session.get("expires_at") or 0) > ts
    }


def _issue_device_session(device_id: str | None, device_label: str | None = None) -> dict[str, Any]:
    now = time.time()
    token = secrets.token_urlsafe(32)
    clean_device_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(device_id or "")).strip("-")[:120] or uuid.uuid4().hex
    clean_label = str(device_label or "").strip()[:160]
    sessions = _pruned_device_sessions(_load_device_sessions(), now)
    expires_at = now + DEVICE_SESSION_TTL_SECONDS
    sessions[token] = {
        "device_id": clean_device_id,
        "device_label": clean_label,
        "created_at": now,
        "last_seen_at": now,
        "expires_at": expires_at,
    }
    _save_device_sessions(sessions)
    return {
        "session_token": token,
        "device_id": clean_device_id,
        "expires_at": int(expires_at),
    }


def _device_session_ok(token: str | None) -> bool:
    session_token = str(token or "").strip()
    if not session_token:
        return False
    now = time.time()
    raw_sessions = _load_device_sessions()
    sessions = _pruned_device_sessions(raw_sessions, now)
    session = sessions.get(session_token)
    if not isinstance(session, dict):
        if len(sessions) != len(raw_sessions):
            _save_device_sessions(sessions)
        return False
    if now - float(session.get("last_seen_at") or 0) > DEVICE_SESSION_REFRESH_SECONDS:
        session["last_seen_at"] = now
        sessions[session_token] = session
        _save_device_sessions(sessions)
    return True



def _host_header_is_local(host_header: str | None) -> bool:
    """False for ngrok / public DNS Host headers (even if TCP peer is 127.0.0.1)."""
    raw = (host_header or "").strip().lower()
    if not raw:
        return True
    host = raw.split(",")[0].strip().split("/")[0]
    if ":" in host and not host.startswith("["):
        # strip port (keep IPv6 in [brackets] simple path above)
        host = host.rsplit(":", 1)[0]
    host = host.strip("[]")
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    if host.endswith(".local"):
        return True
    if host.startswith("10.") or host.startswith("192.168."):
        return True
    if host.startswith("172."):
        try:
            second = int(host.split(".")[1])
            if 16 <= second <= 31:
                return True
        except Exception:
            pass
    # ngrok, cloudflare, vercel-preview tunnels, etc.
    if "ngrok" in host or host.endswith(".loca.lt") or host.endswith(".trycloudflare.com"):
        return False
    return False


def _client_is_private_network(request: Request | None = None, host: str | None = None) -> bool:
    """True for loopback + RFC1918 LAN peer addresses."""
    raw = (host or "").strip()
    if not raw and request is not None and request.client is not None:
        raw = str(request.client.host or "").strip()
    # Starlette may put IPv4-mapped IPv6 as ::ffff:192.168.x.x
    if raw.startswith("::ffff:"):
        raw = raw.split("::ffff:", 1)[1]
    if raw in {"127.0.0.1", "::1", "localhost"}:
        return True
    # X-Forwarded-For only trusted when the direct peer is local (e.g. reverse proxy on box)
    if request is not None:
        peer = str(request.client.host or "") if request.client else ""
        if peer.startswith("::ffff:"):
            peer = peer.split("::ffff:", 1)[1]
        peer_local = peer in {"127.0.0.1", "::1"} or peer.startswith("192.168.") or peer.startswith("10.")
        if peer_local:
            xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
            if xff:
                raw = xff
                if raw.startswith("::ffff:"):
                    raw = raw.split("::ffff:", 1)[1]
    if raw.startswith("10."):
        return True
    if raw.startswith("192.168."):
        return True
    if raw.startswith("172."):
        try:
            second = int(raw.split(".")[1])
            if 16 <= second <= 31:
                return True
        except Exception:
            pass
    return False


def _connection_is_local(
    request: Request | None = None,
    host: str | None = None,
    host_header: str | None = None,
) -> bool:
    """Local only if both peer/network and Host header are private.

    Critical: ngrok and similar tunnels connect to 127.0.0.1 on the Mac. Without
    checking the Host header, public phone traffic would skip the password and
    reach Hermes tools on the machine.
    """
    header = host_header
    if header is None and request is not None:
        header = request.headers.get("host") or ""
    if header and not _host_header_is_local(header):
        return False
    return _client_is_private_network(request=request, host=host)


def _password_ok(provided: str | None) -> bool:
    # Empty server password must NOT authorize the public internet.
    if not VOICE_PASSWORD:
        return False
    return bool(provided) and _constant_time_equal(str(provided), VOICE_PASSWORD)


def _require_password(provided: str | None) -> None:
    if VOICE_PUBLIC_NO_AUTH or _REQUEST_AUTH_OK.get(False):
        return
    # Public/tunnel traffic with no password configured: fail closed.
    if not VOICE_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Voice room is locked. Set HERMES_VOICE_PASSWORD on the Mac backend.",
        )
    if not _password_ok(provided):
        raise HTTPException(status_code=401, detail="Voice room password required")


def _constant_time_equal(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a.encode("utf-8"), b.encode("utf-8")):
        result |= x ^ y
    return result == 0


def _safe_attachment_name(filename: str | None) -> str:
    name = Path(filename or "attachment").name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" ._")
    return name[:140] or "attachment"


def _looks_like_text_attachment(content_type: str, filename: str) -> bool:
    value = f"{content_type} {filename}".lower()
    if content_type.startswith("text/"):
        return True
    return any(
        marker in value
        for marker in (
            "json",
            "csv",
            "markdown",
            "xml",
            "yaml",
            "javascript",
            "typescript",
            "html",
            "css",
            ".txt",
            ".md",
            ".log",
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
        )
    )


def _suffix_for_upload(audio: UploadFile) -> str:
    content_type = (audio.content_type or "").split(";", 1)[0].lower()
    filename_suffix = Path(audio.filename or "").suffix.lower()
    if filename_suffix in {".webm", ".mp3", ".m4a", ".wav", ".ogg", ".opus"}:
        return filename_suffix
    if content_type == "audio/mp4":
        return ".m4a"
    if content_type in {"audio/ogg", "audio/opus"}:
        return ".ogg"
    if content_type == "audio/wav":
        return ".wav"
    return ".webm"


def _whisper_server_reachable() -> bool:
    try:
        with urllib.request.urlopen(
            WHISPER_SERVER_URL.replace("/inference", "/") if WHISPER_SERVER_URL.endswith("/inference") else f"{WHISPER_SERVER_URL.rsplit('/', 1)[0]}/",
            timeout=0.6,
        ) as response:
            return response.status == 200
    except Exception:
        try:
            # Health endpoint used by Whisper Flow
            with urllib.request.urlopen("http://127.0.0.1:12712/health", timeout=0.6) as response:
                return response.status == 200
        except Exception:
            return False


def _stt_provider_order() -> list[str]:
    """Order STT backends from the UI runtime engine preference."""
    engine = _effective_stt_engine()
    order: list[str] = []

    def _add(provider: str) -> None:
        p = (provider or "").strip().lower()
        if p and p not in order:
            order.append(p)

    if engine == "browser":
        # Browser mode finalizes client-side; server only used as emergency fallback.
        _add("whisper_server")
        _add("local")
        return order

    if engine == "whisper_flow":
        _add("whisper_server")
        _add("local")
        _add("configured")
        return order

    # auto
    if _whisper_server_reachable():
        _add("whisper_server")
    for provider in (VOICE_STT_PROVIDER, *VOICE_STT_FALLBACKS):
        _add(provider)
    _add("configured")
    return order


def _transcribe_with_provider(module: Any, audio_path: Path, provider: str) -> dict[str, Any]:
    file_path = str(audio_path)
    if provider in {"whisper_server", "whisper.cpp", "whisper_cpp", "whisper_flow"}:
        return _transcribe_whisper_server(audio_path)
    if provider in {"configured", "auto", "hermes"}:
        return module.transcribe_audio(file_path)
    if provider == "xai":
        return module._transcribe_xai(file_path, VOICE_STT_XAI_MODEL)
    if provider == "openai":
        return module._transcribe_openai(file_path, VOICE_STT_OPENAI_MODEL)
    if provider == "groq":
        return module._transcribe_groq(file_path, "whisper-large-v3-turbo")
    if provider == "local":
        return module._transcribe_local(file_path, VOICE_STT_LOCAL_MODEL)
    if provider == "local_command":
        return module._transcribe_local_command(file_path, VOICE_STT_LOCAL_MODEL)
    raise RuntimeError(f"Unknown voice STT provider: {provider}")


_WHISPER_PROMPT = (
    "Hermes, Grok, Claude, Codex, Ollama, Qwen, Vercel, Supabase, ngrok. "
    "Natural spoken English."
)


def _transcribe_whisper_server(audio_path: Path) -> dict[str, Any]:
    """Transcribe through local Whisper Flow (whisper.cpp HTTP on :12712).

    Browser MediaRecorder formats are normalized to mono 16 kHz WAV because
    whisper-server accepts PCM WAV but rejects WebM/Opus uploads directly.
    """
    lang = str(_load_voice_runtime().get("stt_language") or "en")
    with tempfile.TemporaryDirectory(prefix="hermes-whisper-") as tmpdir:
        wav_path = Path(tmpdir) / "speech.wav"
        if audio_path.suffix.lower() == ".wav":
            # Re-encode even WAV to guarantee 16 kHz mono (PCM from phone may vary).
            converted = subprocess.run(
                [
                    "/opt/homebrew/bin/ffmpeg",
                    "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(audio_path),
                    "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if converted.returncode or not wav_path.exists():
                wav_path.write_bytes(audio_path.read_bytes())
        else:
            converted = subprocess.run(
                [
                    "/opt/homebrew/bin/ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if converted.returncode or not wav_path.exists():
                detail = _clean_text(converted.stderr or converted.stdout or "audio conversion failed")[-300:]
                raise RuntimeError(f"Local STT audio conversion failed: {detail}")

        # Reject near-silent clips early (common phone false endpoints).
        try:
            size = wav_path.stat().st_size
            if size < 16000:  # < ~0.5s mono 16k 16-bit
                return {
                    "success": False,
                    "transcript": "",
                    "provider": "whisper_server",
                    "error": "audio too short",
                }
        except Exception:
            pass

        with wav_path.open("rb") as audio_file:
            response = httpx.post(
                WHISPER_SERVER_URL,
                files={"file": ("speech.wav", audio_file, "audio/wav")},
                data={
                    "temperature": "0.0",
                    "temperature_inc": "0.0",
                    "no_speech_thold": "0.5",
                    "language": lang if lang not in {"", "auto"} else "en",
                    "response_format": "verbose_json",
                    "prompt": _WHISPER_PROMPT,
                },
                timeout=60.0,
            )
        if response.status_code != 200:
            # Retry with minimal payload for older whisper-server builds.
            with wav_path.open("rb") as audio_file:
                response = httpx.post(
                    WHISPER_SERVER_URL,
                    files={"file": ("speech.wav", audio_file, "audio/wav")},
                    data={
                        "temperature": "0.0",
                        "response_format": "json",
                    },
                    timeout=60.0,
                )
        if response.status_code != 200:
            raise RuntimeError(
                f"whisper.cpp returned HTTP {response.status_code}: {_clean_text(response.text)[:240]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("whisper.cpp returned invalid JSON") from exc
        transcript = str(payload.get("text") or "").strip()
        # Drop common hallucination leftovers from silence.
        low = transcript.lower().strip(" .")
        if low in {"", "thank you", "thanks", "you", "the", "a", "um", "uh", "mm", "hmm"}:
            return {
                "success": False,
                "transcript": "",
                "provider": "whisper_server",
                "error": "empty or filler transcript",
            }
        return {
            "success": bool(transcript),
            "transcript": transcript,
            "provider": "whisper_server",
            "error": "" if transcript else "empty transcript",
        }


def _transcribe_audio(audio_path: Path) -> tuple[str, dict[str, Any]]:
    import sys

    if str(HERMES_AGENT) not in sys.path:
        sys.path.insert(0, str(HERMES_AGENT))
    from tools import transcription_tools

    attempts: list[dict[str, str]] = []
    for provider in _stt_provider_order():
        started = time.time()
        try:
            result = _transcribe_with_provider(transcription_tools, audio_path, provider)
        except Exception as exc:
            attempts.append({
                "provider": provider,
                "error": _clean_text(str(exc))[:220],
                "seconds": f"{time.time() - started:.3f}",
            })
            continue
        transcript = str(result.get("transcript") or "").strip()
        actual_provider = str(result.get("provider") or provider)
        if result.get("success") and transcript:
            return transcript, {
                "stt_provider": actual_provider,
                "stt_requested_provider": provider,
                "stt_engine": _effective_stt_engine(),
                "stt_attempts": attempts,
            }
        attempts.append({
            "provider": provider,
            "error": _clean_text(str(result.get("error") or "empty transcript"))[:220],
            "seconds": f"{time.time() - started:.3f}",
        })
    detail = "; ".join(f"{a['provider']}: {a['error']}" for a in attempts) or "Transcription failed"
    raise HTTPException(status_code=500, detail=detail)


def _load_model_config() -> dict[str, Any]:
    try:
        import yaml

        config_path = HERMES_CONFIG_PATH
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        model = data.get("model") if isinstance(data, dict) else {}
        return model if isinstance(model, dict) else {}
    except Exception:
        return {}


def _load_reasoning_effort() -> str:
    try:
        _yaml, data = _load_hermes_config_document()
        agent_cfg = data.get("agent") if isinstance(data.get("agent"), dict) else {}
        return str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    except Exception:
        return ""


def _set_model_from_command(arg: str) -> str:
    parts = arg.split()
    provider = ""
    model = ""
    if len(parts) >= 2:
        provider, model = parts[0], " ".join(parts[1:])
    elif "/" in arg:
        provider, model = arg.split("/", 1)
    else:
        model = arg

    model = model.strip()
    provider = provider.strip()
    if not model:
        return "Usage: /model <model> or /model <provider> <model>"

    if provider:
        _run_config_set("model.provider", provider)
    _run_config_set("model.default", model)
    current = _load_model_config()
    return (
        "Updated Hermes model config.\n"
        f"Provider: {current.get('provider') or provider or 'unchanged'}\n"
        f"Model: {current.get('default') or model}"
    )


def _run_config_set(key: str, value: str) -> None:
    proc = subprocess.run(
        [str(HERMES_PYTHON), "-m", "hermes_cli.main", "config", "set", key, value],
        cwd=str(DEFAULT_CWD),
        env=_hermes_env(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        detail = _clean_text(proc.stderr or proc.stdout or "config set failed").strip()
        raise RuntimeError(detail)


def _run_sessions_command(arg: str) -> str:
    args = arg.strip().split()
    # /sessions grok → list Grok Build sessions for remote attach
    if args and args[0].lower() in {"grok", "gk", "build"}:
        rows = _list_grok_sessions_detailed()
        if not rows:
            return "No Grok Build sessions found. Open Grok in Terminal or start New Grok from the web app."
        lines = ["Grok Build sessions (tap Attach in + menu for full remote control):"]
        for s in rows[:20]:
            badge = "LIVE" if s.get("live") else "idle"
            lines.append(f"- [{badge}] {s.get('title')} · {s.get('id')}")
        lines.append("Use the phone + menu → Grok sessions to attach (writable, full tools).")
        return "\n".join(lines)
    source = THREAD_SOURCE
    limit = "30"
    if args and args[0].lower() == "all":
        source = ""
        args = args[1:]
    if args and args[0].isdigit():
        limit = args[0]
    cmd = [str(HERMES_PYTHON), "-m", "hermes_cli.main", "sessions", "list", "--limit", limit]
    if source:
        cmd.extend(["--source", source])
    proc = subprocess.run(
        cmd,
        cwd=str(DEFAULT_CWD),
        env=_hermes_env(),
        text=True,
        capture_output=True,
        timeout=20,
    )
    output = _clean_text(proc.stdout or proc.stderr or "").strip()
    # Always append a short Grok live summary so phone users see both surfaces.
    try:
        grok_rows = _list_grok_sessions_detailed()
        live = [s for s in grok_rows if s.get("live")]
        extra = f"\n\nGrok Build: {len(grok_rows)} sessions, {len(live)} live. Use /sessions grok or + → Grok sessions."
    except Exception:
        extra = ""
    if not output:
        return ("No Hermes session output." + extra).strip()
    return (output + extra)[-2400:]


def _rename_hermes_session_quietly(session_id: str, title: str) -> None:
    try:
        subprocess.run(
            [str(HERMES_PYTHON), "-m", "hermes_cli.main", "sessions", "rename", session_id, title],
            cwd=str(DEFAULT_CWD),
            env=_hermes_env(),
            text=True,
            capture_output=True,
            timeout=20,
        )
    except Exception:
        pass


def _hermes_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HERMES_HOME", str(HERMES_HOME))
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["TERMINAL_CWD"] = str(DEFAULT_CWD)
    env["HERMES_VOICE_CWD"] = str(DEFAULT_CWD)
    return env


# ── Chat sync: web app ⇆ Hermes desktop/terminal (shared store: state.db) ────
# state.db is the single source of truth for the desktop + terminal Hermes apps.
# We surface those sessions in the web chat list (read-only, WAL-safe) AND mirror
# web-app turns back into a Hermes session via the official SessionDB API, so a
# chat opened in either surface shows up in the other. Sub-agent chats are
# included (user's choice); machine sessions (builds/bench/debug) stay hidden.
import sys as _sys

HERMES_STATE_DB = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "state.db"
SYNC_SOURCES = ("tui", "cli", "voice-room", "subagent")
SYNC_VOICE_MODEL = os.environ.get("HERMES_VOICE_MODEL", "grok-4.5")
_session_db = None
_session_db_lock = threading.Lock()

# short-lived snapshots so a deleted chat can be restored with one tap
_undo_snapshots: dict[str, dict[str, Any]] = {}
_undo_lock = threading.Lock()
UNDO_TTL = 30
DELETED_THREAD_TTL = 86_400


def _state_store():
    """Lazy singleton SessionDB (write) — the store the desktop app also uses."""
    global _session_db
    if _session_db is None:
        with _session_db_lock:
            if _session_db is None:
                try:
                    if str(HERMES_AGENT) not in _sys.path:
                        _sys.path.insert(0, str(HERMES_AGENT))
                    from hermes_state import SessionDB
                    _session_db = SessionDB(HERMES_STATE_DB)
                except Exception as exc:
                    print(f"[sync] SessionDB unavailable: {exc}")
                    _session_db = False
    return _session_db or None


def _state_ro():
    return sqlite3.connect(f"file:{HERMES_STATE_DB}?mode=ro", uri=True, timeout=3)


def _msg_text(content: Any) -> str:
    """Extract plain text from a message body that may be JSON multimodal parts."""
    if content is None:
        return ""
    s = str(content)
    if s[:1] in "[{":
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return " ".join(
                    str(p.get("text", "")) for p in v if isinstance(p, dict) and p.get("type") == "text"
                ).strip()
            if isinstance(v, dict):
                return str(v.get("text") or "")
        except Exception:
            pass
    return s


def _session_preview(conn, sid: str) -> str:
    try:
        r = conn.execute(
            "SELECT content FROM messages WHERE session_id=? AND role='user' AND active=1 "
            "AND content!='' ORDER BY timestamp LIMIT 1", (sid,)
        ).fetchone()
        if r and r[0]:
            return _clean_text(_msg_text(r[0])).replace("\n", " ").strip()[:60]
    except Exception:
        pass
    return ""



# ── Grok Build remote bridge (phone web UI ⇆ CLI sessions; no Hermes.app) ────
GROK_SESSIONS_DIR = Path.home() / ".grok" / "sessions"
GROK_ACTIVE_SESSIONS_PATH = Path.home() / ".grok" / "active_sessions.json"
CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
ANTIGRAVITY_CONV_DIR = Path.home() / ".gemini" / "antigravity" / "conversations"
CLI_SESSION_LIMIT = 40


def _grok_active_map() -> dict[str, dict[str, Any]]:
    """session_id → {pid, cwd, opened_at} for sessions currently open in a Grok TUI."""
    out: dict[str, dict[str, Any]] = {}
    try:
        raw = json.loads(GROK_ACTIVE_SESSIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("session_id") or "").strip()
        if not sid:
            continue
        pid = item.get("pid")
        # Drop stale entries if the process is gone.
        if pid is not None:
            try:
                os.kill(int(pid), 0)
            except (ProcessLookupError, ValueError, PermissionError, OSError):
                # PermissionError still means process exists on some systems; only skip missing.
                try:
                    os.kill(int(pid), 0)
                except ProcessLookupError:
                    continue
                except Exception:
                    pass
        out[sid] = {
            "pid": pid,
            "cwd": str(item.get("cwd") or "") or None,
            "opened_at": item.get("opened_at"),
        }
    return out


def _grok_live_info(session_id: str | None) -> dict[str, Any] | None:
    if not session_id:
        return None
    return _grok_active_map().get(str(session_id).strip())


def _grok_session_meta(session_id: str) -> dict[str, Any]:
    """Title + mtime for a Grok session id from disk summaries/history."""
    sid = session_id.strip()
    title = f"Grok · {sid[:8]}"
    mtime = 0.0
    summary = ""
    cwd = None
    for q in GROK_SESSIONS_DIR.glob(f"*/{sid}/summary.json"):
        try:
            data = json.loads(q.read_text(encoding="utf-8"))
            title = str(
                data.get("generated_title")
                or data.get("session_summary")
                or title
            )[:120]
            summary = str(data.get("session_summary") or "")[:400]
            info = data.get("info") or {}
            if isinstance(info, dict) and info.get("cwd"):
                cwd = str(info.get("cwd"))
            mtime = max(mtime, q.stat().st_mtime)
        except Exception:
            try:
                mtime = max(mtime, q.stat().st_mtime)
            except Exception:
                pass
        break
    if mtime <= 0:
        for hist in GROK_SESSIONS_DIR.glob(f"*/{sid}/chat_history.jsonl"):
            try:
                mtime = hist.stat().st_mtime
            except Exception:
                pass
            break
    return {"id": sid, "title": title, "summary": summary, "updated_at": mtime, "cwd": cwd}


def _list_grok_sessions_detailed() -> list[dict[str, Any]]:
    """All recent Grok sessions + live Terminal badges. Full remote attach targets."""
    active = _grok_active_map()
    by_id: dict[str, dict[str, Any]] = {}
    try:
        summaries = sorted(
            GROK_SESSIONS_DIR.glob("*/*/summary.json"),
            key=lambda q: q.stat().st_mtime,
            reverse=True,
        )[:CLI_SESSION_LIMIT]
        for q in summaries:
            try:
                data = json.loads(q.read_text(encoding="utf-8"))
            except Exception:
                continue
            sid = str((data.get("info") or {}).get("id") or q.parent.name)
            if not sid:
                continue
            meta = _grok_session_meta(sid)
            meta["updated_at"] = max(meta.get("updated_at") or 0, q.stat().st_mtime)
            by_id[sid] = meta
    except Exception:
        pass
    # Include live sessions even if summary missing
    for sid, live in active.items():
        if sid not in by_id:
            by_id[sid] = _grok_session_meta(sid)
        by_id[sid]["cwd"] = live.get("cwd") or by_id[sid].get("cwd")
    # Also surface sessions that only have history
    try:
        for hist in sorted(
            GROK_SESSIONS_DIR.glob("*/*/chat_history.jsonl"),
            key=lambda q: q.stat().st_mtime,
            reverse=True,
        )[:CLI_SESSION_LIMIT]:
            sid = hist.parent.name
            if sid not in by_id:
                by_id[sid] = _grok_session_meta(sid)
            by_id[sid]["updated_at"] = max(
                by_id[sid].get("updated_at") or 0, hist.stat().st_mtime
            )
    except Exception:
        pass

    rows: list[dict[str, Any]] = []
    for sid, meta in by_id.items():
        live = active.get(sid)
        rows.append({
            "id": sid,
            "title": meta.get("title") or f"Grok · {sid[:8]}",
            "summary": meta.get("summary") or "",
            "updated_at": meta.get("updated_at") or 0,
            "cwd": (live or {}).get("cwd") or meta.get("cwd"),
            "live": bool(live),
            "pid": (live or {}).get("pid"),
            "opened_at": (live or {}).get("opened_at"),
            "engine": "grok",
            "writable": True,
        })
    rows.sort(key=lambda r: (not r.get("live"), -(r.get("updated_at") or 0)))
    return rows[:CLI_SESSION_LIMIT]


def _find_thread_by_grok_session(session_id: str) -> dict[str, Any] | None:
    sid = session_id.strip()
    for t in store.list_threads(include_archived=True):
        if (t.get("engine") or "") == "grok" and (t.get("engine_session_id") or "") == sid:
            return t
    return None


def _seed_thread_from_grok_history(thread_id: str, session_id: str) -> int:
    """Import Grok chat_history into web turns if the thread is empty. Returns count added."""
    existing = store.recent_turns(thread_id, limit=1)
    if existing:
        return 0
    turns = _cli_session_turns(f"gk:{session_id}")
    if not turns:
        return 0
    added = 0
    base = time.time() - len(turns)
    for i, turn in enumerate(turns):
        user = str(turn.get("transcript") or "").strip()
        reply = str(turn.get("reply") or "").strip()
        if not user and not reply:
            continue
        store.add_turn(
            thread_id=thread_id,
            mode="text",
            user_text=user or "(earlier Grok turn)",
            reply=reply or "(no text)",
            started=base + i,
            metrics={"imported_from": "grok", "engine_session_id": session_id},
            interrupted=False,
            mirror=False,
        )
        added += 1
    return added


def _attach_grok_session(
    session_id: str,
    *,
    title: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Promote a Grok CLI session into a writable web thread (full remote control)."""
    sid = (session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    if not _engine_available("grok"):
        raise HTTPException(
            status_code=503,
            detail=f"Grok Build unavailable: {_engine_unavailable_note('grok')}",
        )
    meta = _grok_session_meta(sid)
    clean_title = (title or "").strip() or meta.get("title") or f"Grok · {sid[:8]}"

    if thread_id:
        thread = store.get_thread(thread_id)
        store.patch_thread(thread_id, engine="grok", title=clean_title)
        store.set_engine_session(thread_id, sid)
        thread = store.get_thread(thread_id)
    else:
        existing = _find_thread_by_grok_session(sid)
        if existing:
            if not existing.get("archived"):
                thread = existing
            else:
                thread = store.patch_thread(existing["id"], archived=False, title=clean_title)
                store.set_engine_session(existing["id"], sid)
                thread = store.get_thread(existing["id"])
        else:
            thread = store.create_thread(
                title=clean_title,
                engine="grok",
                engine_session_id=sid,
            )

    imported = _seed_thread_from_grok_history(thread["id"], sid)
    live = _grok_live_info(sid)
    return {
        "ok": True,
        "thread": store.get_thread(thread["id"]),
        "session_id": sid,
        "imported_turns": imported,
        "live": bool(live),
        "live_info": live,
        "writable": True,
    }


def _new_grok_session_thread(title: str | None = None) -> dict[str, Any]:
    """Create a writable Grok thread; session id appears after the first phone turn."""
    if not _engine_available("grok"):
        raise HTTPException(
            status_code=503,
            detail=f"Grok Build unavailable: {_engine_unavailable_note('grok')}",
        )
    clean = (title or "").strip() or f"Grok · {time.strftime('%H:%M')}"
    thread = store.create_thread(title=clean, engine="grok", engine_session_id=None)
    return {"ok": True, "thread": thread, "session_id": None, "writable": True, "live": False}





def _codex_message_texts(payload: dict) -> tuple[str | None, str | None]:
    """(role, text) for a codex rollout response_item, else (None, None)."""
    if not isinstance(payload, dict) or payload.get("type") != "message":
        return None, None
    role = str(payload.get("role") or "")
    if role not in ("user", "assistant"):
        return None, None
    parts = []
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("text"):
            parts.append(str(block["text"]))
    text = "\n".join(parts).strip()
    if not text:
        return None, None
    # Skip injected preambles that pose as user messages.
    if role == "user" and (text.startswith("# AGENTS.md") or text.startswith("<")):
        return None, None
    return role, text


def _list_cli_sessions() -> list[dict[str, Any]]:
    """Grok / Codex / Antigravity terminal sessions as read-only threads."""
    out: list[dict[str, Any]] = []
    # --- Grok Build (list + live badge; attach for full remote write) ---
    try:
        live_map = _grok_active_map()
        summaries = sorted(
            GROK_SESSIONS_DIR.glob("*/*/summary.json"),
            key=lambda q: q.stat().st_mtime, reverse=True,
        )[:CLI_SESSION_LIMIT]
        for q in summaries:
            try:
                data = json.loads(q.read_text(encoding="utf-8"))
            except Exception:
                continue
            sid = str((data.get("info") or {}).get("id") or q.parent.name)
            title = str(data.get("generated_title") or data.get("session_summary") or f"Grok · {sid[:8]}")
            mtime = q.stat().st_mtime
            live = live_map.get(sid)
            out.append({
                "id": f"gk:{sid}", "title": title[:80], "goal": "", "mode": "agent",
                "engine": "grok", "source": "grok-cli", "external": 1, "archived": 0,
                "hermes_session_id": None, "engine_session_id": sid,
                "live": bool(live), "writable": True,
                "created_at": mtime, "updated_at": mtime,
            })
        # Live sessions without a summary still appear
        seen = {item["engine_session_id"] for item in out if item.get("engine") == "grok"}
        for sid, live in live_map.items():
            if sid in seen:
                continue
            meta = _grok_session_meta(sid)
            out.append({
                "id": f"gk:{sid}", "title": str(meta.get("title") or f"Grok · {sid[:8]}")[:80],
                "goal": "", "mode": "agent", "engine": "grok", "source": "grok-cli",
                "external": 1, "archived": 0, "hermes_session_id": None,
                "engine_session_id": sid, "live": True, "writable": True,
                "created_at": meta.get("updated_at") or time.time(),
                "updated_at": meta.get("updated_at") or time.time(),
            })
    except Exception:
        pass
    # --- Codex ---
    try:
        files = sorted(
            CODEX_SESSIONS_DIR.glob("*/*/*/rollout-*.jsonl"),
            key=lambda q: q.stat().st_mtime, reverse=True,
        )[:CLI_SESSION_LIMIT]
        for q in files:
            sid = q.stem.replace("rollout-", "")
            title = f"Codex · {sid[-8:]}"
            try:
                with q.open(encoding="utf-8") as fh:
                    for _ in range(40):
                        line = fh.readline()
                        if not line:
                            break
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        role, text = _codex_message_texts(item.get("payload") or {})
                        if role == "user":
                            title = text.splitlines()[0][:80]
                            break
            except Exception:
                pass
            mtime = q.stat().st_mtime
            out.append({
                "id": f"cx:{sid}", "title": title, "goal": "", "mode": "fast",
                "engine": "codex", "source": "codex-cli", "external": 1, "archived": 0,
                "hermes_session_id": None, "created_at": mtime, "updated_at": mtime,
            })
    except Exception:
        pass
    # --- Antigravity ---
    try:
        dbs = sorted(
            ANTIGRAVITY_CONV_DIR.glob("*.db"),
            key=lambda q: q.stat().st_mtime, reverse=True,
        )[:CLI_SESSION_LIMIT]
        for q in dbs:
            cid = q.stem
            mtime = q.stat().st_mtime
            out.append({
                "id": f"ag:{cid}", "title": f"Antigravity · {cid[:8]}", "goal": "",
                "mode": "fast", "engine": "antigravity", "source": "antigravity",
                "external": 1, "archived": 0, "hermes_session_id": None,
                "created_at": mtime, "updated_at": mtime,
            })
    except Exception:
        pass
    return out


def _cli_session_stub(thread_id: str) -> dict[str, Any]:
    engine = {"gk": "grok", "cx": "codex", "ag": "antigravity"}[thread_id[:2]]
    for item in _list_cli_sessions():
        if item["id"] == thread_id:
            return item
    return {
        "id": thread_id, "title": thread_id, "goal": "", "mode": "fast",
        "engine": engine, "source": f"{engine}-cli", "external": 1, "archived": 0,
        "hermes_session_id": None, "created_at": 0, "updated_at": 0,
    }


def _antigravity_field8_texts(blob: bytes) -> list[str]:
    """Length-delimited field 8 (0x42) strings from a step payload."""
    texts: list[str] = []
    i = 0
    while i < len(blob) - 1:
        if blob[i] == 0x42:
            j = i + 1
            length = 0
            shift = 0
            while j < len(blob):
                b = blob[j]
                length |= (b & 0x7F) << shift
                j += 1
                if not b & 0x80:
                    break
                shift += 7
            if 0 < length <= len(blob) - j:
                try:
                    texts.append(blob[j:j + length].decode("utf-8"))
                    i = j + length
                    continue
                except Exception:
                    pass
        i += 1
    return texts


def _cli_session_turns(thread_id: str) -> list[dict[str, Any]]:
    prefix, key = thread_id[:2], thread_id[3:]
    turns: list[dict[str, Any]] = []
    if prefix == "gk":
        for hist in GROK_SESSIONS_DIR.glob(f"*/{key}/chat_history.jsonl"):
            pending_user = ""
            for line in hist.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                kind = str(msg.get("type") or msg.get("role") or "")
                content = msg.get("content")
                if isinstance(content, list):
                    content = "\n".join(
                        str(c.get("text") or "") for c in content if isinstance(c, dict)
                    )
                text = str(content or "").strip()
                if not text:
                    continue
                if kind == "user":
                    pending_user = text
                elif kind == "assistant":
                    turns.append({"id": f"{thread_id}:{len(turns)}", "mode": "text",
                                  "transcript": pending_user, "reply": text[:6000]})
                    pending_user = ""
            break
    elif prefix == "cx":
        for q in CODEX_SESSIONS_DIR.glob(f"*/*/*/rollout-{key}.jsonl"):
            pending_user = ""
            with q.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        item = json.loads(line)
                    except Exception:
                        continue
                    role, text = _codex_message_texts(item.get("payload") or {})
                    if role == "user":
                        pending_user = text
                    elif role == "assistant":
                        turns.append({"id": f"{thread_id}:{len(turns)}", "mode": "text",
                                      "transcript": pending_user, "reply": text[:6000]})
                        pending_user = ""
            break
    elif prefix == "ag":
        db_path = ANTIGRAVITY_CONV_DIR / f"{key}.db"
        if db_path.exists():
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                rows = conn.execute(
                    "SELECT step_payload FROM steps WHERE step_type=15 AND status=3 ORDER BY rowid"
                ).fetchall()
                conn.close()
                for row in rows:
                    for text in _antigravity_field8_texts(row[0] or b""):
                        if len(text) > 2:
                            turns.append({"id": f"{thread_id}:{len(turns)}", "mode": "text",
                                          "transcript": "", "reply": text[:6000]})
            except Exception:
                pass
    return turns[-60:]


def _list_hermes_sessions(include_archived: bool = False) -> list[dict[str, Any]]:
    """Hermes sessions (allowed sources) as thread-shaped dicts for the web list."""
    out: list[dict[str, Any]] = []
    try:
        conn = _state_ro(); conn.row_factory = sqlite3.Row
        ph = ",".join("?" for _ in SYNC_SOURCES)
        arch = "" if include_archived else "AND COALESCE(archived,0)=0"
        rows = conn.execute(
            f"SELECT id,title,display_name,source,started_at,ended_at,message_count "
            f"FROM sessions WHERE source IN ({ph}) {arch} AND COALESCE(message_count,0)>0 "
            f"ORDER BY COALESCE(ended_at,started_at) DESC LIMIT 80", SYNC_SOURCES
        ).fetchall()
        for r in rows:
            sid = r["id"]
            title = (r["title"] or r["display_name"] or "").strip() or _session_preview(conn, sid) or "Untitled chat"
            ts = int(r["ended_at"] or r["started_at"] or 0)
            out.append({
                "id": "hs:" + sid, "title": title[:80], "hermes_session_id": sid,
                "goal": "", "mode": "fast", "archived": False, "external": True,
                "created_at": int(r["started_at"] or ts), "updated_at": ts, "source": r["source"],
            })
        conn.close()
    except Exception as exc:
        print(f"[sync] list sessions failed: {exc}")
    return out


def _session_as_turns(sid: str) -> list[dict[str, Any]]:
    """A Hermes session's messages mapped to the web app's turn shape."""
    turns: list[dict[str, Any]] = []
    try:
        conn = _state_ro(); conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT role,content,timestamp FROM messages WHERE session_id=? AND active=1 "
            "AND role IN ('user','assistant') ORDER BY timestamp", (sid,)
        ).fetchall()
        conn.close()
        pend = None
        i = 0

        def emit(transcript, reply, at):
            nonlocal i
            turns.append({
                "id": f"{sid}:{i}", "thread_id": "hs:" + sid, "mode": "agent",
                "transcript": transcript, "reply": reply, "latency_seconds": 0,
                "metrics": {}, "interrupted": False, "at": at,
            })
            i += 1

        for r in rows:
            c = _clean_text(_msg_text(r["content"])).strip()
            if not c:
                continue
            t = int(r["timestamp"] or 0)
            if r["role"] == "user":
                if pend is not None:
                    emit(pend[0], "", pend[1])
                pend = (c, t)
            else:
                emit(pend[0] if pend else "", c, t)
                pend = None
        if pend is not None:
            emit(pend[0], "", pend[1])
    except Exception as exc:
        print(f"[sync] session turns failed: {exc}")
    return turns


def _mirror_web_turn(
    thread_id: str,
    mode: str,
    user_text: str,
    reply: str,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Mirror a FAST-mode web turn into the shared Hermes session store so it
    appears in the desktop/terminal apps. Agent-mode already persists via the
    Hermes CLI. Creates the session on first turn (and back-fills any earlier
    local turns so nothing is lost). Best-effort."""
    user_text = (user_text or "").strip()
    reply = (reply or "").strip()
    if not user_text and not reply:
        return
    sdb = _state_store()
    if not sdb:
        return
    try:
        thread = store.get_thread(thread_id)
    except Exception:
        return
    thread_mode = str((metrics or {}).get("thread_mode") or thread.get("mode") or "fast")
    transport = str((metrics or {}).get("transport") or "")
    if thread_mode != "fast" or transport != "stream":
        return
    if thread.get("external"):
        # imported chat: continuing it resumes the real session; the agent CLI
        # path writes those messages, so don't double-write here.
        return
    sid = (thread.get("hermes_session_id") or "").strip()
    try:
        if not sid:
            sid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
            sdb.create_session(sid, source=THREAD_SOURCE, model=SYNC_VOICE_MODEL, cwd=str(DEFAULT_CWD))
            title = (thread.get("title") or user_text[:44] or "Voice chat").strip()
            try:
                sdb.set_session_title(sid, title[:80])
            except Exception:
                pass
            # back-fill prior local turns (chronological) into the new session
            prior = list(store.recent_turns(thread_id, limit=200))
            for pt in sorted(prior, key=lambda x: x.get("at") or 0):
                if pt.get("interrupted"):
                    continue
                u = (pt.get("transcript") or "").strip()
                a = (pt.get("reply") or "").strip()
                if u == user_text and a == reply:
                    continue  # this very turn is appended below
                if u:
                    sdb.append_message(sid, "user", u)
                if a and a != "I processed that, but did not get a response.":
                    sdb.append_message(sid, "assistant", a)
            store.set_hermes_session(thread_id, sid)
        if user_text:
            sdb.append_message(sid, "user", user_text)
        if reply:
            sdb.append_message(sid, "assistant", reply)
    except Exception as exc:
        print(f"[sync] mirror turn failed: {exc}")


def _openai_key_configured() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    for path in (ROOT / ".env.local", ROOT / ".env", DEFAULT_CWD / ".env.local", DEFAULT_CWD / ".env"):
        try:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("OPENAI_API_KEY=") and stripped.partition("=")[2].strip().strip('"').strip("'"):
                    return True
        except OSError:
            continue
    return False


def _thread_row(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    mode = row["mode"] if "mode" in keys else "fast"
    engine = row["engine"] if "engine" in keys else "hermes"
    return {
        "id": row["id"],
        "title": row["title"],
        "hermes_session_id": row["hermes_session_id"],
        "goal": row["goal"],
        "mode": mode if mode in THREAD_MODES else "fast",
        "engine": engine if engine in AGENT_ENGINES else "hermes",
        "model": row["model"] if "model" in keys else None,
        "reasoning_effort": row["reasoning_effort"] if "reasoning_effort" in keys else None,
        "engine_session_id": row["engine_session_id"] if "engine_session_id" in keys else None,
        "archived": bool(row["archived"]),
        "external": bool(row["external"]) if "external" in keys else False,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _turn_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        metrics = json.loads(row["metrics_json"] or "{}")
    except json.JSONDecodeError:
        metrics = {}
    return {
        "id": row["id"],
        "thread_id": row["thread_id"],
        "mode": row["mode"],
        "transcript": row["user_text"],
        "reply": row["reply"],
        "latency_seconds": row["latency_seconds"],
        "metrics": metrics,
        "interrupted": bool(row["interrupted"]),
        "at": row["created_at"],
    }


def _clean_text(text: str) -> str:
    return ANSI_RE.sub("", text).strip()


def _now() -> int:
    return int(time.time())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cert", default=str(ROOT / "certs/voice-room.crt"))
    parser.add_argument("--key", default=str(ROOT / "certs/voice-room.key"))
    args = parser.parse_args()

    import uvicorn

    cert = Path(args.cert)
    key = Path(args.key)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=str(cert) if cert.exists() else None,
        ssl_keyfile=str(key) if key.exists() else None,
        log_level="info",
    )


if __name__ == "__main__":
    main()

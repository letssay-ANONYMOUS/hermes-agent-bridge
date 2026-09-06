#!/usr/bin/env python3
"""Antigravity 2.0 headless adapter for the Hermes Voice Room.

Antigravity (Google's Codex-style agent IDE) exposes an internal CLI,
`agentapi`, that its language_server answers over gRPC. It is fire-and-forget:
`new-conversation` / `send-message` kick off a background "cascade" (agent run)
that streams into the IDE; the reply lands in a per-conversation SQLite store.

This module drives that CLI from outside the IDE:
  * discovers the running language_server's gRPC address + CSRF token,
  * picks the Antigravity project whose folder best matches a working dir,
  * dispatches a prompt (new conversation or resume),
  * polls the conversation's SQLite trajectory until the agent's reply lands,
  * extracts the reply text (protobuf field 8 inside assistant steps).

Everything here is best-effort and self-contained; the server treats a raised
RuntimeError as "engine unavailable / failed" and speaks a hint.

Requires: the Antigravity IDE running (its language_server process alive).
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
import urllib.parse
from pathlib import Path

GEMINI_HOME = Path(os.environ.get("ANTIGRAVITY_HOME", os.path.expanduser("~/.gemini")))
AGENTAPI_BIN = Path(os.environ.get(
    "VOICE_AGENTAPI_BIN", str(GEMINI_HOME / "antigravity/bin/agentapi")))
CONVERSATIONS_DIR = GEMINI_HOME / "antigravity/conversations"
PROJECTS_DIR = GEMINI_HOME / "config/projects"
ANTIGRAVITY_APP = Path(os.environ.get("VOICE_ANTIGRAVITY_APP", "/Applications/Antigravity.app"))

# Assistant-message step in the trajectory SQLite store.
_ASSISTANT_STEP_TYPE = 15
_STATUS_DONE = 3
# protobuf field 8, wire type 2 (length-delimited) → the clean reply text.
_FIELD8 = 0x42


class AntigravityUnavailable(RuntimeError):
    """The Antigravity IDE / language_server is not reachable."""


def is_available() -> bool:
    if not AGENTAPI_BIN.exists():
        return False
    try:
        return _find_language_server() is not None
    except Exception:
        return False


def can_start() -> bool:
    """Whether this Mac has an Antigravity app that the adapter can launch."""
    return AGENTAPI_BIN.exists() and ANTIGRAVITY_APP.exists()


def start(timeout: float = 15.0) -> bool:
    """Launch Antigravity in the background and wait for its language server."""
    if is_available():
        return True
    if not can_start():
        return False
    try:
        subprocess.run(
            ["open", "-gj", str(ANTIGRAVITY_APP)],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_available():
            return True
        time.sleep(0.5)
    return False


def _find_language_server() -> dict[str, str] | None:
    """Return {'pid','csrf','address'} for the running Antigravity LS, or None."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "language_server --standalone"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    pids = [p for p in out.split() if p.strip()]
    for pid in pids:
        try:
            cmd = subprocess.run(["ps", "-o", "command=", "-p", pid],
                                 capture_output=True, text=True, timeout=5).stdout
        except Exception:
            continue
        if "antigravity" not in cmd.lower():
            continue
        m = re.search(r"--csrf_token\s+([0-9a-fA-F-]{16,})", cmd)
        csrf = m.group(1) if m else ""
        address = _grpc_address_for_pid(pid, csrf)
        if csrf and address:
            return {"pid": pid, "csrf": csrf, "address": address}
    return None


def _grpc_address_for_pid(pid: str, csrf: str) -> str:
    """Probe the LS's listening TCP ports; return the one that speaks agentapi gRPC."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-a", "-p", pid],
            capture_output=True, text=True, timeout=6,
        ).stdout
    except Exception:
        return ""
    ports: list[str] = []
    for line in out.splitlines():
        m = re.search(r"127\.0\.0\.1:(\d+)\s+\(LISTEN\)", line)
        if m:
            ports.append(m.group(1))
    # The gRPC port answers agentapi (it complains about CSRF / runs the command);
    # the other port returns a non-gRPC preface error. Probe cheaply.
    for port in ports:
        addr = f"127.0.0.1:{port}"
        env = _cli_env(addr, csrf, project_id="")
        try:
            res = subprocess.run(
                [str(AGENTAPI_BIN), "get-conversation-metadata", "__probe__"],
                capture_output=True, text=True, timeout=8, env=env,
            )
        except Exception:
            continue
        blob = (res.stdout or "") + (res.stderr or "")
        low = blob.lower()
        # gRPC port: recognizes the command (usage/not-found/valid), not "server preface"
        if "server preface" in low or "eof" in low:
            continue
        if any(tok in low for tok in ("usage", "csrf", "conversation", "not found", "project_id")):
            return addr
    return ports[-1] if ports else ""


def _cli_env(address: str, csrf: str, project_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["ANTIGRAVITY_LS_ADDRESS"] = address
    env["ANTIGRAVITY_CSRF_TOKEN"] = csrf
    if project_id:
        env["ANTIGRAVITY_PROJECT_ID"] = project_id
    return env


def _decode_uri(uri: str) -> str:
    if uri.startswith("file://"):
        uri = uri[len("file://"):]
    return urllib.parse.unquote(uri)


def pick_project(cwd: str) -> tuple[str, str]:
    """Return (project_id, project_name) whose folder best matches cwd.

    Prefers an exact / ancestor folder match; else the project literally named
    'Antigravity'; else the first project found.
    """
    cwd = os.path.abspath(cwd)
    best: tuple[int, str, str] | None = None
    fallback_named: tuple[str, str] | None = None
    first: tuple[str, str] | None = None
    for path in sorted(glob.glob(str(PROJECTS_DIR / "*.json"))):
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        pid = str(data.get("id") or "")
        if not pid:
            continue
        name = urllib.parse.unquote(str(data.get("name") or ""))
        if first is None:
            first = (pid, name)
        if name.lower() == "antigravity" and fallback_named is None:
            fallback_named = (pid, name)
        for res in (data.get("projectResources") or {}).get("resources", []):
            folder = res.get("gitFolder", {}).get("folderUri") or res.get("folderUri") or ""
            folder = _decode_uri(folder)
            if not folder:
                continue
            folder_abs = os.path.abspath(folder)
            if cwd == folder_abs or cwd.startswith(folder_abs.rstrip("/") + "/"):
                score = len(folder_abs)  # deepest matching folder wins
                if best is None or score > best[0]:
                    best = (score, pid, name)
    if best:
        return best[1], best[2]
    if fallback_named:
        return fallback_named
    if first:
        return first
    raise AntigravityUnavailable("No Antigravity projects are configured.")


def _extract_reply(payload: bytes) -> str:
    """Pull the assistant reply (protobuf field 8, wire-type 2) from a step blob.

    Concatenates every top-level field-8 string, which is the human-readable
    assistant text; ignores nested render/metadata copies.
    """
    parts: list[str] = []
    i = 0
    n = len(payload)
    while i < n:
        # find next field-8 length-delimited marker
        j = payload.find(bytes([_FIELD8]), i)
        if j < 0 or j + 1 >= n:
            break
        k = j + 1
        length = 0
        shift = 0
        ok = True
        while k < n:
            by = payload[k]
            k += 1
            length |= (by & 0x7F) << shift
            if not (by & 0x80):
                break
            shift += 7
            if shift > 35:
                ok = False
                break
        if not ok or length <= 0 or k + length > n:
            i = j + 1
            continue
        chunk = payload[k:k + length]
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            i = j + 1
            continue
        if text.isprintable() and len(text.strip()) >= 2 and not text.startswith("bot-"):
            parts.append(text.strip())
        i = k + length
    # de-dup consecutive identical fragments (render + content copies)
    cleaned: list[str] = []
    for p in parts:
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)
    return "\n\n".join(cleaned).strip()


def _latest_reply(conversation_id: str, after_mtime: float) -> str | None:
    db = CONVERSATIONS_DIR / f"{conversation_id}.db"
    if not db.exists() or db.stat().st_mtime <= after_mtime:
        return None
    import sqlite3
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
    except Exception:
        return None
    try:
        rows = conn.execute(
            "SELECT idx, step_payload FROM steps WHERE step_type=? AND status=? ORDER BY idx",
            (_ASSISTANT_STEP_TYPE, _STATUS_DONE),
        ).fetchall()
    except Exception:
        return None
    finally:
        conn.close()
    if not rows:
        return None
    # the final assistant step is the reply to the latest prompt
    reply = _extract_reply(rows[-1][1] or b"")
    return reply or None


def dispatch(prompt: str, conversation_id: str | None, cwd: str,
             title: str | None = None) -> str:
    """Start (or continue) an Antigravity conversation. Returns conversation_id."""
    ls = _find_language_server()
    if not ls:
        raise AntigravityUnavailable(
            "Antigravity isn't running — open the Antigravity app on the Mac first.")
    project_id, _name = pick_project(cwd)
    env = _cli_env(ls["address"], ls["csrf"], project_id)

    if conversation_id:
        argv = [str(AGENTAPI_BIN), "send-message", conversation_id, prompt]
    else:
        argv = [str(AGENTAPI_BIN), "new-conversation"]
        if title:
            argv.append(f"--title={title}")
        argv.append(prompt)

    res = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env)
    out = (res.stdout or "").strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"Antigravity CLI returned non-JSON: {(out or res.stderr)[:200]}")
    err = data.get("error")
    if err:
        raise RuntimeError(f"Antigravity: {err}")
    resp = data.get("response") or {}
    cid = (resp.get("newConversation") or {}).get("conversationId") or conversation_id
    if not cid:
        raise RuntimeError("Antigravity did not return a conversation id.")
    return cid


def ask(prompt: str, conversation_id: str | None, cwd: str,
        title: str | None = None, timeout: float = 240.0) -> tuple[str, str]:
    """Dispatch a prompt and block until Antigravity's reply lands.

    Returns (reply_text, conversation_id). Raises on unavailability/timeout.
    """
    db_before = CONVERSATIONS_DIR / f"{conversation_id}.db" if conversation_id else None
    base_mtime = db_before.stat().st_mtime if (db_before and db_before.exists()) else 0.0

    cid = dispatch(prompt, conversation_id, cwd, title=title)

    # A resumed conversation reuses its db; wait for a *newer* assistant step.
    deadline = time.time() + timeout
    poll = 1.0
    last = ""
    stable_since = 0.0
    while time.time() < deadline:
        time.sleep(poll)
        reply = _latest_reply(cid, base_mtime if conversation_id else 0.0)
        if reply:
            # require the reply to stay unchanged briefly (agent finished writing)
            if reply == last:
                if stable_since and time.time() - stable_since >= 1.5:
                    return reply, cid
            else:
                last = reply
                stable_since = time.time()
        poll = min(poll * 1.3, 4.0)
    if last:
        return last, cid
    raise RuntimeError(
        "Antigravity started the task but no reply landed in time — check the Antigravity app.")


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Reply with exactly: adapter works"
    cwd = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        "~/Desktop/home screen folders/MY Business")
    print("available:", is_available())
    t0 = time.time()
    reply, cid = ask(q, None, cwd, title="Voice adapter test")
    print(f"[{time.time()-t0:.1f}s] conversation={cid}")
    print("reply:", reply)

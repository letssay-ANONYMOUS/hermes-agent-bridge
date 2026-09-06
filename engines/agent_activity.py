"""Normalized agent activity model for the Hermes Voice Room UI.

Server and (via JSON over WebSocket) client share this vocabulary so text chat
and voice mode render the same execution state. Events are presentation-only —
they must never carry secrets, full command lines with credentials, or env vars.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class RequestState(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    THINKING = "thinking"
    PLANNING = "planning"
    USING_TOOL = "using_tool"
    STREAMING_RESPONSE = "streaming_response"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskState(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Canonical event types the client reducer accepts.
EVENT_TYPES = frozenset(
    {
        "request_started",
        "status_changed",
        "plan_created",
        "task_started",
        "task_completed",
        "task_failed",
        "tool_started",
        "tool_progress",
        "tool_completed",
        "tool_failed",
        "text_delta",
        "response_completed",
        "response_cancelled",
        "response_failed",
    }
)

# Human labels for status shimmer (never invent chain-of-thought).
STATUS_LABELS = {
    "thinking": "Thinking",
    "planning": "Preparing a plan",
    "investigating_tools": "Investigating tools",
    "searching_web": "Searching the web",
    "reading_files": "Reading project files",
    "running_command": "Running a command",
    "editing": "Editing the implementation",
    "running_tests": "Running tests",
    "verifying": "Verifying the result",
    "retrying": "Retrying the operation",
    "finalizing": "Finalizing the response",
    "streaming": "Writing a response",
    "transcribing": "Transcribing",
    "saving": "Saving to memory",
    "connecting": "Connecting",
}


# Internal tool name / family → clean activity label.
_TOOL_LABELS: Dict[str, str] = {
    # Terminal / shell
    "terminal": "Running a command",
    "run_terminal": "Running a command",
    "run_command": "Running a command",
    "execute_code": "Running code",
    "code_execution": "Running code",
    "process": "Checking a process",
    # Files
    "read_file": "Reading project files",
    "read_files": "Reading project files",
    "write_file": "Editing project files",
    "edit_file": "Editing project files",
    "search_replace": "Editing project files",
    "apply_patch": "Editing project files",
    "list_dir": "Browsing project files",
    "list_directory": "Browsing project files",
    "search_files": "Searching project files",
    "glob": "Searching project files",
    "grep": "Searching project files",
    # Web
    "web_search": "Searching the web",
    "web_extract": "Reading a web page",
    "fetch_url": "Reading a web page",
    "browser": "Investigating the website",
    "browser_navigate": "Investigating the website",
    "browser_click": "Investigating the website",
    "browser_type": "Investigating the website",
    "browser_snapshot": "Investigating the website",
    # Memory / skills
    "memory": "Searching memory",
    "memory_search": "Searching memory",
    "memory_get": "Searching memory",
    "memory_add": "Updating memory",
    "skill": "Loading a skill",
    "skills": "Loading a skill",
    # Tests / deploy
    "test": "Running tests",
    "run_tests": "Running tests",
    "pytest": "Running tests",
    "deploy": "Checking the deployment",
    "vercel": "Checking the deployment",
    # Misc
    "todo": "Updating the plan",
    "todo_write": "Updating the plan",
    "delegate": "Delegating work",
    "subagent": "Delegating work",
    "image": "Inspecting an image",
    "vision": "Inspecting an image",
    "get_current_time": "Checking the time",
    "tell_composer": "Starting a build",
    "check_composer": "Checking the build",
    "mcp": "Using a connected tool",
}


_SECRETISH = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|authorization|bearer|private[_-]?key)"
)
_ENV_ASSIGN = re.compile(r"(?i)\b[A-Z][A-Z0-9_]{2,}=(('[^']*')|(\"[^\"]*\")|\S+)")


def tool_activity_label(tool_name: str | None) -> str:
    """Map an internal tool name to a clean human-readable activity label."""
    if not tool_name:
        return "Using a tool"
    raw = str(tool_name).strip()
    if not raw or raw.startswith("_"):
        return "Working"
    key = raw.lower().replace("-", "_")
    if key in _TOOL_LABELS:
        return _TOOL_LABELS[key]
    # MCP tools often look like server__tool
    if "__" in key:
        leaf = key.rsplit("__", 1)[-1]
        if leaf in _TOOL_LABELS:
            return _TOOL_LABELS[leaf]
        return f"Using {leaf.replace('_', ' ')}"
    # Family match
    for family, label in (
        ("browser", "Investigating the website"),
        ("web", "Searching the web"),
        ("search", "Searching"),
        ("file", "Working with project files"),
        ("terminal", "Running a command"),
        ("shell", "Running a command"),
        ("memory", "Searching memory"),
        ("test", "Running tests"),
        ("deploy", "Checking the deployment"),
        ("git", "Working with git"),
    ):
        if family in key:
            return label
    pretty = re.sub(r"[_\-.]+", " ", raw).strip()
    return f"Using {pretty}" if pretty else "Using a tool"


def status_label_for_tool(tool_name: str | None) -> str:
    label = tool_activity_label(tool_name)
    # Prefer the STATUS_LABELS phrasing when we can map cleanly.
    inverse = {
        "Searching the web": "searching_web",
        "Reading project files": "reading_files",
        "Running a command": "running_command",
        "Editing project files": "editing",
        "Running tests": "running_tests",
        "Investigating the website": "investigating_tools",
    }
    key = inverse.get(label)
    return STATUS_LABELS.get(key, label) if key else label


def safe_preview(value: Any, *, limit: int = 160) -> str:
    """Short, non-secret preview for expandable execution details."""
    if value is None:
        return ""
    if isinstance(value, dict):
        # Drop secret-shaped keys entirely.
        cleaned = {
            k: ("[redacted]" if _SECRETISH.search(str(k)) else v)
            for k, v in value.items()
            if not _SECRETISH.search(str(k))
        }
        text = ", ".join(f"{k}={_short(v)}" for k, v in list(cleaned.items())[:6])
    else:
        text = str(value)
    text = _ENV_ASSIGN.sub(lambda m: m.group(0).split("=", 1)[0] + "=[redacted]", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1[redacted]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _short(value: Any, limit: int = 48) -> str:
    s = str(value)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


@dataclass
class AgentEvent:
    type: str
    request_id: str
    timestamp: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # Optional fields
    status: Optional[str] = None
    label: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    task_id: Optional[str] = None
    title: Optional[str] = None
    detail: Optional[str] = None
    preview: Optional[str] = None
    ok: Optional[bool] = None
    error: Optional[str] = None
    delta: Optional[str] = None
    tasks: Optional[List[Dict[str, Any]]] = None
    seq: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if v is not None}
        return data


def make_event(
    event_type: str,
    request_id: str,
    *,
    seq: Optional[int] = None,
    **fields: Any,
) -> Dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type: {event_type}")
    event = AgentEvent(
        type=event_type,
        request_id=request_id,
        timestamp=time.time(),
        seq=seq,
        **fields,
    )
    return event.to_dict()


def hermes_tool_progress_to_events(
    event_type: str,
    tool_name: str | None,
    preview: str | None,
    args: Any,
    *,
    request_id: str,
    seq: int,
    tool_index: int = 0,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Map Hermes tool_progress_callback args → normalized client events."""
    call_id = str(kwargs.get("tool_call_id") or f"tool_{tool_index}_{tool_name or 'x'}")
    label = tool_activity_label(tool_name)
    safe = safe_preview(preview or args)

    if event_type == "tool.started":
        return [
            make_event(
                "status_changed",
                request_id,
                seq=seq,
                status=RequestState.USING_TOOL.value,
                label=label,
            ),
            make_event(
                "tool_started",
                request_id,
                seq=seq + 1 if seq is not None else None,
                tool_call_id=call_id,
                tool_name=tool_name or "",
                label=label,
                preview=safe,
                title=label,
            ),
            make_event(
                "task_started",
                request_id,
                seq=seq + 2 if seq is not None else None,
                task_id=call_id,
                title=label,
                detail=safe or None,
            ),
        ]

    if event_type in {"tool.completed", "tool.failed"}:
        is_error = bool(kwargs.get("is_error") or event_type == "tool.failed")
        duration = kwargs.get("duration")
        detail = safe
        if duration is not None:
            try:
                detail = (detail + f" · {float(duration):.1f}s").strip(" ·")
            except (TypeError, ValueError):
                pass
        out: List[Dict[str, Any]] = [
            make_event(
                "tool_failed" if is_error else "tool_completed",
                request_id,
                seq=seq,
                tool_call_id=call_id,
                tool_name=tool_name or "",
                label=label,
                ok=not is_error,
                preview=detail or None,
                error=safe_preview(kwargs.get("error")) if is_error else None,
            ),
            make_event(
                "task_failed" if is_error else "task_completed",
                request_id,
                seq=seq + 1 if seq is not None else None,
                task_id=call_id,
                title=label,
                detail=detail or None,
                ok=not is_error,
            ),
        ]
        return out

    if event_type == "tool.progress":
        return [
            make_event(
                "tool_progress",
                request_id,
                seq=seq,
                tool_call_id=call_id,
                tool_name=tool_name or "",
                label=label,
                preview=safe,
            )
        ]

    # reasoning.available is intentionally not forwarded as fabricated CoT.
    return []


def reduce_execution(state: Dict[str, Any] | None, event: Dict[str, Any]) -> Dict[str, Any]:
    """Pure reducer for unit tests and optional server-side mirroring."""
    s = dict(state or _empty_state())
    et = event.get("type")
    rid = event.get("request_id")
    if rid and s.get("request_id") and rid != s["request_id"]:
        # Stale event for a previous request — ignore.
        if et != "request_started":
            return s

    seen = set(s.get("seen_event_ids") or [])
    eid = event.get("id")
    if eid and eid in seen:
        return s
    if eid:
        seen.add(eid)
        s["seen_event_ids"] = list(seen)[-200:]

    if et == "request_started":
        s = _empty_state()
        s["request_id"] = rid
        s["status"] = RequestState.THINKING.value
        s["label"] = event.get("label") or STATUS_LABELS["thinking"]
        s["started_at"] = event.get("timestamp") or time.time()
        return s

    if et == "status_changed":
        s["status"] = event.get("status") or s.get("status")
        if event.get("label"):
            s["label"] = event["label"]
        return s

    if et == "plan_created":
        tasks = event.get("tasks") or []
        s["tasks"] = [
            {
                "id": t.get("id") or f"plan_{i}",
                "title": t.get("title") or f"Step {i + 1}",
                "detail": t.get("detail") or "",
                "state": t.get("state") or TaskState.PENDING.value,
            }
            for i, t in enumerate(tasks)
        ]
        s["status"] = RequestState.PLANNING.value
        s["label"] = event.get("label") or STATUS_LABELS["planning"]
        s["show_progress_card"] = len(s["tasks"]) > 0
        return s

    if et == "task_started":
        s["tasks"] = _upsert_task(
            s.get("tasks") or [],
            event.get("task_id") or uuid.uuid4().hex[:8],
            title=event.get("title") or "Working",
            detail=event.get("detail") or "",
            state=TaskState.ACTIVE.value,
        )
        s["status"] = RequestState.USING_TOOL.value
        s["label"] = event.get("title") or s.get("label")
        s["show_progress_card"] = True
        return s

    if et in {"task_completed", "task_failed"}:
        state_name = TaskState.FAILED.value if et == "task_failed" else TaskState.COMPLETED.value
        s["tasks"] = _upsert_task(
            s.get("tasks") or [],
            event.get("task_id") or "",
            title=event.get("title"),
            detail=event.get("detail"),
            state=state_name,
        )
        return s

    if et == "tool_started":
        tools = list(s.get("tools") or [])
        tools.append(
            {
                "id": event.get("tool_call_id"),
                "name": event.get("tool_name"),
                "label": event.get("label") or tool_activity_label(event.get("tool_name")),
                "preview": event.get("preview") or "",
                "state": "active",
            }
        )
        s["tools"] = tools[-40:]
        s["status"] = RequestState.USING_TOOL.value
        s["label"] = event.get("label") or s.get("label")
        s["show_progress_card"] = True
        return s

    if et in {"tool_completed", "tool_failed"}:
        tools = []
        for t in s.get("tools") or []:
            if t.get("id") == event.get("tool_call_id"):
                t = dict(t)
                t["state"] = "failed" if et == "tool_failed" else "completed"
                if event.get("preview"):
                    t["preview"] = event["preview"]
                if event.get("error"):
                    t["error"] = event["error"]
            tools.append(t)
        s["tools"] = tools
        return s

    if et == "tool_progress":
        tools = []
        for t in s.get("tools") or []:
            if t.get("id") == event.get("tool_call_id"):
                t = dict(t)
                if event.get("preview"):
                    t["preview"] = event["preview"]
            tools.append(t)
        s["tools"] = tools
        return s

    if et == "text_delta":
        s["status"] = RequestState.STREAMING_RESPONSE.value
        s["label"] = STATUS_LABELS["streaming"]
        s["response_text"] = (s.get("response_text") or "") + (event.get("delta") or "")
        s["has_streamed_text"] = True
        return s

    if et == "response_completed":
        s["status"] = RequestState.COMPLETED.value
        s["label"] = None
        s["tasks"] = [
            {**t, "state": TaskState.COMPLETED.value if t.get("state") == TaskState.ACTIVE.value else t.get("state")}
            for t in (s.get("tasks") or [])
        ]
        s["tools"] = [
            {**t, "state": "completed" if t.get("state") == "active" else t.get("state")}
            for t in (s.get("tools") or [])
        ]
        return s

    if et == "response_cancelled":
        s["status"] = RequestState.CANCELLED.value
        s["label"] = "Cancelled"
        s["tasks"] = [
            {
                **t,
                "state": TaskState.CANCELLED.value
                if t.get("state") in {TaskState.ACTIVE.value, TaskState.PENDING.value}
                else t.get("state"),
            }
            for t in (s.get("tasks") or [])
        ]
        return s

    if et == "response_failed":
        s["status"] = RequestState.FAILED.value
        s["label"] = "Failed"
        s["error"] = event.get("error") or "Request failed"
        return s

    return s


def _empty_state() -> Dict[str, Any]:
    return {
        "request_id": None,
        "status": RequestState.IDLE.value,
        "label": None,
        "tasks": [],
        "tools": [],
        "response_text": "",
        "has_streamed_text": False,
        "show_progress_card": False,
        "error": None,
        "seen_event_ids": [],
        "started_at": None,
    }


def _upsert_task(
    tasks: List[Dict[str, Any]],
    task_id: str,
    *,
    title: Optional[str],
    detail: Optional[str],
    state: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    found = False
    for t in tasks:
        if t.get("id") == task_id:
            found = True
            nt = dict(t)
            if title:
                nt["title"] = title
            if detail is not None:
                nt["detail"] = detail
            nt["state"] = state
            # Only one dominant active task visually, but data allows parallel.
            out.append(nt)
        else:
            nt = dict(t)
            if state == TaskState.ACTIVE.value and nt.get("state") == TaskState.ACTIVE.value:
                # Keep parallel actives if different tools; leave as-is.
                pass
            out.append(nt)
    if not found and task_id:
        out.append(
            {
                "id": task_id,
                "title": title or "Working",
                "detail": detail or "",
                "state": state,
            }
        )
    return out


def mock_stream_scenarios() -> Dict[str, List[Dict[str, Any]]]:
    """Deterministic fixture streams for UI/dev verification."""
    rid = "req_mock"

    def ev(i: int, typ: str, **kw: Any) -> Dict[str, Any]:
        return make_event(typ, rid, seq=i, **kw)

    return {
        "simple_thinking": [
            ev(1, "request_started", label=STATUS_LABELS["thinking"]),
            ev(2, "status_changed", status="thinking", label=STATUS_LABELS["thinking"]),
            ev(3, "text_delta", delta="Hello — "),
            ev(4, "text_delta", delta="here's a quick answer."),
            ev(5, "response_completed"),
        ],
        "web_search": [
            ev(1, "request_started", label=STATUS_LABELS["thinking"]),
            ev(2, "tool_started", tool_call_id="c1", tool_name="web_search", label="Searching the web", preview="query=hermes agent"),
            ev(3, "task_started", task_id="c1", title="Searching the web"),
            ev(4, "tool_completed", tool_call_id="c1", tool_name="web_search", ok=True, preview="3 results"),
            ev(5, "task_completed", task_id="c1", title="Searching the web"),
            ev(6, "text_delta", delta="I found a few relevant sources."),
            ev(7, "response_completed"),
        ],
        "three_step": [
            ev(1, "request_started", label=STATUS_LABELS["thinking"]),
            ev(2, "plan_created", tasks=[
                {"id": "s1", "title": "Inspect the existing implementation"},
                {"id": "s2", "title": "Build the Hermes streaming bridge"},
                {"id": "s3", "title": "Connect the activity interface"},
            ]),
            ev(3, "task_started", task_id="s1", title="Inspect the existing implementation"),
            ev(4, "tool_started", tool_call_id="t1", tool_name="read_file", label="Reading project files"),
            ev(5, "tool_completed", tool_call_id="t1", tool_name="read_file", ok=True),
            ev(6, "task_completed", task_id="s1", title="Inspect the existing implementation"),
            ev(7, "task_started", task_id="s2", title="Build the Hermes streaming bridge"),
            ev(8, "tool_started", tool_call_id="t2", tool_name="edit_file", label="Editing project files"),
            ev(9, "tool_completed", tool_call_id="t2", tool_name="edit_file", ok=True),
            ev(10, "task_completed", task_id="s2", title="Build the Hermes streaming bridge"),
            ev(11, "task_started", task_id="s3", title="Connect the activity interface"),
            ev(12, "task_completed", task_id="s3", title="Connect the activity interface"),
            ev(13, "text_delta", delta="All three steps are done."),
            ev(14, "response_completed"),
        ],
        "parallel_tools": [
            ev(1, "request_started", label=STATUS_LABELS["thinking"]),
            ev(2, "tool_started", tool_call_id="p1", tool_name="web_search", label="Searching the web"),
            ev(3, "tool_started", tool_call_id="p2", tool_name="read_file", label="Reading project files"),
            ev(4, "task_started", task_id="p1", title="Searching the web"),
            ev(5, "task_started", task_id="p2", title="Reading project files"),
            ev(6, "tool_completed", tool_call_id="p1", tool_name="web_search", ok=True),
            ev(7, "task_completed", task_id="p1", title="Searching the web"),
            ev(8, "tool_completed", tool_call_id="p2", tool_name="read_file", ok=True),
            ev(9, "task_completed", task_id="p2", title="Reading project files"),
            ev(10, "text_delta", delta="Combined both results."),
            ev(11, "response_completed"),
        ],
        "failed_retry": [
            ev(1, "request_started", label=STATUS_LABELS["thinking"]),
            ev(2, "tool_started", tool_call_id="f1", tool_name="terminal", label="Running a command"),
            ev(3, "task_started", task_id="f1", title="Running a command"),
            ev(4, "tool_failed", tool_call_id="f1", tool_name="terminal", ok=False, error="exit 1"),
            ev(5, "task_failed", task_id="f1", title="Running a command", detail="exit 1"),
            ev(6, "status_changed", status="thinking", label=STATUS_LABELS["retrying"]),
            ev(7, "tool_started", tool_call_id="f2", tool_name="terminal", label="Running a command"),
            ev(8, "task_started", task_id="f2", title="Running a command"),
            ev(9, "tool_completed", tool_call_id="f2", tool_name="terminal", ok=True),
            ev(10, "task_completed", task_id="f2", title="Running a command"),
            ev(11, "text_delta", delta="Retry succeeded."),
            ev(12, "response_completed"),
        ],
        "cancelled": [
            ev(1, "request_started", label=STATUS_LABELS["thinking"]),
            ev(2, "tool_started", tool_call_id="x1", tool_name="web_search", label="Searching the web"),
            ev(3, "response_cancelled"),
        ],
        "disconnect_recovery": [
            ev(1, "request_started", label=STATUS_LABELS["thinking"]),
            ev(2, "status_changed", status="connecting", label=STATUS_LABELS["connecting"]),
            ev(3, "status_changed", status="thinking", label=STATUS_LABELS["thinking"]),
            ev(4, "text_delta", delta="Reconnected and finished."),
            ev(5, "response_completed"),
        ],
    }


def apply_events(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    state: Dict[str, Any] | None = None
    for event in events:
        state = reduce_execution(state, event)
    return state or _empty_state()

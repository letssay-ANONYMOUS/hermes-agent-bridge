"""Unit tests for the normalized agent activity model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.agent_activity import (
    apply_events,
    hermes_tool_progress_to_events,
    make_event,
    mock_stream_scenarios,
    reduce_execution,
    safe_preview,
    tool_activity_label,
)


def test_tool_labels():
    assert tool_activity_label("web_search") == "Searching the web"
    assert tool_activity_label("browser_navigate") == "Investigating the website"
    assert tool_activity_label("terminal") == "Running a command"
    assert tool_activity_label("read_file") == "Reading project files"
    assert "unknown" in tool_activity_label("totally_unknown_tool").lower() or "Using" in tool_activity_label(
        "totally_unknown_tool"
    )


def test_safe_preview_redacts_secrets():
    text = safe_preview({"api_key": "sk-secret", "path": "/tmp/x"})
    assert "sk-secret" not in text
    assert "[redacted]" in text
    assert "path" in text


def test_status_transitions_and_streaming():
    rid = "r1"
    events = [
        make_event("request_started", rid, label="Thinking"),
        make_event("status_changed", rid, status="thinking", label="Thinking"),
        make_event("text_delta", rid, delta="Hello"),
        make_event("text_delta", rid, delta=" world"),
        make_event("response_completed", rid),
    ]
    state = apply_events(events)
    assert state["status"] == "completed"
    assert state["response_text"] == "Hello world"
    assert state["has_streamed_text"] is True
    assert state["label"] is None


def test_tool_start_complete_and_multi_step():
    rid = "r2"
    events = [
        make_event("request_started", rid, label="Thinking"),
        make_event("tool_started", rid, tool_call_id="t1", tool_name="web_search", label="Searching the web"),
        make_event("task_started", rid, task_id="t1", title="Searching the web"),
        make_event("tool_completed", rid, tool_call_id="t1", tool_name="web_search", ok=True),
        make_event("task_completed", rid, task_id="t1", title="Searching the web"),
        make_event("text_delta", rid, delta="Done."),
        make_event("response_completed", rid),
    ]
    state = apply_events(events)
    assert len(state["tools"]) == 1
    assert state["tools"][0]["state"] == "completed"
    assert state["tasks"][0]["state"] == "completed"


def test_duplicate_events_ignored():
    ev = make_event("request_started", "r3", label="Thinking")
    s1 = reduce_execution(None, ev)
    s2 = reduce_execution(s1, ev)
    assert s1["request_id"] == s2["request_id"]
    assert s1["status"] == s2["status"]


def test_stale_request_ignored():
    s = reduce_execution(None, make_event("request_started", "old", label="Thinking"))
    s = reduce_execution(s, make_event("text_delta", "new", delta="nope"))
    assert s["response_text"] == ""


def test_failed_tool_and_cancel():
    rid = "r4"
    state = apply_events(
        [
            make_event("request_started", rid, label="Thinking"),
            make_event("tool_started", rid, tool_call_id="x", tool_name="terminal", label="Running a command"),
            make_event("tool_failed", rid, tool_call_id="x", tool_name="terminal", ok=False, error="boom"),
            make_event("response_cancelled", rid),
        ]
    )
    assert state["status"] == "cancelled"
    assert state["tools"][0]["state"] == "failed"


def test_hermes_tool_progress_mapping():
    events = hermes_tool_progress_to_events(
        "tool.started",
        "web_search",
        "q=test",
        {"query": "test"},
        request_id="r5",
        seq=1,
        tool_index=1,
    )
    types = [e["type"] for e in events]
    assert "tool_started" in types
    assert "task_started" in types
    assert all(e["request_id"] == "r5" for e in events)


def test_mock_fixtures_complete():
    scenarios = mock_stream_scenarios()
    required = {
        "simple_thinking",
        "web_search",
        "three_step",
        "parallel_tools",
        "failed_retry",
        "cancelled",
        "disconnect_recovery",
    }
    assert required.issubset(scenarios.keys())
    three = apply_events(scenarios["three_step"])
    assert three["status"] == "completed"
    assert len(three["tasks"]) >= 3
    cancelled = apply_events(scenarios["cancelled"])
    assert cancelled["status"] == "cancelled"


if __name__ == "__main__":
    test_tool_labels()
    test_safe_preview_redacts_secrets()
    test_status_transitions_and_streaming()
    test_tool_start_complete_and_multi_step()
    test_duplicate_events_ignored()
    test_stale_request_ignored()
    test_failed_tool_and_cancel()
    test_hermes_tool_progress_mapping()
    test_mock_fixtures_complete()
    print("all ok")

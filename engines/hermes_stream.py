"""Stream full Hermes agent turns with structured tool progress.

Runs AIAgent in-process (same machine as voice room) so tool_progress and
stream deltas reach the WebSocket without scraping CLI logs. Provider API keys
stay in Hermes config / process env — never sent to the browser.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from engines.agent_activity import (
    STATUS_LABELS,
    hermes_tool_progress_to_events,
    make_event,
    safe_preview,
)

logger = logging.getLogger("hermes_voice_room.hermes_stream")

SendFn = Callable[[Dict[str, Any]], Awaitable[None]]


def _ensure_hermes_path(agent_dir: Path) -> None:
    path = str(agent_dir)
    if path not in sys.path:
        sys.path.insert(0, path)


async def stream_hermes_agent_turn(
    *,
    agent_dir: Path,
    user_text: str,
    session_id: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    ephemeral_system_prompt: Optional[str] = None,
    cancel_event: Optional[asyncio.Event] = None,
    send: SendFn,
    speak_sentence: Optional[Callable[[str], Awaitable[None]]] = None,
    max_iterations: int = 30,
    cwd: Optional[str] = None,
) -> tuple[str, bool, Dict[str, Any]]:
    """Run one agent turn, emitting normalized execution events via ``send``.

    Returns (final_text, interrupted, metrics).
    """
    _ensure_hermes_path(agent_dir)
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    metrics: Dict[str, Any] = {
        "transport": "hermes_agent_stream",
        "request_id": request_id,
    }
    started = time.time()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
    seq = 0
    tool_index = 0
    reply_parts: List[str] = []
    interrupted = False
    cancel = cancel_event or asyncio.Event()

    def _next_seq() -> int:
        nonlocal seq
        seq += 1
        return seq

    def _push(event: Dict[str, Any]) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception:
            pass

    def _on_delta(text: Optional[str]) -> None:
        if not text:
            return
        reply_parts.append(text)
        _push(
            make_event(
                "text_delta",
                request_id,
                seq=_next_seq(),
                delta=text,
                status="streaming_response",
                label=STATUS_LABELS["streaming"],
            )
        )

    def _on_tool_progress(
        event_type: str,
        name: str | None = None,
        preview: str | None = None,
        args: Any = None,
        **kwargs: Any,
    ) -> None:
        nonlocal tool_index
        if event_type == "tool.started":
            tool_index += 1
        events = hermes_tool_progress_to_events(
            event_type,
            name,
            preview,
            args,
            request_id=request_id,
            seq=_next_seq(),
            tool_index=tool_index,
            **kwargs,
        )
        for event in events:
            _push(event)

    # Kick off request lifecycle for the UI immediately.
    await send(
        {
            "type": "execution",
            "event": make_event(
                "request_started",
                request_id,
                seq=_next_seq(),
                label=STATUS_LABELS["thinking"],
                status="thinking",
            ),
        }
    )
    await send({"type": "state", "state": "thinking"})
    await send(
        {
            "type": "execution",
            "event": make_event(
                "status_changed",
                request_id,
                seq=_next_seq(),
                status="thinking",
                label=STATUS_LABELS["thinking"],
            ),
        }
    )

    def _run_sync() -> Dict[str, Any]:
        from run_agent import AIAgent
        from gateway.run import (
            GatewayRunner,
            _resolve_gateway_model,
            _resolve_runtime_agent_kwargs,
        )
        from hermes_cli.tools_config import _get_platform_tools
        from hermes_cli.config import load_config

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        runtime_model = runtime_kwargs.pop("model", None)
        model = runtime_model or _resolve_gateway_model()
        reasoning_config = GatewayRunner._load_reasoning_config()

        try:
            user_config = load_config()
        except Exception:
            user_config = {}

        # Prefer CLI / voice-room toolsets; fall back to api_server platform.
        try:
            enabled = sorted(_get_platform_tools(user_config, "cli") or [])
        except Exception:
            enabled = []
        if not enabled:
            try:
                enabled = sorted(_get_platform_tools(user_config, "api_server") or [])
            except Exception:
                enabled = None

        if cwd:
            os.environ.setdefault("HERMES_VOICE_CWD", cwd)

        agent = AIAgent(
            model=model,
            max_iterations=max_iterations,
            enabled_toolsets=enabled,
            quiet_mode=True,
            tool_progress_mode="all",
            ephemeral_system_prompt=ephemeral_system_prompt,
            session_id=session_id or request_id,
            stream_delta_callback=_on_delta,
            tool_progress_callback=_on_tool_progress,
            reasoning_config=reasoning_config,
            platform="voice_room",
            skip_context_files=False,
            **runtime_kwargs,
        )

        history = list(conversation_history or [])
        result = agent.run_conversation(
            user_message=user_text,
            conversation_history=history,
            task_id=session_id or request_id,
        )
        return result if isinstance(result, dict) else {"final_response": str(result or "")}

    agent_task = asyncio.create_task(asyncio.to_thread(_run_sync))
    sentence_buf = ""
    first_token_at: Optional[float] = None

    async def _drain_until(done_task: asyncio.Task[Any]) -> None:
        nonlocal interrupted, sentence_buf, first_token_at
        while True:
            if cancel.is_set():
                interrupted = True
                # Best-effort: agent.interrupt if the object is still running
                # is handled when the task exits; we just stop emitting.
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.15)
            except asyncio.TimeoutError:
                if done_task.done() and queue.empty():
                    break
                continue
            if item is None:
                break
            et = item.get("type")
            if et == "text_delta":
                if first_token_at is None:
                    first_token_at = time.time()
                    metrics["first_token_seconds"] = round(first_token_at - started, 3)
                # Client activity system + live bubble
                await send({"type": "execution", "event": item})
                delta = item.get("delta") or ""
                await send({"type": "token", "text": delta})
                if speak_sentence:
                    sentence_buf += delta
                    # lightweight sentence flush
                    while True:
                        cut = -1
                        for sep in (". ", "! ", "? ", "\n"):
                            idx = sentence_buf.find(sep)
                            if idx != -1 and (cut == -1 or idx < cut):
                                cut = idx + len(sep)
                        if cut == -1 or cut < 24:
                            break
                        chunk = sentence_buf[:cut].strip()
                        sentence_buf = sentence_buf[cut:]
                        if chunk:
                            await speak_sentence(chunk)
            else:
                await send({"type": "execution", "event": item})
                # Also mirror activity chip for older UI paths
                if et in {"tool_started", "status_changed"} and item.get("label"):
                    await send(
                        {
                            "type": "activity",
                            "engine": "hermes",
                            "label": item.get("label"),
                            "tool": item.get("tool_name"),
                            "tool_call_id": item.get("tool_call_id"),
                        }
                    )

    drain_task = asyncio.create_task(_drain_until(agent_task))

    result: Dict[str, Any] = {}
    error_text: Optional[str] = None
    try:
        while not agent_task.done():
            if cancel.is_set():
                interrupted = True
                # Let the agent finish or timeout naturally; interrupt is
                # process-level via runtime elsewhere for CLI. For in-process
                # we mark cancelled and stop waiting after a short grace.
                try:
                    result = await asyncio.wait_for(asyncio.shield(agent_task), timeout=0.5)
                except (asyncio.TimeoutError, Exception):
                    agent_task.cancel()
                    break
                break
            await asyncio.sleep(0.05)
        if not agent_task.done():
            try:
                result = await agent_task
            except Exception as exc:
                error_text = str(exc)
        else:
            try:
                result = agent_task.result()
            except Exception as exc:
                error_text = str(exc)
    except Exception as exc:
        error_text = str(exc)
        logger.exception("Hermes agent stream failed")
    finally:
        try:
            queue.put_nowait(None)
        except Exception:
            pass
        try:
            await asyncio.wait_for(drain_task, timeout=2.0)
        except Exception:
            drain_task.cancel()

    if speak_sentence and sentence_buf.strip() and not interrupted:
        try:
            await speak_sentence(sentence_buf.strip())
        except Exception:
            pass
        sentence_buf = ""

    final = ""
    if isinstance(result, dict):
        final = str(result.get("final_response") or "").strip()
        if result.get("failed"):
            error_text = safe_preview(result.get("error") or "agent run failed", limit=400)
    if not final and reply_parts:
        final = "".join(reply_parts).strip()

    metrics["hermes_seconds"] = round(time.time() - started, 3)
    metrics["tool_calls"] = tool_index

    if interrupted or cancel.is_set():
        await send(
            {
                "type": "execution",
                "event": make_event("response_cancelled", request_id, seq=_next_seq()),
            }
        )
        return final, True, metrics

    if error_text and not final:
        await send(
            {
                "type": "execution",
                "event": make_event(
                    "response_failed",
                    request_id,
                    seq=_next_seq(),
                    error=safe_preview(error_text, limit=400),
                ),
            }
        )
        raise RuntimeError(error_text)

    if final and not reply_parts:
        # Agent returned a final answer without streaming deltas.
        await send(
            {
                "type": "execution",
                "event": make_event(
                    "text_delta",
                    request_id,
                    seq=_next_seq(),
                    delta=final,
                ),
            }
        )
        await send({"type": "token", "text": final})

    await send(
        {
            "type": "execution",
            "event": make_event("response_completed", request_id, seq=_next_seq()),
        }
    )
    return final, False, metrics

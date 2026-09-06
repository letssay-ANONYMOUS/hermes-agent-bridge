/**
 * Hermes Voice Room — agent activity / progress / tool-feedback system.
 *
 * Shared by text chat and voice mode. Consumes normalized `execution` events
 * (and legacy `activity` / `token` / `state` messages) into one reducer-driven
 * store, then renders:
 *   - AgentStatusShimmer
 *   - AgentActivityLabel
 *   - AgentProgressCard / AgentTaskStepper / AgentTaskItem
 *   - ToolActivityItem
 *   - ExecutionDetails
 *   - StreamingMessage helpers
 *   - VoiceTaskOverlay (mirrors progress into the call UI)
 *
 * Secrets never appear here — the server redacts before emit.
 */
(function (global) {
  'use strict';

  const STATUS = {
    idle: 'idle',
    connecting: 'connecting',
    thinking: 'thinking',
    planning: 'planning',
    using_tool: 'using_tool',
    streaming_response: 'streaming_response',
    completed: 'completed',
    failed: 'failed',
    cancelled: 'cancelled',
  };

  const TASK = {
    pending: 'pending',
    active: 'active',
    completed: 'completed',
    failed: 'failed',
    cancelled: 'cancelled',
  };

  const TOOL_LABELS = {
    terminal: 'Running a command',
    run_terminal: 'Running a command',
    run_command: 'Running a command',
    execute_code: 'Running code',
    code_execution: 'Running code',
    read_file: 'Reading project files',
    write_file: 'Editing project files',
    edit_file: 'Editing project files',
    search_replace: 'Editing project files',
    list_dir: 'Browsing project files',
    list_directory: 'Browsing project files',
    search_files: 'Searching project files',
    web_search: 'Searching the web',
    fetch_url: 'Reading a web page',
    browser: 'Investigating the website',
    browser_navigate: 'Investigating the website',
    memory: 'Searching memory',
    memory_search: 'Searching memory',
    test: 'Running tests',
    run_tests: 'Running tests',
    deploy: 'Checking the deployment',
    todo: 'Updating the plan',
    todo_write: 'Updating the plan',
    delegate: 'Delegating work',
    tell_composer: 'Starting a build',
    check_composer: 'Checking the build',
    get_current_time: 'Checking the time',
  };

  function toolActivityLabel(name) {
    if (!name) return 'Using a tool';
    const key = String(name).toLowerCase().replace(/-/g, '_');
    if (TOOL_LABELS[key]) return TOOL_LABELS[key];
    if (key.includes('browser')) return 'Investigating the website';
    if (key.includes('web') || key.includes('search')) return 'Searching the web';
    if (key.includes('file')) return 'Working with project files';
    if (key.includes('terminal') || key.includes('shell')) return 'Running a command';
    if (key.includes('memory')) return 'Searching memory';
    if (key.includes('test')) return 'Running tests';
    return `Using ${key.replace(/[_\-.]+/g, ' ').trim() || 'a tool'}`;
  }

  function emptyState() {
    return {
      requestId: null,
      status: STATUS.idle,
      label: null,
      tasks: [],
      tools: [],
      responseText: '',
      hasStreamedText: false,
      showProgressCard: false,
      error: null,
      seenEventIds: new Set(),
      startedAt: null,
      generation: 0,
    };
  }

  function upsertTask(tasks, id, patch) {
    const out = [];
    let found = false;
    for (const t of tasks) {
      if (t.id === id) {
        found = true;
        out.push({ ...t, ...patch, id });
      } else {
        out.push(t);
      }
    }
    if (!found && id) {
      out.push({
        id,
        title: patch.title || 'Working',
        detail: patch.detail || '',
        state: patch.state || TASK.pending,
      });
    }
    return out;
  }

  function reduceExecution(state, event) {
    if (!event || !event.type) return state;
    const s = {
      ...state,
      tasks: [...(state.tasks || [])],
      tools: [...(state.tools || [])],
      seenEventIds: new Set(state.seenEventIds || []),
    };

    if (event.id) {
      if (s.seenEventIds.has(event.id)) return state;
      s.seenEventIds.add(event.id);
      if (s.seenEventIds.size > 200) {
        s.seenEventIds = new Set([...s.seenEventIds].slice(-120));
      }
    }

    const rid = event.request_id;
    if (rid && s.requestId && rid !== s.requestId && event.type !== 'request_started') {
      return state; // stale
    }

    switch (event.type) {
      case 'request_started':
        return {
          ...emptyState(),
          requestId: rid,
          status: STATUS.thinking,
          label: event.label || 'Thinking',
          startedAt: event.timestamp ? event.timestamp * 1000 : performance.now(),
          generation: (state.generation || 0) + 1,
          seenEventIds: event.id ? new Set([event.id]) : new Set(),
        };
      case 'status_changed':
        s.status = event.status || s.status;
        if (event.label) s.label = event.label;
        return s;
      case 'plan_created':
        s.tasks = (event.tasks || []).map((t, i) => ({
          id: t.id || `plan_${i}`,
          title: t.title || `Step ${i + 1}`,
          detail: t.detail || '',
          state: t.state || TASK.pending,
        }));
        s.status = STATUS.planning;
        s.label = event.label || 'Preparing a plan';
        s.showProgressCard = s.tasks.length > 0;
        return s;
      case 'task_started':
        s.tasks = upsertTask(s.tasks, event.task_id || event.tool_call_id, {
          title: event.title || event.label || 'Working',
          detail: event.detail || event.preview || '',
          state: TASK.active,
        });
        s.status = STATUS.using_tool;
        s.label = event.title || event.label || s.label;
        s.showProgressCard = true;
        return s;
      case 'task_completed':
      case 'task_failed':
        s.tasks = upsertTask(s.tasks, event.task_id || event.tool_call_id, {
          title: event.title,
          detail: event.detail || event.preview || '',
          state: event.type === 'task_failed' ? TASK.failed : TASK.completed,
        });
        return s;
      case 'tool_started': {
        const label = event.label || toolActivityLabel(event.tool_name);
        s.tools = [
          ...s.tools,
          {
            id: event.tool_call_id,
            name: event.tool_name,
            label,
            preview: event.preview || '',
            state: 'active',
          },
        ].slice(-40);
        s.status = STATUS.using_tool;
        s.label = label;
        s.showProgressCard = true;
        // Correlate tool → task when no explicit plan exists
        if (event.tool_call_id && !s.tasks.some((t) => t.id === event.tool_call_id)) {
          s.tasks = upsertTask(s.tasks, event.tool_call_id, {
            title: label,
            detail: event.preview || '',
            state: TASK.active,
          });
        }
        return s;
      }
      case 'tool_completed':
      case 'tool_failed': {
        s.tools = s.tools.map((t) =>
          t.id === event.tool_call_id
            ? {
                ...t,
                state: event.type === 'tool_failed' ? 'failed' : 'completed',
                preview: event.preview || t.preview,
                error: event.error || t.error,
              }
            : t
        );
        if (event.tool_call_id) {
          s.tasks = upsertTask(s.tasks, event.tool_call_id, {
            state: event.type === 'tool_failed' ? TASK.failed : TASK.completed,
            detail: event.preview || event.error || '',
          });
        }
        return s;
      }
      case 'tool_progress':
        s.tools = s.tools.map((t) =>
          t.id === event.tool_call_id
            ? { ...t, preview: event.preview || t.preview }
            : t
        );
        return s;
      case 'text_delta':
        s.status = STATUS.streaming_response;
        s.label = s.hasStreamedText ? null : null; // hide thinking shimmer once text flows
        s.responseText = (s.responseText || '') + (event.delta || '');
        s.hasStreamedText = true;
        return s;
      case 'response_completed':
        s.status = STATUS.completed;
        s.label = null;
        s.tasks = s.tasks.map((t) =>
          t.state === TASK.active ? { ...t, state: TASK.completed } : t
        );
        s.tools = s.tools.map((t) =>
          t.state === 'active' ? { ...t, state: 'completed' } : t
        );
        return s;
      case 'response_cancelled':
        s.status = STATUS.cancelled;
        s.label = 'Cancelled';
        s.tasks = s.tasks.map((t) =>
          t.state === TASK.active || t.state === TASK.pending
            ? { ...t, state: TASK.cancelled }
            : t
        );
        return s;
      case 'response_failed':
        s.status = STATUS.failed;
        s.label = 'Failed';
        s.error = event.error || 'Request failed';
        return s;
      default:
        return state;
    }
  }

  function prefersReducedMotion() {
    try {
      return (
        document.documentElement.dataset.motion === 'reduced' ||
        window.matchMedia('(prefers-reduced-motion: reduce)').matches
      );
    } catch (_) {
      return false;
    }
  }

  /**
   * Cross-fading shimmer status label.
   *
   * Important: enter/exit motion lives on a *wrapper* row. The shimmer animation
   * lives only on the inner text node. Putting both on one element makes the
   * enter keyframes replace `thinking-word-sweep` and kills the shimmer.
   */
  class AgentStatusShimmer {
    constructor(host) {
      this.host = host;
      this.el = document.createElement('div');
      this.el.className = 'aa-status-shimmer';
      this.el.setAttribute('aria-live', 'polite');
      this.el.setAttribute('role', 'status');
      this.row = this._makeRow('');
      this.el.appendChild(this.row);
      host.appendChild(this.el);
      this._label = '';
      this._active = false;
      this._exitTimer = null;
    }

    _makeRow(text, { enter = false, shimmer = true } = {}) {
      const row = document.createElement('span');
      row.className = 'aa-status-row' + (enter ? ' aa-status-enter' : '');
      const textEl = document.createElement('span');
      textEl.className = shimmer && !prefersReducedMotion() ? 'aa-shimmer-text' : 'aa-status-static';
      textEl.textContent = text;
      row.appendChild(textEl);
      return row;
    }

    setLabel(label, { animate = true } = {}) {
      const next = (label || '').trim();
      if (!next) {
        this.stop();
        return;
      }
      if (next === this._label && this._active) {
        // Re-assert shimmer class if something stripped it.
        const textEl = this.row?.querySelector('.aa-shimmer-text, .aa-status-static');
        if (textEl && !prefersReducedMotion() && !textEl.classList.contains('aa-shimmer-text')) {
          textEl.className = 'aa-shimmer-text';
        }
        this.el.hidden = false;
        return;
      }

      const reduced = prefersReducedMotion() || !animate;
      const shimmer = !prefersReducedMotion();

      if (!this._active || reduced || !this._label) {
        if (this._exitTimer) {
          clearTimeout(this._exitTimer);
          this._exitTimer = null;
        }
        this.el.innerHTML = '';
        this.row = this._makeRow(next, { shimmer });
        this.el.appendChild(this.row);
        this.el.hidden = false;
        this._label = next;
        this._active = true;
        return;
      }

      // Cross-fade: previous row exits up, new row enters from below.
      const outgoing = this.row;
      outgoing.classList.remove('aa-status-enter');
      outgoing.classList.add('aa-status-exit');
      const incoming = this._makeRow(next, { enter: true, shimmer });
      this.el.appendChild(incoming);
      this.row = incoming;
      this._label = next;
      this._active = true;
      this.el.hidden = false;
      if (this._exitTimer) clearTimeout(this._exitTimer);
      this._exitTimer = setTimeout(() => {
        outgoing.remove();
        this._exitTimer = null;
      }, 240);
    }

    stop() {
      this._active = false;
      this._label = '';
      this.el.hidden = true;
      if (this._exitTimer) {
        clearTimeout(this._exitTimer);
        this._exitTimer = null;
      }
      const textEl = this.row?.querySelector('.aa-shimmer-text, .aa-status-static');
      if (textEl) textEl.textContent = '';
    }

    destroy() {
      if (this._exitTimer) clearTimeout(this._exitTimer);
      this.el.remove();
    }
  }

  function taskStateLabel(state) {
    switch (state) {
      case TASK.active:
        return 'active';
      case TASK.completed:
        return 'completed';
      case TASK.failed:
        return 'failed';
      case TASK.cancelled:
        return 'cancelled';
      default:
        return 'pending';
    }
  }

  class AgentProgressCard {
    constructor(host, { onInterrupt, onToggleDetails } = {}) {
      this.host = host;
      this.onInterrupt = onInterrupt;
      this.onToggleDetails = onToggleDetails;
      this.expanded = false;
      this.el = document.createElement('div');
      this.el.className = 'aa-progress-card work-status-card';
      this.el.innerHTML = `
        <div class="work-status-head">
          <div>
            <div class="work-status-kicker">Live activity</div>
            <div class="work-status-title aa-card-title"></div>
          </div>
          <div class="work-status-pill aa-card-pill">thinking</div>
        </div>
        <ol class="work-status-steps aa-task-list" aria-label="Task progress"></ol>
        <div class="aa-tools" hidden></div>
        <div class="aa-details" hidden></div>
        <div class="work-status-foot">
          <button type="button" class="aa-details-btn">Details</button>
          <span class="aa-elapsed">0s</span>
          <button type="button" class="work-stop aa-interrupt">Interrupt</button>
        </div>
      `;
      host.appendChild(this.el);
      this.titleEl = this.el.querySelector('.aa-card-title');
      this.pillEl = this.el.querySelector('.aa-card-pill');
      this.listEl = this.el.querySelector('.aa-task-list');
      this.toolsEl = this.el.querySelector('.aa-tools');
      this.detailsEl = this.el.querySelector('.aa-details');
      this.elapsedEl = this.el.querySelector('.aa-elapsed');
      this.el.querySelector('.aa-interrupt').addEventListener('click', () => {
        this.onInterrupt?.();
      });
      this.el.querySelector('.aa-details-btn').addEventListener('click', () => {
        this.expanded = !this.expanded;
        this.detailsEl.hidden = !this.expanded;
        this.el.querySelector('.aa-details-btn').setAttribute('aria-expanded', String(this.expanded));
        this.onToggleDetails?.(this.expanded);
      });
      this._timer = null;
      this._started = performance.now();
    }

    startTimer(startedAt) {
      this._started = startedAt || performance.now();
      if (this._timer) clearInterval(this._timer);
      const tick = () => {
        const sec = Math.max(0, (performance.now() - this._started) / 1000);
        this.elapsedEl.textContent = `${sec.toFixed(0)}s`;
      };
      tick();
      this._timer = setInterval(tick, 1000);
    }

    stopTimer() {
      if (this._timer) {
        clearInterval(this._timer);
        this._timer = null;
      }
    }

    render(state) {
      if (!state || state.status === STATUS.idle) {
        this.el.hidden = true;
        return;
      }
      const show =
        state.showProgressCard ||
        (state.tasks && state.tasks.length > 0) ||
        (state.tools && state.tools.length > 0);
      if (!show && !['thinking', 'planning', 'using_tool', 'streaming_response'].includes(state.status)) {
        // Compact completed: keep if history exists
        if (state.tasks?.length && state.status === STATUS.completed) {
          this.el.hidden = false;
          this.el.classList.add('aa-complete');
        } else {
          this.el.hidden = true;
          return;
        }
      } else {
        this.el.hidden = false;
        this.el.classList.toggle('aa-complete', state.status === STATUS.completed);
      }

      const title =
        state.hasStreamedText && state.status === STATUS.streaming_response
          ? 'Working in the background'
          : state.label || 'Thinking';
      this.titleEl.textContent = title;
      // Shimmer only while waiting for first token — never after stream starts.
      const titleShimmer =
        !state.hasStreamedText &&
        !prefersReducedMotion() &&
        state.status !== STATUS.completed &&
        state.status !== STATUS.failed &&
        state.status !== STATUS.cancelled;
      this.titleEl.classList.toggle('aa-shimmer-text', titleShimmer);
      this.titleEl.classList.toggle('thinking-word', titleShimmer);

      this.pillEl.textContent = (state.status || 'thinking').replace(/_/g, ' ');
      this.listEl.innerHTML = '';
      const tasks = state.tasks || [];
      tasks.forEach((task) => {
        const li = document.createElement('li');
        li.className = `work-step aa-task-item is-${task.state}`;
        li.setAttribute('aria-label', `${task.title}, ${taskStateLabel(task.state)}`);
        const glyph = document.createElement('span');
        glyph.className = 'work-step-dot aa-task-glyph';
        glyph.setAttribute('aria-hidden', 'true');
        const body = document.createElement('div');
        const t = document.createElement('div');
        t.className = 'work-step-title';
        t.textContent = task.title;
        body.appendChild(t);
        if (task.detail) {
          const d = document.createElement('div');
          d.className = 'work-step-detail';
          d.textContent = task.detail;
          body.appendChild(d);
        }
        li.appendChild(glyph);
        li.appendChild(body);
        this.listEl.appendChild(li);
      });

      // Tools (collapsed summary; details panel holds more)
      const tools = state.tools || [];
      if (tools.length) {
        this.toolsEl.hidden = false;
        this.toolsEl.innerHTML = tools
          .slice(-6)
          .map(
            (tool) =>
              `<div class="aa-tool-item is-${tool.state}" role="listitem">
                <span class="aa-tool-label">${escapeHtml(tool.label || tool.name || 'Tool')}</span>
                <span class="aa-tool-state">${escapeHtml(tool.state)}</span>
              </div>`
          )
          .join('');
      } else {
        this.toolsEl.hidden = true;
        this.toolsEl.innerHTML = '';
      }

      this.detailsEl.innerHTML = `
        <div class="aa-details-inner">
          <p class="aa-details-note">Safe execution summary. Secrets and full credentials are never shown.</p>
          <ul class="aa-details-list">
            ${(tools || [])
              .map(
                (tool) =>
                  `<li><strong>${escapeHtml(tool.label || tool.name || 'Tool')}</strong>
                    <span>${escapeHtml(tool.preview || tool.error || tool.state || '')}</span></li>`
              )
              .join('')}
          </ul>
        </div>
      `;
    }

    destroy() {
      this.stopTimer();
      this.el.remove();
    }
  }

  function escapeHtml(s) {
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  class AgentExecutionController {
    constructor({
      chatHost,
      voiceHost,
      activityStrip,
      onInterrupt,
    } = {}) {
      this.state = emptyState();
      this.listeners = new Set();
      this.chatHost = chatHost || null;
      this.voiceHost = voiceHost || null;
      this.activityStrip = activityStrip || null;
      this.onInterrupt = onInterrupt;
      this._chatWrap = null;
      this._shimmer = null;
      this._card = null;
      this._voiceCard = null;
      this._voiceShimmer = null;
    }

    subscribe(fn) {
      this.listeners.add(fn);
      return () => this.listeners.delete(fn);
    }

    _emit() {
      for (const fn of this.listeners) {
        try {
          fn(this.state);
        } catch (_) {}
      }
      this._render();
    }

    dispatch(event) {
      if (!event) return;
      if (event.type === 'request_started') {
        this._teardownUi();
      }
      this.state = reduceExecution(this.state, event);
      this._emit();
    }

    /** Map legacy WS messages into normalized events */
    ingestMessage(msg) {
      if (!msg || !msg.type) return;

      if (msg.type === 'execution' && msg.event) {
        this.dispatch(msg.event);
        return;
      }

      if (msg.type === 'state') {
        const map = {
          transcribing: { status: STATUS.thinking, label: 'Transcribing' },
          thinking: { status: STATUS.thinking, label: 'Thinking' },
          saving: { status: STATUS.thinking, label: 'Saving to memory' },
        };
        const m = map[msg.state];
        if (m) {
          if (!this.state.requestId) {
            this.dispatch({
              type: 'request_started',
              request_id: `req_local_${Date.now()}`,
              id: `local_start_${Date.now()}`,
              label: m.label,
              timestamp: Date.now() / 1000,
            });
          }
          this.dispatch({
            type: 'status_changed',
            request_id: this.state.requestId,
            id: `st_${msg.state}_${Date.now()}`,
            status: m.status,
            label: m.label,
            timestamp: Date.now() / 1000,
          });
        }
        return;
      }

      if (msg.type === 'activity') {
        const label = msg.label || toolActivityLabel(msg.tool);
        if (!this.state.requestId) {
          this.dispatch({
            type: 'request_started',
            request_id: `req_local_${Date.now()}`,
            id: `local_act_${Date.now()}`,
            label,
            timestamp: Date.now() / 1000,
          });
        }
        this.dispatch({
          type: 'status_changed',
          request_id: this.state.requestId,
          id: `act_${Date.now()}`,
          status: STATUS.using_tool,
          label,
          timestamp: Date.now() / 1000,
        });
        if (msg.tool_call_id || msg.tool) {
          this.dispatch({
            type: 'tool_started',
            request_id: this.state.requestId,
            id: `tool_${msg.tool_call_id || Date.now()}`,
            tool_call_id: msg.tool_call_id || `t_${Date.now()}`,
            tool_name: msg.tool,
            label,
            timestamp: Date.now() / 1000,
          });
        }
        return;
      }

      if (msg.type === 'token' && msg.text) {
        if (!this.state.requestId) {
          this.dispatch({
            type: 'request_started',
            request_id: `req_local_${Date.now()}`,
            id: `local_tok_${Date.now()}`,
            label: 'Thinking',
            timestamp: Date.now() / 1000,
          });
        }
        this.dispatch({
          type: 'text_delta',
          request_id: this.state.requestId,
          id: `tok_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
          delta: msg.text,
          timestamp: Date.now() / 1000,
        });
        return;
      }

      if (msg.type === 'done') {
        if (!this.state.requestId) return;
        if (msg.reason === 'interrupted') {
          this.dispatch({
            type: 'response_cancelled',
            request_id: this.state.requestId,
            id: `done_cancel_${Date.now()}`,
            timestamp: Date.now() / 1000,
          });
        } else if (msg.ok === false && msg.reason === 'error') {
          this.dispatch({
            type: 'response_failed',
            request_id: this.state.requestId,
            id: `done_fail_${Date.now()}`,
            error: msg.message || 'Failed',
            timestamp: Date.now() / 1000,
          });
        } else {
          this.dispatch({
            type: 'response_completed',
            request_id: this.state.requestId,
            id: `done_ok_${Date.now()}`,
            timestamp: Date.now() / 1000,
          });
        }
        return;
      }

      if (msg.type === 'error') {
        if (this.state.requestId) {
          this.dispatch({
            type: 'response_failed',
            request_id: this.state.requestId,
            id: `err_${Date.now()}`,
            error: msg.message || 'Error',
            timestamp: Date.now() / 1000,
          });
        }
      }
    }

    beginLocal(label = 'Thinking') {
      this.dispatch({
        type: 'request_started',
        request_id: `req_local_${Date.now()}`,
        id: `begin_${Date.now()}`,
        label,
        timestamp: Date.now() / 1000,
      });
    }

    reset() {
      this.state = emptyState();
      this._teardownUi();
      this._emit();
    }

    _ensureChatUi() {
      if (!this.chatHost) return;
      if (this._chatWrap && this._chatWrap.isConnected) return;
      this._chatWrap = document.createElement('div');
      this._chatWrap.className = 'msg ai pending work-card-wrap aa-turn-panel';
      this._chatWrap.id = 'agentActivityPanel';
      this.chatHost.appendChild(this._chatWrap);
      const shimmerHost = document.createElement('div');
      shimmerHost.className = 'aa-shimmer-host thinking-line';
      this._chatWrap.appendChild(shimmerHost);
      this._shimmer = new AgentStatusShimmer(shimmerHost);
      const cardHost = document.createElement('div');
      this._chatWrap.appendChild(cardHost);
      this._card = new AgentProgressCard(cardHost, {
        onInterrupt: () => this.onInterrupt?.(),
      });
      this._card.startTimer(this.state.startedAt || performance.now());
    }

    _ensureVoiceUi() {
      if (!this.voiceHost) return;
      if (this._voiceCard && this._voiceCard.el.isConnected) return;
      this.voiceHost.innerHTML = '';
      const shimmerHost = document.createElement('div');
      shimmerHost.className = 'aa-voice-shimmer';
      this.voiceHost.appendChild(shimmerHost);
      this._voiceShimmer = new AgentStatusShimmer(shimmerHost);
      const cardHost = document.createElement('div');
      cardHost.className = 'aa-voice-card-host';
      this.voiceHost.appendChild(cardHost);
      this._voiceCard = new AgentProgressCard(cardHost, {
        onInterrupt: () => this.onInterrupt?.(),
      });
      this._voiceCard.startTimer(this.state.startedAt || performance.now());
    }

    _teardownUi() {
      this._shimmer?.destroy();
      this._card?.destroy();
      this._voiceShimmer?.destroy();
      this._voiceCard?.destroy();
      this._shimmer = null;
      this._card = null;
      this._voiceShimmer = null;
      this._voiceCard = null;
      this._chatWrap?.remove();
      this._chatWrap = null;
      if (this.voiceHost) this.voiceHost.innerHTML = '';
      if (this.activityStrip) this.activityStrip.innerHTML = '';
    }

    _render() {
      const st = this.state;
      const active = st.status && st.status !== STATUS.idle;

      if (!active) {
        // Keep completed card briefly if tasks exist
        if (st.status === STATUS.completed && st.tasks?.length) {
          this._ensureChatUi();
          this._card?.render(st);
          this._shimmer?.stop();
          if (this.voiceHost && this.voiceHost.offsetParent !== null) {
            this._ensureVoiceUi();
            this._voiceCard?.render(st);
            this._voiceShimmer?.stop();
          }
          return;
        }
        this._teardownUi();
        return;
      }

      this._ensureChatUi();

      // Shimmer only before meaningful streamed text
      if (!st.hasStreamedText && st.label) {
        this._shimmer?.setLabel(st.label);
      } else {
        this._shimmer?.stop();
      }

      // Progress card when multi-step / tools, or long running without text
      const wantCard =
        st.showProgressCard ||
        (st.tasks && st.tasks.length > 0) ||
        (st.tools && st.tools.length > 0);
      if (wantCard) {
        this._card?.render(st);
        this._chatWrap?.classList.remove('thinking-only');
      } else {
        // Lightweight shimmer only
        if (this._card) {
          this._card.el.hidden = true;
        }
        this._chatWrap?.classList.add('thinking-only');
      }

      if (this.activityStrip) {
        if (!st.hasStreamedText && st.label) {
          this.activityStrip.innerHTML = '';
          const pill = document.createElement('span');
          pill.className = 'activity-pill';
          const word = document.createElement('span');
          word.className = prefersReducedMotion() ? 'aa-status-static' : 'aa-shimmer-text thinking-word';
          word.textContent = st.label;
          pill.appendChild(word);
          this.activityStrip.appendChild(pill);
        } else if (st.status === STATUS.using_tool && st.label && !st.hasStreamedText) {
          this.activityStrip.innerHTML = '';
          const pill = document.createElement('span');
          pill.className = 'activity-pill';
          const word = document.createElement('span');
          word.className = prefersReducedMotion() ? 'aa-status-static' : 'aa-shimmer-text thinking-word';
          word.textContent = st.label;
          pill.appendChild(word);
          this.activityStrip.appendChild(pill);
        } else if (st.hasStreamedText) {
          this.activityStrip.innerHTML = '';
        }
      }

      // Voice overlay — same store
      if (this.voiceHost && (this.voiceHost.offsetParent !== null || this.voiceHost.dataset.force === '1')) {
        this._ensureVoiceUi();
        if (!st.hasStreamedText && st.label) this._voiceShimmer?.setLabel(st.label);
        else this._voiceShimmer?.stop();
        this._voiceCard?.render(st);
      }
    }
  }

  // Inject companion CSS once (must not clobber shimmer animation).
  function injectStyles() {
    if (document.getElementById('agent-activity-styles')) return;
    const style = document.createElement('style');
    style.id = 'agent-activity-styles';
    style.textContent = `
      .aa-status-shimmer {
        position: relative;
        min-height: 1.45em;
        overflow: hidden;
        line-height: 1.35;
      }
      .aa-status-row {
        display: inline-block;
        max-width: 100%;
      }
      /* Cross-fade ONLY on the row wrapper — never on the shimmer text node. */
      .aa-status-row.aa-status-exit {
        position: absolute;
        left: 0;
        top: 0;
        animation: aa-exit .2s ease forwards;
        pointer-events: none;
      }
      .aa-status-row.aa-status-enter {
        animation: aa-enter .2s ease;
      }
      @keyframes aa-exit {
        from { opacity: 1; transform: translateY(0); }
        to   { opacity: 0; transform: translateY(-6px); }
      }
      @keyframes aa-enter {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
      }
      /* Self-contained shimmer: theme tokens, ~1.9s loop, no layout shift.
         !important on animation avoids later rules wiping the sweep. */
      .aa-shimmer-text {
        display: inline-block !important;
        background-image: linear-gradient(
          90deg,
          var(--muted) 0%,
          var(--muted) 38%,
          var(--text) 50%,
          var(--muted) 62%,
          var(--muted) 100%
        ) !important;
        background-size: 220% 100% !important;
        background-repeat: no-repeat !important;
        background-position: 100% 0 !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: transparent !important;
        animation: thinking-word-sweep 1.9s linear infinite !important;
      }
      .aa-status-static {
        display: inline-block;
        color: var(--muted);
      }
      .work-status-title.aa-shimmer-text {
        /* Keep title size/weight; shimmer fill only. */
        color: transparent !important;
      }
      .aa-shimmer-host.thinking-line {
        color: inherit;
      }
      .aa-progress-card { margin-top: 6px; }
      .aa-task-list { list-style: none; margin: 0; padding: 0; }
      .aa-task-item.is-active .work-step-dot {
        border: 2px solid transparent;
        border-top-color: var(--accent);
        border-right-color: var(--accent);
        background: transparent !important;
        animation: aa-spin 1s linear infinite;
        border-radius: 50%;
        width: 10px; height: 10px;
      }
      .aa-task-item.is-completed .work-step-dot {
        background: var(--accent) !important;
        border-color: var(--accent);
        position: relative;
      }
      .aa-task-item.is-completed .work-step-dot::after {
        content: "";
        position: absolute; left: 2px; top: 1px;
        width: 3px; height: 6px;
        border: solid #041c1c;
        border-width: 0 1.5px 1.5px 0;
        transform: rotate(45deg);
      }
      .aa-task-item.is-failed .work-step-dot {
        background: #c45 !important;
        border-color: #c45;
      }
      .aa-task-item.is-cancelled { opacity: .55; }
      .aa-task-item.is-completed { opacity: .72; }
      @keyframes aa-spin { to { transform: rotate(360deg); } }
      .aa-tools { display: grid; gap: 4px; margin-top: 10px; }
      .aa-tool-item {
        display: flex; justify-content: space-between; gap: 8px;
        font-size: 12px; color: var(--muted);
        padding: 4px 8px; border-radius: 10px;
        background: color-mix(in srgb, var(--panel-2) 60%, transparent);
      }
      .aa-tool-item.is-active .aa-tool-label { color: var(--text); }
      .aa-details-btn {
        border: 0; background: transparent; color: var(--muted);
        font-size: 11.5px; font-weight: 700; cursor: pointer;
        text-decoration: underline; text-underline-offset: 2px;
      }
      .aa-details { margin-top: 8px; }
      .aa-details-inner {
        border-radius: 12px; padding: 10px;
        border: 1px solid var(--line);
        background: color-mix(in srgb, var(--panel-2) 70%, transparent);
        font-size: 12px; color: var(--muted);
      }
      .aa-details-list { margin: 6px 0 0; padding-left: 1.1em; display: grid; gap: 4px; }
      .aa-details-list strong { color: var(--text); font-weight: 700; margin-right: 6px; }
      .aa-voice-card-host { width: min(92vw, 420px); margin: 0 auto; }
      .aa-voice-shimmer {
        text-align: center;
        min-height: 1.45em;
        margin-bottom: 8px;
        font-size: 16px;
        font-weight: 760;
        letter-spacing: -.02em;
      }
      .aa-complete .work-status-pill::before { animation: none; background: var(--muted); }
      :root[data-motion="reduced"] .thinking-word,
      :root[data-motion="reduced"] .aa-shimmer-text,
      :root[data-motion="reduced"] .aa-task-item.is-active .work-step-dot,
      :root[data-motion="reduced"] .work-status-card::before {
        animation: none !important;
      }
      :root[data-motion="reduced"] .thinking-word,
      :root[data-motion="reduced"] .aa-shimmer-text {
        background: none !important;
        -webkit-background-clip: border-box !important;
        background-clip: border-box !important;
        -webkit-text-fill-color: var(--muted) !important;
        color: var(--muted) !important;
      }
      @media (max-width: 430px) {
        .aa-progress-card { width: 100%; border-radius: 18px; }
        .aa-tool-item { font-size: 12.5px; }
      }
    `;
    document.head.appendChild(style);
  }

  injectStyles();

  global.HermesAgentActivity = {
    STATUS,
    TASK,
    toolActivityLabel,
    reduceExecution,
    emptyState,
    AgentStatusShimmer,
    AgentProgressCard,
    AgentExecutionController,
  };
})(typeof window !== 'undefined' ? window : globalThis);

"""Agent runtime for JSON-RPC/ZMQ based ChiSurf fitting agent.

Separates LLM interaction, tool parsing, allowlist enforcement, and
fit-loop bookkeeping from the Qt UI layer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

_LOG = logging.getLogger("chisurf.agent.runtime")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_TOOLS: set[str] = {
    "session.describe",
    "fit.list",
    "fit.get",
    "fit.diagnostics",
    "fit.parameter_snapshot",
    "fit.restore_parameters",
    "fit.run",
    "fit.curve_data",
    "fit.update",
    "fit.select",
    "parameter.get",
    "parameter.set_value",
    "parameter.set_fixed",
    "parameter.set_bounds",
    "parameter.set_bounds_on",
    "project.info",
    "project.save",
    "log.write",
    "editor.document.list",
    "editor.document.get",
    "editor.document.apply_edits",
    "editor.document.ruff_check",
    "agent.code.run",
}

BLOCKED_TOOLS: set[str] = {
    "fit.clear",
    "fit.remove",
    "dataset.clear",
    "dataset.remove",
    "session.clear",
    "session.restore",
}

TOOLS_REQUIRING_PARAMS: dict[str, list[str]] = {
    "parameter.set_value": ["parameter_name", "value"],
    "parameter.set_fixed": ["parameter_name", "fixed"],
    "parameter.set_bounds": ["parameter_name", "bounds"],
    "parameter.set_bounds_on": ["parameter_name", "bounds_on"],
    "agent.code.run": ["code"],
    "project.save": ["target_path"],
    "editor.document.apply_edits": ["edits"],
    "log.write": ["message"],
    "fit.restore_parameters": ["snapshot"],
    "fit.diagnostics": [],
    "fit.parameter_snapshot": [],
    "fit.run": [],
    "fit.get": [],
    "fit.list": [],
    "fit.curve_data": [],
    "fit.update": [],
    "fit.select": [],
    "session.describe": [],
    "parameter.get": ["parameter_name"],
    "project.info": [],
    "editor.document.list": [],
    "editor.document.get": [],
    "editor.document.ruff_check": [],
}


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------


class AgentMode(Enum):
    CHAT_ONLY = "chat_only"
    CHISURF_TOOLS = "chisurf_tools"
    AUTONOMOUS_FIT = "autonomous_fit"


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentRunConfig:
    mode: AgentMode = AgentMode.CHAT_ONLY
    default_mode: AgentMode = AgentMode.CHAT_ONLY
    max_tool_iterations: int = 25
    tool_timeout_ms: int = 30000
    invalid_json_retries: int = 2
    blocked_tool_policy: str = "reject"
    chisurf_rpc_host: str = "127.0.0.1"
    chisurf_rpc_cmd_port: int = 8765
    chisurf_rpc_pub_port: int = 8766
    editor_rpc_host: str = "127.0.0.1"
    editor_rpc_cmd_port: int = 8775
    editor_rpc_pub_port: int = 8776
    code_run_enabled: bool = True
    code_timeout_ms: int = 5000
    output_max_chars: int = 20000


@dataclass
class AgentToolCall:
    tool: str
    params: Dict[str, Any]


@dataclass
class AgentObservation:
    tool: str
    result: Dict[str, Any]
    elapsed_ms: int
    ok: bool


@dataclass
class AgentRunState:
    iteration: int = 0
    fit_runs: int = 0
    status: AgentStatus = AgentStatus.IDLE
    best_metrics: Optional[Dict[str, Any]] = None
    best_snapshot: Optional[Dict[str, Any]] = None
    worsening_count: int = 0
    initial_metrics: Optional[Dict[str, Any]] = None
    started_at: Optional[float] = None
    runtime_timeout_s: float = 300.0
    cancelled: bool = False
    selected_fit_uid: Optional[str] = None
    selected_fit_name: Optional[str] = None
    rollback_done: bool = False


# ---------------------------------------------------------------------------
# Event callback type
# ---------------------------------------------------------------------------

AgentEventCallback = Callable[[str, Dict[str, Any]], None]


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------


class AgentToolRegistry:
    """Registry that maps tool names to RPC call wrappers."""

    def __init__(
        self,
        chisurf_client: Any,
        editor_client: Any,
        config: AgentRunConfig,
    ):
        self._chisurf = chisurf_client
        self._editor = editor_client
        self._config = config

    @staticmethod
    def _run_code_locally(
        code: str,
        timeout_ms: int,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute ``agent.code.run`` locally via subprocess.

        This runs in the agent/UI process to avoid deadlocking the
        single-threaded ChiSurf REP server when the child code calls
        back into it through ``ChisurfClient``.

        Parameters
        ----------
        code : str
            Python source code.
        timeout_ms : int
            Subprocess timeout in milliseconds.
        cwd : str, optional
            Working directory.

        Returns
        -------
        dict
            Service-style result dict.

        """
        import os as _os
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile
        import time as _time

        if not code:
            return {"ok": False, "error": "no code provided"}
        timeout_s = max(1.0, float(timeout_ms) / 1000.0)
        max_output = 20000

        import chisurf as _cs
        repo_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(_cs.__file__)))
        work_dir = str(cwd) if cwd else repo_root

        tmp = _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="agent_code_", delete=False,
        )
        try:
            tmp.write(code)
            tmp.close()
            env = _os.environ.copy()
            env["CHISURF_RPC_HOST"] = "127.0.0.1"
            env["CHISURF_RPC_CMD_PORT"] = "8765"
            env["CHISURF_RPC_PUB_PORT"] = "8766"

            t0 = _time.perf_counter()
            proc = _subprocess.run(
                [_sys.executable, tmp.name],
                capture_output=True, text=True,
                timeout=timeout_s, cwd=work_dir, env=env, shell=False,
            )
            elapsed_ms = int((_time.perf_counter() - t0) * 1000)

            stdout = (proc.stdout or "")[:max_output]
            stderr = (proc.stderr or "")[:max_output]

            result_json = None
            stdout_lines = stdout.splitlines()
            if stdout_lines and stdout_lines[-1].startswith("RESULT_JSON:"):
                import json as _json
                try:
                    payload = stdout_lines[-1][len("RESULT_JSON:"):]
                    result_json = _json.loads(payload)
                except Exception:
                    pass

            return {
                "ok": True, "stdout": stdout, "stderr": stderr,
                "result_json": result_json, "elapsed_ms": elapsed_ms,
            }
        except _subprocess.TimeoutExpired:
            return {"ok": False, "error": f"code execution timed out after {timeout_s}s"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            try:
                _os.unlink(tmp.name)
            except Exception:
                pass

    def execute(
        self,
        call: AgentToolCall,
    ) -> Dict[str, Any]:
        """Execute a tool call through the appropriate RPC client.

        ``agent.code.run`` is executed locally (not via RPC) to avoid
        deadlocking the single-threaded REP server.

        Parameters
        ----------
        call : AgentToolCall
            The tool call to execute.

        Returns
        -------
        dict
            The RPC result dict.

        """
        tool = call.tool
        params = dict(call.params)

        if tool == "agent.code.run":
            if not self._config.code_run_enabled:
                return {"ok": False, "error": "code execution disabled"}
            return self._run_code_locally(
                code=params.get("code", ""),
                timeout_ms=params.get("timeout_ms", self._config.code_timeout_ms),
                cwd=params.get("cwd"),
            )

        editor_tools = {
            "editor.document.list",
            "editor.document.get",
            "editor.document.apply_edits",
            "editor.document.ruff_check",
        }
        if tool in editor_tools:
            if self._editor is None:
                return {"ok": False, "error": "editor RPC not available"}
            return self._editor.call(tool, params)

        # Backward compat: convert lower/upper to bounds for parameter.set_bounds
        if tool == "parameter.set_bounds" and "lower" in params and "upper" in params:
            raw = dict(params)
            raw["bounds"] = [raw.pop("lower"), raw.pop("upper")]
            return self._chisurf.call(tool, raw)
        return self._chisurf.call(tool, params)


# ---------------------------------------------------------------------------
# Metric comparison helpers
# ---------------------------------------------------------------------------


def _extract_metrics(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract metrics dict from a ``fit.diagnostics`` result."""
    if not result.get("ok"):
        return None
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics


def _compare_metrics(
    current: Optional[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Compare two metrics dicts.

    Returns ``(improved, reason)``.  Uses ``chi2r`` as primary,
    ``chi2`` as fallback.  Lower finite values are better.

    Parameters
    ----------
    current : dict or None
        Current metrics.
    best : dict or None
        Best-known metrics.

    """
    if current is None:
        return False, "no current metrics"
    if best is None:
        return True, "first measurement"

    for key in ("chi2r", "chi2"):
        c = current.get(key)
        b = best.get(key)
        if c is not None and b is not None:
            import math
            if not math.isfinite(c):
                return False, f"non-finite current {key}={c}"
            if not math.isfinite(b):
                return True, f"best {key} was non-finite"
            if c < b:
                return True, f"{key} improved: {b} -> {c}"
            if c > b:
                return False, f"{key} worsened: {b} -> {c}"
            return True, f"{key} unchanged"

    return False, "no comparable metrics in chi2r or chi2"


# ---------------------------------------------------------------------------
# JSON Protocol Parser
# ---------------------------------------------------------------------------


def parse_llm_response(
    text: str,
    allowed_tools: set[str],
    blocked_tools: set[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse an LLM response into a validated action dict.

    Returns ``(action, error)``.  On success ``error`` is ``None``.

    Parameters
    ----------
    text : str
        Raw LLM response text.
    allowed_tools : set of str
        Set of permitted tool names.
    blocked_tools : set of str
        Set of blocked tool names.

    """
    # Try to find a JSON block in the text
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, "invalid JSON"

    if not isinstance(obj, dict):
        return None, "response must be a JSON object"

    if "type" not in obj:
        return None, "missing 'type' field"

    msg_type = obj["type"]
    if msg_type not in ("message", "tool_call", "final"):
        return None, f"unknown type: {msg_type}"

    if msg_type in ("message", "final"):
        return obj, None

    tool = obj.get("tool")
    if not isinstance(tool, str) or not tool:
        return None, "tool_call missing 'tool' (string)"

    if tool not in allowed_tools:
        if tool in blocked_tools:
            return None, f"blocked tool: {tool}"
        return None, f"unknown tool: {tool}"

    params = obj.get("params", {})
    if params is None:
        params = {}

    required = TOOLS_REQUIRING_PARAMS.get(tool, [])
    for rp in required:
        if rp not in params:
            return None, f"missing required parameter '{rp}' for {tool}"

    return obj, None


# ---------------------------------------------------------------------------
# Agent Runtime
# ---------------------------------------------------------------------------


class AgentRuntime:
    """Orchestrates the LLM-driven agent loop.

    The runtime owns LLM request/response, tool parsing, execution,
    fit-loop bookkeeping, and event emission.  The UI layer receives
    events via callbacks.
    """

    def __init__(
        self,
        config: AgentRunConfig,
        tool_registry: AgentToolRegistry,
        event_callback: Optional[AgentEventCallback] = None,
        llm_call_fn: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
        system_prompt_fn: Optional[Callable[[], str]] = None,
    ):
        self._config = config
        self._registry = tool_registry
        self._event = event_callback
        self._llm_call = llm_call_fn
        self._system_prompt_fn = system_prompt_fn
        self._state = AgentRunState()
        self._messages: List[Dict[str, Any]] = []
        self._system_prompt: str = ""

    # ── Public API ────────────────────────────────────────────────────

    def start(self, user_message: str) -> None:
        """Start the agent loop with a user message.

        Parameters
        ----------
        user_message : str
            The user's input.

        """
        self._state = AgentRunState(
            status=AgentStatus.RUNNING,
            started_at=time.perf_counter(),
            runtime_timeout_s=self._config.max_tool_iterations * 5.0,
        )
        self._system_prompt = self._build_system_prompt()
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._messages.append({"role": "user", "content": user_message})

        self._emit("message.started", {"content": user_message})

        self._run_loop()

    def cancel(self) -> None:
        """Request cancellation (best-effort)."""
        self._state.cancelled = True
        self._state.status = AgentStatus.CANCELLING

    @property
    def state(self) -> AgentRunState:
        return self._state

    # ── Internal Loop ─────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main agent loop."""
        try:
            while not self._should_stop():
                text = self._call_llm()
                if text is None:
                    break

                action, error = self._try_parse(text)
                if action is None:
                    self._emit("tool.failed", {"error": error or "parse failed"})
                    if "blocked tool" in (error or "") or "unknown tool" in (error or ""):
                        continue
                    break

                if self._handle_action(action):
                    break

            self._finalize()
        except Exception as e:
            _LOG.exception("agent runtime error")
            self._state.status = AgentStatus.FAILED
            self._emit("agent.failed", {"error": str(e)})

    def _should_stop(self) -> bool:
        """Check all stopping conditions."""
        if self._state.cancelled:
            self._emit("agent.cancelled", {})
            return True
        if self._state.iteration >= self._config.max_tool_iterations:
            self._emit("tool.failed", {"error": "max iterations reached"})
            return True
        if self._state.worsening_count >= 3:
            self._emit("tool.failed", {"error": "no improvement for 3 consecutive fit runs"})
            return True
        if time.perf_counter() - (self._state.started_at or 0) > self._state.runtime_timeout_s:
            self._emit("tool.failed", {"error": "runtime timeout"})
            return True
        return False

    def _call_llm(self) -> Optional[str]:
        """Call the LLM and return text, or None on failure."""
        if self._llm_call is None:
            self._state.status = AgentStatus.FAILED
            self._emit("agent.failed", {"error": "no LLM call function configured"})
            return None

        text = self._llm_call(self._messages)
        self._messages.append({"role": "assistant", "content": text})
        return text

    def _try_parse(self, text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Try parsing LLM text, with retries on invalid JSON only.

        Returns ``(action, None)`` or ``(None, error)``.

        """
        retries = self._config.invalid_json_retries
        for attempt in range(retries + 1):
            action, error = parse_llm_response(
                text,
                ALLOWED_TOOLS,
                BLOCKED_TOOLS,
            )
            if action is not None:
                return action, None

            # Retry only invalid JSON; report other errors immediately
            if error and "invalid JSON" not in error:
                return None, error

            if attempt >= retries:
                return None, error or "invalid JSON after retries"

            corrective = (
                "Your previous response could not be parsed as valid JSON. "
                "Respond with ONLY a JSON object of the forms:\n"
                '  {"type": "message", "content": "..."}\n'
                '  {"type": "tool_call", "tool": "...", "params": {...}}\n'
                '  {"type": "final", "content": "..."}\n'
                "Do not include markdown fences or any other text."
            )
            self._messages.append({"role": "user", "content": corrective})
            text = self._call_llm()
            if text is None:
                return None, "LLM call failed during retry"

        return None, "invalid JSON after retries"

    def _handle_action(self, action: Dict[str, Any]) -> bool:
        """Handle a parsed action. Returns True if loop should stop."""
        msg_type = action["type"]

        if msg_type == "message":
            content = action.get("content", "")
            self._emit("message.completed", {"content": content})
            self._messages.append({"role": "user", "content": content})
            return False

        if msg_type == "final":
            content = action.get("content", "")
            self._emit("message.completed", {"content": content})
            return True

        if msg_type == "tool_call":
            self._state.iteration += 1
            tool = action["tool"]
            params = action.get("params", {})
            return self._execute_tool(tool, params)

        return False

    def _execute_tool(self, tool: str, params: Dict[str, Any]) -> bool:
        """Execute a tool and handle fit-loop bookkeeping.

        Returns True if the loop should stop.

        """
        call = AgentToolCall(tool=tool, params=params)
        t0 = time.perf_counter()

        is_fit_run = tool == "fit.run"
        if is_fit_run:
            self._state.fit_runs += 1
            self._emit("fit.iteration.started", {"tool": tool, "params": params})

        self._emit("tool.started", {"tool": tool, "params": params})

        try:
            result = self._registry.execute(call)
        except Exception as e:
            elapsed = int((time.perf_counter() - t0) * 1000)
            self._emit("tool.failed", {"tool": tool, "error": str(e), "elapsed_ms": elapsed})
            return True

        elapsed = int((time.perf_counter() - t0) * 1000)
        ok = result.get("ok", False)

        observation = AgentObservation(
            tool=tool,
            result=result,
            elapsed_ms=elapsed,
            ok=ok,
        )

        self._emit("tool.completed", {
            "tool": tool,
            "ok": ok,
            "elapsed_ms": elapsed,
            "result": self._compact_result(result),
        })

        if is_fit_run:
            self._handle_fit_completion(result)

        self._messages.append({
            "role": "user",
            "content": self._format_observation(observation),
        })

        return not ok

    def _handle_fit_completion(self, result: Dict[str, Any]) -> None:
        """Track fit-run results for metric comparison and rollback."""
        if not result.get("ok"):
            self._state.worsening_count += 1
            self._emit("fit.iteration.completed", {
                "ok": False,
                "worsening_count": self._state.worsening_count,
            })
            return

        diag = self._registry.execute(AgentToolCall(
            tool="fit.diagnostics",
            params={"fit_uid": self._state.selected_fit_uid} if self._state.selected_fit_uid else {},
        ))
        current_metrics = _extract_metrics(diag)
        if current_metrics is None:
            self._state.worsening_count += 1
            self._emit("fit.iteration.completed", {
                "ok": False,
                "error": "cannot extract metrics",
                "worsening_count": self._state.worsening_count,
            })
            return

        if self._state.initial_metrics is None:
            self._state.initial_metrics = dict(current_metrics)

        improved, reason = _compare_metrics(current_metrics, self._state.best_metrics)

        if improved:
            snapshot = self._registry.execute(AgentToolCall(
                tool="fit.parameter_snapshot",
                params={"fit_uid": self._state.selected_fit_uid} if self._state.selected_fit_uid else {},
            ))
            if snapshot.get("ok"):
                self._state.best_snapshot = snapshot.get("snapshot")
            self._state.best_metrics = current_metrics
            self._state.worsening_count = 0
        else:
            self._state.worsening_count += 1

        self._emit("fit.iteration.completed", {
            "ok": improved,
            "reason": reason,
            "metrics": current_metrics,
            "worsening_count": self._state.worsening_count,
        })

    # ── Rollback ──────────────────────────────────────────────────────

    def _rollback_if_needed(self) -> None:
        """Restore best-known snapshot if fit worsened."""
        if self._state.rollback_done:
            return
        if self._state.worsening_count <= 0 and self._state.best_snapshot is not None:
            return
        if self._state.best_snapshot is None:
            return

        try:
            self._registry.execute(AgentToolCall(
                tool="fit.restore_parameters",
                params={
                    "snapshot": self._state.best_snapshot,
                    "fit_uid": self._state.selected_fit_uid,
                } if self._state.selected_fit_uid else {
                    "snapshot": self._state.best_snapshot,
                },
            ))
            self._state.rollback_done = True
            self._emit("fit.rollback.completed", {})
        except Exception as e:
            _LOG.warning("rollback failed: %s", e)

    # ── Finalize ──────────────────────────────────────────────────────

    def _finalize(self) -> None:
        """Emit final summary, rollback if needed, mark completed/failed."""
        self._rollback_if_needed()

        if self._state.status == AgentStatus.RUNNING:
            self._state.status = AgentStatus.COMPLETED

        summary: Dict[str, Any] = {
            "fit_name": self._state.selected_fit_name,
            "fit_uid": self._state.selected_fit_uid,
            "initial_metrics": self._state.initial_metrics,
            "best_metrics": self._state.best_metrics,
            "fit_runs": self._state.fit_runs,
            "iterations": self._state.iteration,
            "rollback_done": self._state.rollback_done,
            "status": self._state.status.value,
        }

        if self._state.status == AgentStatus.FAILED:
            self._emit("agent.failed", summary)
        else:
            self._emit("agent.completed", summary)

    # ── Helpers ───────────────────────────────────────────────────────

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Emit a runtime event to the registered callback."""
        if self._event is not None:
            try:
                self._event(event, data)
            except Exception:
                _LOG.exception("event callback error")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for tool-using modes."""
        if self._system_prompt_fn:
            return self._system_prompt_fn()

        allowed = "\n".join(sorted(ALLOWED_TOOLS))
        prompt = (
            "You are a ChiSurf fitting agent. You interact with the ChiSurf "
            "fluorescence spectroscopy application through JSON-RPC tools.\n\n"
            "## Rules\n"
            "- Respond with ONLY a JSON object. No markdown, no explanation.\n"
            "- Use allowed tools only.\n"
            "- Do not invent RPC method names.\n"
            "- For fitting, always take a parameter snapshot before modifying.\n\n"
            "## Response Shapes\n"
            '{"type": "message", "content": "..."}\n'
            '{"type": "tool_call", "tool": "...", "params": {...}}\n'
            '{"type": "final", "content": "summary"}\n\n'
            "## Allowed Tools\n"
            f"{allowed}\n\n"
            "## Strategy\n"
            "1. Call session.describe to inspect the session.\n"
            "2. Call fit.get to inspect the target fit.\n"
            "3. Call fit.parameter_snapshot before modifying parameters.\n"
            "4. Use parameter.set_value etc. to adjust parameters.\n"
            "5. Call fit.run to execute the fit.\n"
            "6. Call fit.diagnostics to evaluate quality.\n"
            "7. Iterate until satisfied, then emit final.\n"
            "8. If the target fit is ambiguous, ask the user with a message."
        )
        return prompt

    @staticmethod
    def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact, JSON-safe version of the result for observations."""
        compact: Dict[str, Any] = {}
        for k, v in result.items():
            if k in ("curve",) and isinstance(v, dict):
                compact[k] = {sk: f"<{len(sv)} points>" for sk, sv in v.items() if isinstance(sv, (list, tuple))}
            elif isinstance(v, (list, tuple)) and len(v) > 10:
                compact[k] = f"<{len(v)} items>"
            elif k in ("stdout", "stderr") and isinstance(v, str) and len(v) > 500:
                compact[k] = v[:500] + "..."
            else:
                compact[k] = v
        return compact

    @staticmethod
    def _format_observation(obs: AgentObservation) -> str:
        """Format a tool observation for feeding back to the LLM."""
        status = "OK" if obs.ok else "FAILED"
        result_json = json.dumps(obs.result, default=str, indent=2)
        max_chars = 5000
        if len(result_json) > max_chars:
            result_json = result_json[:max_chars] + "\n... (truncated)"
        return (
            f"Tool: {obs.tool}\n"
            f"Status: {status}\n"
            f"Elapsed: {obs.elapsed_ms}ms\n"
            f"Result:\n{result_json}"
        )


# ---------------------------------------------------------------------------
# Quick convenience: create a ``ChisurfClient``-based registry
# ---------------------------------------------------------------------------

def default_tool_registry(config: AgentRunConfig) -> AgentToolRegistry:
    """Build a tool registry backed by real ZMQ clients."""
    from chisurf.core.api._client import ChisurfClient

    chisurf = ChisurfClient(
        cmd_port=config.chisurf_rpc_cmd_port,
        pub_port=config.chisurf_rpc_pub_port,
        host=config.chisurf_rpc_host,
        timeout_ms=config.tool_timeout_ms,
    )
    chisurf.connect()

    editor_client: Any = None
    try:
        from chisurf.core.api._client import ChisurfClient as EdClient
        editor_client = EdClient(
            cmd_port=config.editor_rpc_cmd_port,
            pub_port=config.editor_rpc_pub_port,
            host=config.editor_rpc_host,
            timeout_ms=config.tool_timeout_ms,
        )
        editor_client.connect()
    except Exception:
        _LOG.warning("editor RPC client not available")

    return AgentToolRegistry(chisurf, editor_client, config)

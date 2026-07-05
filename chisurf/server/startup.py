from __future__ import annotations

import logging
import selectors
import socket
import subprocess
from typing import Any, Dict, Optional

_LOG = logging.getLogger("chisurf.server.startup")


def _read_available_stderr(proc: subprocess.Popen[Any], limit: int) -> bytes:
    """Read currently available stderr without blocking."""
    if proc.stderr is None:
        return b""
    selector = selectors.DefaultSelector()
    selector.register(proc.stderr, selectors.EVENT_READ)
    try:
        if not selector.select(timeout=0):
            return b""
        return proc.stderr.read1(limit)
    finally:
        selector.unregister(proc.stderr)


def terminate_and_collect_stderr(
    proc: subprocess.Popen[Any],
    limit: int = 2048,
    timeout: float = 1.0,
) -> str:
    """Terminate *proc* and return a bounded stderr sample.

    Reading from ``proc.stderr`` directly can block forever while the child is
    still alive. Use ``communicate()`` after terminating the child so GUI
    startup failure handling remains bounded.
    """
    stderr = _read_available_stderr(proc, limit)
    if proc.poll() is None:
        proc.terminate()
        try:
            _, remaining = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, remaining = proc.communicate(timeout=timeout)
        if remaining:
            stderr += remaining
    else:
        _, remaining = proc.communicate(timeout=timeout)
        if remaining:
            stderr += remaining

    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    else:
        text = str(stderr)
    return text[:limit]


def rpc_config_from_settings(namespace: str = "agent") -> Dict[str, Any]:
    """Read RPC connection parameters from ChiSurf editor settings.

    Parameters
    ----------
    namespace : str
        Settings prefix.  ``"agent"`` reads ``agent_chisurf_rpc_*`` and
        ``agent_editor_rpc_*``.  ``"mfdb"`` reads ``mfdb_rpc_*``.
        ``"server"`` reads plain ``cmd_port`` / ``pub_port``.

    Returns
    -------
    dict
        ``{"host": ..., "cmd_port": ..., "pub_port": ...}`` for both
        ``"chisurf"`` and ``"editor"`` keys.

    """
    config: Dict[str, Any] = {
        "chisurf": {"host": "127.0.0.1", "cmd_port": 8765, "pub_port": 8766},
        "editor": {"host": "127.0.0.1", "cmd_port": 8775, "pub_port": 8776},
    }
    try:
        from chisurf.plugins.core.code_editor.settings import get_editor_settings
        settings = get_editor_settings()
        if namespace == "agent":
            config["chisurf"]["host"] = str(settings.get("agent_chisurf_rpc_host", "127.0.0.1"))
            config["chisurf"]["cmd_port"] = int(settings.get("agent_chisurf_rpc_cmd_port", 8765))
            config["chisurf"]["pub_port"] = int(settings.get("agent_chisurf_rpc_pub_port", 8766))
            config["editor"]["host"] = str(settings.get("agent_editor_rpc_host", "127.0.0.1"))
            config["editor"]["cmd_port"] = int(settings.get("agent_editor_rpc_cmd_port", 8775))
            config["editor"]["pub_port"] = int(settings.get("agent_editor_rpc_pub_port", 8776))
        elif namespace == "mfdb":
            config["chisurf"]["host"] = str(settings.get("mfdb_rpc_host", "127.0.0.1"))
            config["chisurf"]["cmd_port"] = int(settings.get("mfdb_rpc_cmd_port", 8765))
            config["chisurf"]["pub_port"] = int(settings.get("mfdb_rpc_pub_port", 8766))
    except Exception:
        _LOG.debug("editor settings not available, using defaults")
    return config


def rpc_is_available(
    host: str,
    cmd_port: int,
    pub_port: Optional[int] = None,
    timeout_ms: int = 500,
) -> bool:
    """Check whether a ZMQ JSON-RPC server is reachable.

    Opens a plain TCP socket to the command port and sends a minimal
    JSON-RPC ping.  If the socket connects and receives a response
    within the timeout, the server is considered available.

    Parameters
    ----------
    host : str
        Server hostname.
    cmd_port : int
        Command (REQ/REP) port.
    pub_port : int, optional
        PUB port (not checked directly).
    timeout_ms : int, default 500
        Connection timeout in milliseconds.

    Returns
    -------
    bool
        ``True`` if the server responds.

    """
    del pub_port
    try:
        import zmq
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.connect(f"tcp://{host}:{cmd_port}")
        request = {
            "jsonrpc": "2.0",
            "method": "meta.ping",
            "params": {},
            "id": 1,
        }
        import json as _json
        sock.send_string(_json.dumps(request))
        reply = sock.recv_string()
        sock.close(linger=0)
        ctx.term()
        parsed = _json.loads(reply)
        return parsed.get("result", {}).get("ok") is True
    except Exception:
        return False


def get_shared_event_bus() -> Any:
    """Return the event bus of the running embedded ChiSurf RPC server.

    Returns ``None`` when no embedded server is running (e.g. in tests
    that connect to an external server, or in headless mode).

    Returns
    -------
    InProcessEventBus or None
        The event bus, or ``None``.
    """
    try:
        import chisurf as _cs
        server = (
            getattr(_cs, "__chisurf_rpc_server__", None)
            or getattr(_cs, "__mfdb_rpc_server__", None)
        )
        if server is not None:
            return getattr(server, "event_bus", None)
    except Exception:
        pass
    return None


def session_state_from_live_chisurf() -> Any:
    """Create a ``SessionState`` that references the live GUI session objects.

    Wraps ``chisurf.fits`` / ``chisurf.imported_datasets`` directly
    (shared list objects — mutations are visible to both).  The
    ``ChiSurfAPI`` also aliases these same lists via its own
    ``_state``, so there is a single logical owner.

    Returns
    -------
    SessionState or None
        A state sharing the session list objects, or ``None`` if
        ``chisurf`` is not importable (headless).

    """
    try:
        import chisurf as _cs
        if not hasattr(_cs, "fits") or not hasattr(_cs, "imported_datasets"):
            return None
        from chisurf.server.session import SessionState
        state = SessionState(
            datasets=_cs.imported_datasets,
            fits=_cs.fits,
        )
        sync_current_fit_uid_from_live_chisurf(state)
        return state
    except Exception:
        return None


def sync_current_fit_uid_from_live_chisurf(state: Any) -> None:
    """Refresh ``state.current_fit_uid`` from the live GUI selection.

    Lookup order:
    1. ``chisurf.current_fit``
    2. ``chisurf.cs.current_fit``
    3. ``chisurf.fits[chisurf.current_fit_idx]``
    4. ``getattr(chisurf.fits, "selected", None)``

    Parameters
    ----------
    state : SessionState
        State to update.

    """
    try:
        import chisurf as _cs
        uid = None
        candidates = [
            getattr(_cs, "current_fit", None),
            getattr(getattr(_cs, "cs", None), "current_fit", None),
        ]
        for candidate in candidates:
            if candidate is not None:
                try:
                    uid = str(getattr(candidate, "unique_identifier", ""))
                except Exception:
                    pass
                if uid:
                    break

        if not uid:
            idx = getattr(_cs, "current_fit_idx", -1)
            if isinstance(idx, int) and idx >= 0 and hasattr(_cs, "fits"):
                try:
                    fit_obj = _cs.fits[idx]
                    uid = str(getattr(fit_obj, "unique_identifier", ""))
                except Exception:
                    pass

        if not uid:
            try:
                selected = getattr(_cs.fits, "selected", None) if hasattr(_cs, "fits") else None
                if selected is not None:
                    uid = str(getattr(selected, "unique_identifier", ""))
            except Exception:
                pass

        if uid:
            state.current_fit_uid = uid
    except Exception:
        pass


def ensure_embedded_chisurf_rpc_server(
    host: str,
    cmd_port: int,
    pub_port: int,
    timeout_s: float = 5.0,
    state: Any = None,
) -> bool:
    """Check and wait for the embedded ChiSurf RPC server to be available.

    Polls ``rpc_is_available`` until the timeout expires.  Designed to
    be called from the agent panel before starting tool execution.

    If no server responds and a *state* is provided, creates one using
    live ``chisurf`` session references.

    Parameters
    ----------
    host : str
        Server hostname.
    cmd_port : int
        Command port.
    pub_port : int
        PUB port.
    timeout_s : float, default 5.0
        Maximum time to wait for the server.
    state : SessionState, optional
        Pre-populated session state to use when starting the server.

    Returns
    -------
    bool
        ``True`` if the server is available within the timeout.

    """
    import time
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if rpc_is_available(host, cmd_port, pub_port, timeout_ms=500):
            return True
    # Optionally auto-start a server with live state
    if state is not None:
        try:
            from chisurf.server.app import ChiSurfServer
            server = ChiSurfServer(
                cmd_port=cmd_port,
                pub_port=pub_port,
                host=host,
                state=state,
            )
            import threading
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            time.sleep(0.3)
            return rpc_is_available(host, cmd_port, pub_port, timeout_ms=2000)
        except Exception:
            pass
    return False

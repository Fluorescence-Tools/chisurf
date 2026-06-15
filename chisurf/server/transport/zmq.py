from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    import zmq

_log = logging.getLogger(__name__)
_SERVER_DISPATCH_CONTEXT = threading.local()


def in_server_dispatch() -> bool:
    """Return whether the current thread is handling a ZMQ RPC request."""
    return bool(getattr(_SERVER_DISPATCH_CONTEXT, "active", False))


class ZmqServer:
    """ZeroMQ REQ/REP + PUB server.

    Listens for JSON-RPC requests on a REQ/REP socket and broadcasts
    events on a PUB socket.

    Parameters
    ----------
    handler : Callable[[Dict[str, Any]], Dict[str, Any]]
        Called with ``(method, params)`` for each incoming request.
        Must return a result dict.
    cmd_port : int
        TCP port for the REQ/REP command socket.
    pub_port : int
        TCP port for the PUB event broadcast socket.
    host : str
        Bind address (default ``"127.0.0.1"``).
    """

    def __init__(
        self,
        handler: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        cmd_port: int = 8765,
        pub_port: int = 8766,
        host: str = "127.0.0.1",
    ):
        """Initialise the ZMQ server.

        Parameters
        ----------
        handler : callable
            Request handler ``(method, params) -> dict``.
        cmd_port : int
            TCP port for the REQ/REP socket.
        pub_port : int
            TCP port for the PUB socket.
        host : str
            Bind address.

        """
        self._handler = handler
        self._cmd_port = cmd_port
        self._pub_port = pub_port
        self._host = host
        if host not in ("127.0.0.1", "localhost", "::1", ""):
            raise ValueError(
                f"Non-loopback host '{host}' requires CURVE/ZAP transport security "
                "which is not yet implemented. Use 127.0.0.1 (loopback) instead."
            )
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ctx: zmq.Context | None = None
        self._rep_socket: zmq.Socket | None = None
        self._pub_socket: zmq.Socket | None = None

    def serve_forever(self) -> None:
        """Run the ZMQ event loop (blocking)."""
        import zmq

        self._ctx = zmq.Context()

        self._rep_socket = self._ctx.socket(zmq.REP)
        self._rep_socket.bind(f"tcp://{self._host}:{self._cmd_port}")

        self._pub_socket = self._ctx.socket(zmq.PUB)
        self._pub_socket.bind(f"tcp://{self._host}:{self._pub_port}")

        self._running = True
        _log.info(
            "ZMQ server listening on tcp://%s:%s (cmd) and tcp://%s:%s (pub)",
            self._host, self._cmd_port, self._host, self._pub_port,
        )

        poller = zmq.Poller()
        poller.register(self._rep_socket, zmq.POLLIN)

        try:
            while self._running:
                try:
                    socks = dict(poller.poll(timeout=500))
                except zmq.ZMQError:
                    if not self._running:
                        break
                    continue

                if self._rep_socket in socks:
                    self._handle_one()
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Signal the server loop to stop."""
        self._running = False

    def broadcast_event(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish an event on the PUB socket."""
        if self._pub_socket is None:
            return
        try:
            data = json.dumps(payload)
            self._pub_socket.send_multipart([topic.encode("utf-8"), data.encode("utf-8")])
        except Exception:
            _log.exception("Failed to broadcast event")

    def _handle_one(self) -> None:
        """Receive and respond to a single JSON-RPC request."""
        try:
            raw = self._rep_socket.recv_json()
        except Exception:
            self._rep_socket.send_json({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None})
            return

        if not isinstance(raw, dict) or not isinstance(raw.get("method"), str) or not raw.get("method"):
            req_id = raw.get("id") if isinstance(raw, dict) else None
            self._rep_socket.send_json({
                "jsonrpc": "2.0",
                "error": {"code": -32600, "message": "Invalid Request"},
                "id": req_id,
            })
            return

        if raw.get("jsonrpc") != "2.0":
            _log.warning("Request missing 'jsonrpc': '2.0' — treating leniently")

        method = raw.get("method", "")
        params = raw.get("params", {}) or {}
        req_id = raw.get("id")

        try:
            previous = in_server_dispatch()
            _SERVER_DISPATCH_CONTEXT.active = True
            try:
                result = self._handler(method, params)
            finally:
                _SERVER_DISPATCH_CONTEXT.active = previous
            self._rep_socket.send_json({
                "jsonrpc": "2.0",
                "result": result,
                "id": req_id,
            })
        except Exception as e:
            self._rep_socket.send_json({
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": req_id,
            })

    def _cleanup(self) -> None:
        """Close ZMQ sockets and terminate the context."""
        for sock in (self._rep_socket, self._pub_socket):
            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass


class ZmqClient:
    """ZMQ REQ client for talking to a ZmqServer.

    Parameters
    ----------
    cmd_port : int
        TCP port of the server's REQ/REP socket.
    pub_port : int
        TCP port of the server's PUB socket (optional, for event subscription).
    host : str
        Server address (default ``"127.0.0.1"``).
    timeout_ms : int
        Send/receive timeout in milliseconds.
    """

    def __init__(
        self,
        cmd_port: int = 8765,
        pub_port: int = 8766,
        host: str = "127.0.0.1",
        timeout_ms: int = 5000,
    ):
        """Initialise the ZMQ client.

        Parameters
        ----------
        cmd_port : int
            TCP port of the server's REQ/REP socket.
        pub_port : int
            TCP port of the server's PUB socket.
        host : str
            Server address.
        timeout_ms : int
            Send/receive timeout in milliseconds.

        """
        self._cmd_port = cmd_port
        self._pub_port = pub_port
        self._host = host
        self._timeout_ms = timeout_ms
        self._ctx: zmq.Context | None = None
        self._req_socket: zmq.Socket | None = None
        self._sub_socket: zmq.Socket | None = None
        self._request_id: int = 0
        # Subscriber dispatch is driven by drain() from the owning event loop.
        self._subscribers: Dict[str, list[Callable]] = {}
        self._call_lock = threading.RLock()

    def connect(self) -> None:
        """Create ZMQ sockets and connect to the server."""
        import zmq
        if self._ctx is None:
            self._ctx = zmq.Context()
        self._req_socket = self._ctx.socket(zmq.REQ)
        self._req_socket.setsockopt(zmq.LINGER, 0)
        self._req_socket.connect(f"tcp://{self._host}:{self._cmd_port}")

    def close(self) -> None:
        """Close all ZMQ sockets and release resources."""
        with self._call_lock:
            if self._req_socket is not None:
                try:
                    self._req_socket.close(linger=0)
                except Exception:
                    pass
                self._req_socket = None
            if self._sub_socket is not None:
                try:
                    self._sub_socket.close(linger=0)
                except Exception:
                    pass
                self._sub_socket = None
            if self._ctx is not None:
                try:
                    self._ctx.term()
                except Exception:
                    pass
                self._ctx = None
            self._subscribers.clear()

    def _reset_socket(self) -> None:
        """Tear down the REQ socket so the next call() creates a fresh one.

        After a timeout or any error that leaves the REQ socket in
        "reply pending" state, ZMQ's strict state machine rejects
        further send_json() calls.  Closing the socket and clearing
        the reference forces connect() to start from a clean state.
        The shared context is intentionally kept alive because the
        subscriber socket may still be using it.
        """
        if self._req_socket is not None:
            try:
                self._req_socket.close(linger=0)
            except Exception:
                pass
            self._req_socket = None

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for the response."""
        with self._call_lock:
            if self._req_socket is None:
                self.connect()

            import zmq
            self._request_id += 1
            msg = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
                "id": self._request_id,
            }

            try:
                self._req_socket.send_json(msg)
            except zmq.ZMQError as e:
                self._reset_socket()
                return {"ok": False, "error": f"send failed: {e}"}

            poller = zmq.Poller()
            poller.register(self._req_socket, zmq.POLLIN)
            try:
                socks = dict(poller.poll(timeout=self._timeout_ms))
            except zmq.ZMQError as e:
                self._reset_socket()
                return {"ok": False, "error": f"poll failed: {e}"}

            if self._req_socket not in socks:
                self._reset_socket()
                return {"ok": False, "error": "timeout: no response within {}ms".format(self._timeout_ms)}

            try:
                return self._req_socket.recv_json()
            except zmq.ZMQError as e:
                self._reset_socket()
                return {"ok": False, "error": f"recv failed: {e}"}

    def subscribe(self, topic: str = "", callback: Optional[Callable] = None) -> Any:
        """Register a subscriber for *topic*.

        ZMQ uses prefix matching, so ``"dataset."`` matches
        ``"dataset.added"``, ``"dataset.removed"`` etc.  The *callback*
        is invoked only by :meth:`drain`, which should be called from the
        event-loop thread that owns the callback targets.

        If *callback* is ``None``, the raw SUB socket is returned for
        custom use.
        """
        import zmq
        if self._ctx is None:
            self._ctx = zmq.Context()
        if self._sub_socket is None:
            self._sub_socket = self._ctx.socket(zmq.SUB)
            self._sub_socket.setsockopt(zmq.LINGER, 0)
            self._sub_socket.connect(f"tcp://{self._host}:{self._pub_port}")

        # Strip trailing glob characters — ZMQ uses literal prefix matching
        zmq_topic = topic.rstrip("*?")
        self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, zmq_topic)

        if callback is not None:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(callback)

        return self._sub_socket

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Remove a previously registered subscriber callback.

        Parameters
        ----------
        topic : str
            The topic the callback was registered for.
        callback : Callable
            The callback to remove.
        """
        if topic in self._subscribers:
            self._subscribers[topic] = [cb for cb in self._subscribers[topic] if cb is not callback]
            if not self._subscribers[topic]:
                del self._subscribers[topic]
                zmq_topic = topic.rstrip("*?")
                if self._sub_socket is not None:
                    try:
                        self._sub_socket.setsockopt_string(zmq.UNSUBSCRIBE, zmq_topic)
                    except Exception:
                        pass

    def drain(self) -> None:
        """Process all buffered subscriber events.

        Must be called from the thread that owns the registered
        callbacks (typically the main thread).  Topic dispatch uses
        the same prefix-matching semantics as ZMQ: *pattern* matches
        *topic* when ``topic.startswith(pattern)`` after stripping
        trailing glob characters (``*?``) from *pattern*.

        Callers who want glob/fnmatch-style matching should already
        have received the event from ZMQ's own prefix filter; this
        method replicates that same prefix logic for the final
        dispatch.
        """
        if self._sub_socket is None:
            return
        import zmq

        while True:
            try:
                topic_bytes, data = self._sub_socket.recv_multipart(flags=zmq.NOBLOCK)
                topic = topic_bytes.decode("utf-8") if isinstance(topic_bytes, bytes) else str(topic_bytes)
                payload = json.loads(data.decode("utf-8"))
                for pattern, cbs in list(self._subscribers.items()):
                    # Use same prefix-matching semantics as ZMQ SUBSCRIBE
                    prefix = pattern.rstrip("*?")
                    if topic.startswith(prefix):
                        for cb in cbs:
                            try:
                                cb(payload)
                            except Exception:
                                _log.exception(
                                    "Subscriber callback error for topic '%s' (pattern '%s')",
                                    topic, pattern,
                                )
            except zmq.Again:
                break
            except zmq.ZMQError:
                break
            except Exception:
                _log.exception("Error draining subscriber event")
                break

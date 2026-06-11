import base64
import json
import logging
import sqlite3
import threading
import traceback
from typing import Optional

import numpy as np
import zmq

from .repository import FluorophoreDatabase

logger = logging.getLogger(__name__)


def _encode(obj):
    """Encode numpy types and sqlite3.Row to JSON-safe Python objects."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "dtype": str(obj.dtype), "data": obj.tolist()}
    if isinstance(obj, bytes):
        return {"__bytes__": True, "data": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (sqlite3.Row,)):
        return {k: _encode(v) for k, v in dict(obj).items()}
    if hasattr(obj, "keys"):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    return obj


def _decode(obj):
    """Decode JSON-encoded values back to numpy arrays."""
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.array(obj["data"], dtype=obj["dtype"])
    if isinstance(obj, dict) and obj.get("__bytes__"):
        return base64.b64decode(obj["data"])
    if isinstance(obj, dict):
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_decode(v) for v in obj]
    return obj


class FlrDatabaseServer:
    """ZMQ server wrapping a FluorophoreDatabase instance.

    Runs a REP socket in a daemon thread.  Handles method dispatch for
    all public methods of FluorophoreDatabase.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        bind: str = "tcp://127.0.0.1:5559",
        *,
        database: Optional[FluorophoreDatabase] = None,
    ):
        self.db = database or FluorophoreDatabase(db_path)
        self._bind = bind
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    @property
    def endpoint(self) -> str:
        return self._bind

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._running.wait(timeout=5)

    def stop(self):
        self._running.clear()

    def _serve(self):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._bind)
        self._running.set()
        logger.info("FlrDatabaseServer listening on %s", self._bind)
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        try:
            while self._running.is_set():
                try:
                    socks = dict(poller.poll(500))
                    if self._socket not in socks:
                        continue
                    raw = self._socket.recv()
                    response = self._handle(raw)
                    self._socket.send(response)
                except zmq.ZMQError:
                    break
        finally:
            self._socket.close(linger=0)
            self._context.term()

    def _handle(self, raw: bytes) -> bytes:
        try:
            msg = json.loads(raw.decode("utf-8"))
            method_name = msg.get("method", "")
            args = _decode(msg.get("args", []))
            kwargs = _decode(msg.get("kwargs", {}))
            method = getattr(self.db, method_name, None)
            if method is None:
                raise AttributeError(f"Unknown method: {method_name}")
            result = method(*args, **kwargs)
            result = _encode(result)
            return json.dumps({"result": result, "error": None}).encode("utf-8")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("FlrDatabaseServer error: %s\n%s", exc, tb)
            return json.dumps({"result": None, "error": str(exc)}).encode("utf-8")

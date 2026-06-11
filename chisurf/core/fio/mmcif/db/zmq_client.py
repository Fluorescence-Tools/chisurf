import json
import logging
from types import MethodType
from typing import Any, Optional

import numpy as np
import zmq

from .zmq_server import _encode, _decode

logger = logging.getLogger(__name__)


def _to_json_safe(obj):
    """Convert object to JSON-safe form for sending over ZMQ."""
    return _encode(obj)


def _from_json_safe(obj):
    """Restore JSON-safe form back to original types."""
    return _decode(obj)


class FlrDatabaseClient:
    """ZMQ client proxy for FluorophoreDatabase.

    Usage::

        client = FlrDatabaseClient("tcp://127.0.0.1:5559")
        client.add_probe_type("organic_dye", "Organic dye")
        pid = client.add_probe("Alexa488", 1)
        meta = client.get_analysis_metadata("analysis_1")
    """

    def __init__(self, connect: str = "tcp://127.0.0.1:5559", timeout_ms: int = 30000):
        self._connect = connect
        self._timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(connect)

    def close(self):
        self._socket.close(linger=0)
        self._context.term()

    def _call(self, method: str, *args, **kwargs) -> Any:
        payload = json.dumps({
            "method": method,
            "args": _to_json_safe(args),
            "kwargs": _to_json_safe(kwargs),
        }).encode("utf-8")
        self._socket.send(payload)
        raw = self._socket.recv()
        msg = json.loads(raw.decode("utf-8"))
        if msg.get("error"):
            raise RuntimeError(msg["error"])
        return _from_json_safe(msg.get("result"))

    def __getattr__(self, name: str):
        # Forward any unknown attribute access as a remote method call.
        # This lets us use the same API as FluorophoreDatabase without
        # listing every method explicitly.
        if name.startswith("_"):
            raise AttributeError(name)

        def _remote_method(*args, **kwargs):
            return self._call(name, *args, **kwargs)

        return _remote_method

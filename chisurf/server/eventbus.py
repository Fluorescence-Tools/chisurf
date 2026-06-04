from __future__ import annotations

import fnmatch
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional


class EventBus:
    """Abstract event bus interface.

    Concrete implementations must provide ``subscribe``, ``unsubscribe``,
    ``publish``, and ``clear``.
    """

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> str:
        """Register a handler for *topic*.

        Parameters
        ----------
        topic : str
            Topic string. May contain ``*`` / ``?`` wildcards.
        handler : callable
            Callback receiving the event dict.

        Returns
        -------
        str
            Subscription token (used by :meth:`unsubscribe`).

        """
        raise NotImplementedError

    def unsubscribe(self, token: str) -> None:
        """Remove a subscription by token.

        Parameters
        ----------
        token : str
            Token returned by :meth:`subscribe`.

        """
        raise NotImplementedError

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish an event on *topic*.

        Parameters
        ----------
        topic : str
            Event topic.
        payload : dict
            Event payload.

        """
        raise NotImplementedError

    def clear(self) -> None:
        """Remove all subscriptions."""
        raise NotImplementedError


class InProcessEventBus(EventBus):
    """In-memory pub/sub event bus for single-process use.

    Supports exact topic matching and ``fnmatch``-style wildcard patterns
    (e.g. ``"fit.*"`` matches ``"fit.added"``, ``"fit.removed"``).
    """

    def __init__(self):
        """Initialise an empty in-process event bus."""
        self._lock = threading.RLock()
        self._subscriptions: Dict[str, List[tuple[str, Callable]]] = {}
        self._pattern_subscriptions: List[tuple[str, str, Callable]] = []

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> str:
        """Register a handler for *topic*.

        Parameters
        ----------
        topic : str
            Topic string. May contain ``*`` / ``?`` wildcards.
        handler : callable
            Callback receiving the event dict.

        Returns
        -------
        str
            Subscription token.

        """
        token = str(uuid.uuid4())
        with self._lock:
            if "*" in topic or "?" in topic:
                self._pattern_subscriptions.append((topic, token, handler))
            else:
                self._subscriptions.setdefault(topic, []).append((token, handler))
        return token

    def unsubscribe(self, token: str) -> None:
        """Remove a subscription by token.

        Parameters
        ----------
        token : str
            Token returned by :meth:`subscribe`.

        """
        with self._lock:
            for topic in list(self._subscriptions):
                self._subscriptions[topic] = [
                    (t, h) for t, h in self._subscriptions[topic] if t != token
                ]
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]
            self._pattern_subscriptions = [
                (p, t, h) for p, t, h in self._pattern_subscriptions if t != token
            ]

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish an event on *topic*.

        Parameters
        ----------
        topic : str
            Event topic.
        payload : dict
            Event payload (augmented with ``topic`` and ``timestamp``).

        """
        event = dict(payload)
        event.setdefault("topic", topic)
        event.setdefault("timestamp", time.time())
        with self._lock:
            handlers = list(self._subscriptions.get(topic, []))
            pattern_handlers = [
                (t, h)
                for pattern, t, h in self._pattern_subscriptions
                if fnmatch.fnmatch(topic, pattern)
            ]
        for _token, handler in handlers:
            self._safe_call(handler, event)
        for _token, handler in pattern_handlers:
            self._safe_call(handler, event)

    def clear(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._subscriptions.clear()
            self._pattern_subscriptions.clear()

    @staticmethod
    def _safe_call(handler: Callable, event: Dict[str, Any]) -> None:
        """Invoke *handler* and log any exception without propagating it.

        Parameters
        ----------
        handler : callable
            Callback to invoke.
        event : dict
            Event payload forwarded to the handler.

        """
        try:
            handler(event)
        except Exception:
            import logging
            logging.exception("EventBus handler error")

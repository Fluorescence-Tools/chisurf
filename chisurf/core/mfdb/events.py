"""In-process, post-commit event bus for MFDB (PRD-21 Task 3).

A minimal publish/subscribe bus so registration and state changes can drive reactive
behaviour (audit, lifecycle, cache invalidation, downstream triggers) without polling
— the data-side analog of chinet's reactive ports.

Contract (PRD-21):

- **Post-commit.** Publishers fire *after* the registering transaction commits, so a
  handler never runs against uncommitted state.
- **Best-effort, isolated.** A failing handler is logged and skipped; it never
  propagates into — and so never breaks — the registration that published the event.
- **Synchronous, ordered.** Handlers run in subscription order on the publishing
  thread (name-specific subscribers first, then global subscribers).

Event vocabulary (names are stable strings):

- ``artifact.registered`` — an artifact was registered. Payload: ``artifact_id``,
  ``kind``, ``operation_type``, ``operation_id``, ``sample_id``.
- ``operation.succeeded`` — an operation node was recorded. Payload: ``operation_id``,
  ``operation_type``, ``inputs``, ``outputs``.
- ``state.changed`` — a lifecycle state transition (PRD-12). Payload: ``entity_type``,
  ``entity_id``, ``from_state``, ``to_state``.
- ``calibration.updated`` — a calibration value changed (PRD-05). Payload:
  ``setup_id``, ``calibration_id``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

# -- event name constants ----------------------------------------------------

EVENT_ARTIFACT_REGISTERED = "artifact.registered"
EVENT_OPERATION_SUCCEEDED = "operation.succeeded"
EVENT_STATE_CHANGED = "state.changed"
EVENT_CALIBRATION_UPDATED = "calibration.updated"


@dataclass(frozen=True)
class Event:
    """A published event: a stable ``name`` and a JSON-able ``payload``."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __getitem__(self, key: str) -> Any:  # convenience: event["artifact_id"]
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


Handler = Callable[[Event], None]


class EventBus:
    """A synchronous, best-effort publish/subscribe bus."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = {}
        self._global: list[Handler] = []

    def subscribe(self, name: str, handler: Handler) -> Handler:
        """Subscribe ``handler`` to events named ``name``; returns the handler."""
        self._subscribers.setdefault(name, []).append(handler)
        return handler

    def subscribe_all(self, handler: Handler) -> Handler:
        """Subscribe ``handler`` to *every* event; returns the handler."""
        self._global.append(handler)
        return handler

    def unsubscribe(self, name: str, handler: Handler) -> None:
        handlers = self._subscribers.get(name)
        if handlers and handler in handlers:
            handlers.remove(handler)

    def unsubscribe_all(self, handler: Handler) -> None:
        if handler in self._global:
            self._global.remove(handler)

    def clear(self) -> None:
        """Drop all subscribers (primarily for tests)."""
        self._subscribers.clear()
        self._global.clear()

    def publish(self, name: str, **payload: Any) -> Event:
        """Build and dispatch an event; never raises (best-effort).

        Returns the :class:`Event` that was dispatched so callers/tests can inspect
        it. Handler exceptions are logged and swallowed so a subscriber can never
        break the publisher (i.e. the registration transaction).
        """
        event = Event(
            name=name,
            payload=dict(payload),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        for handler in list(self._subscribers.get(name, ())) + list(self._global):
            try:
                handler(event)
            except Exception:  # pragma: no cover - defensive; logged not raised
                logger.exception("event handler failed for %s", name)
        return event


# -- process-default bus -----------------------------------------------------

_DEFAULT_BUS = EventBus()


def get_event_bus() -> EventBus:
    """Return the process-default event bus."""
    return _DEFAULT_BUS


def publish(name: str, **payload: Any) -> Event:
    """Publish on the default bus (best-effort; never raises)."""
    return _DEFAULT_BUS.publish(name, **payload)


def subscribe(name: str, handler: Handler) -> Handler:
    """Subscribe on the default bus."""
    return _DEFAULT_BUS.subscribe(name, handler)


def subscribe_all(handler: Handler) -> Handler:
    """Subscribe to every event on the default bus."""
    return _DEFAULT_BUS.subscribe_all(handler)


def audit_log_subscriber(event: Event) -> None:
    """A ready-made subscriber that logs every event at INFO (audit trail).

    Attach with ``subscribe_all(audit_log_subscriber)``. Demonstrates the DoD's
    "at least one subscriber consumes events" without imposing a side effect by
    default (callers opt in).
    """
    logger.info("mfdb event: %s %s", event.name, event.payload)

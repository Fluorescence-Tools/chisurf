from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

from chisurf import typing


class OperationHistory:
    """Append-only operation history for traceable user actions."""

    DEFAULT_CHECKPOINT_INTERVAL = 50

    def __init__(self, checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL):
        self._events: typing.List[typing.Dict[str, typing.Any]] = []
        self._lock = threading.RLock()
        self._subscribers: typing.List[typing.Callable[[typing.Dict[str, typing.Any]], None]] = []
        self._checkpoints: typing.Dict[int, typing.Dict[str, typing.Any]] = {}
        self._checkpoint_interval = max(1, int(checkpoint_interval))
        self._checkpoint_capture_fn: typing.Optional[typing.Callable[[], typing.Dict[str, typing.Any]]] = None

    def record(
            self,
            action_type: str,
            summary: str,
            payload: typing.Optional[typing.Dict[str, typing.Any]] = None,
            source_uid: typing.Optional[str] = None,
            target_uid: typing.Optional[str] = None,
    ) -> typing.Dict[str, typing.Any]:
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_type": str(action_type),
            "source_uid": None if source_uid is None else str(source_uid),
            "target_uid": None if target_uid is None else str(target_uid),
            "payload": payload or {},
            "summary": str(summary),
        }
        with self._lock:
            self._events.append(event)
            event_index = len(self._events) - 1
            subscribers = list(self._subscribers)
        self._maybe_create_checkpoint(event_index)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                pass
        self._emit_log(event)
        return event

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._checkpoints.clear()

    def subscribe(self, callback: typing.Callable[[typing.Dict[str, typing.Any]], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: typing.Callable[[typing.Dict[str, typing.Any]], None]) -> None:
        with self._lock:
            self._subscribers = [cb for cb in self._subscribers if cb is not callback]

    def list_events(self) -> typing.List[typing.Dict[str, typing.Any]]:
        with self._lock:
            return list(self._events)

    def tail(self, n: int = 100) -> typing.List[typing.Dict[str, typing.Any]]:
        if n <= 0:
            return []
        with self._lock:
            return list(self._events[-n:])

    def save_jsonl(self, filename: typing.Union[str, Path]) -> Path:
        path = Path(filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = list(self._events)
        with path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, sort_keys=True))
                fp.write("\n")
        return path

    def load_jsonl(self, filename: typing.Union[str, Path], replace: bool = True) -> typing.List[typing.Dict[str, typing.Any]]:
        path = Path(filename).resolve()
        if not path.exists():
            raise FileNotFoundError(f"History file not found: {path}")
        rows: typing.List[typing.Dict[str, typing.Any]] = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
        with self._lock:
            if replace:
                self._events = rows
                self._checkpoints.clear()
            else:
                self._events.extend(rows)
        return rows

    def replay(
            self,
            handlers: typing.Dict[str, typing.Callable[[typing.Dict[str, typing.Any]], None]],
            events: typing.Optional[typing.Iterable[typing.Dict[str, typing.Any]]] = None,
            stop_on_error: bool = True,
    ) -> typing.Dict[str, typing.Any]:
        """Replay history events using action-type handlers.

        Parameters
        ----------
        handlers:
            Mapping from ``action_type`` to a callable taking one event dict.
        events:
            Optional iterable of events. If omitted, uses the current in-memory
            history list.
        stop_on_error:
            If ``True`` (default), abort replay on first handler error.

        Returns
        -------
        dict
            Replay report with counts and collected errors.
        """

        if events is None:
            events_list = self.list_events()
        else:
            events_list = list(events)

        report = {
            "total": len(events_list),
            "replayed": 0,
            "skipped": 0,
            "errors": [],
        }

        for idx, event in enumerate(events_list):
            action_type = str(event.get("action_type", ""))
            handler = handlers.get(action_type)
            if handler is None:
                report["skipped"] += 1
                continue
            try:
                handler(event)
                report["replayed"] += 1
            except Exception as exc:
                report["errors"].append({
                    "index": idx,
                    "action_type": action_type,
                    "event_id": event.get("event_id"),
                    "error": str(exc),
                })
                if stop_on_error:
                    break

        return report

    @staticmethod
    def _emit_log(event: typing.Dict[str, typing.Any]) -> None:
        line = f"# HIST {event.get('action_type', '?')}: {event.get('summary', '')}"
        try:
            import chisurf
            log_fn = getattr(chisurf, "log", None)
            if callable(log_fn):
                log_fn(line)
            else:
                chisurf.logging.info(line)
        except Exception:
            pass

    def set_checkpoint_capture(
            self,
            capture_fn: typing.Optional[typing.Callable[[], typing.Dict[str, typing.Any]]],
    ) -> None:
        """Set the function used to capture domain state for checkpoints.

        The capture function should return a JSON-serializable dict representing
        the current scientific state (datasets, fits, parameters, links, etc.).
        """
        with self._lock:
            self._checkpoint_capture_fn = capture_fn

    def create_checkpoint(self, event_index: int) -> bool:
        """Create a checkpoint at the given event index.

        Returns True if checkpoint was created, False if capture function not set
        or capture failed.
        """
        if self._checkpoint_capture_fn is None:
            return False
        try:
            snapshot = self._checkpoint_capture_fn()
            if not isinstance(snapshot, dict):
                return False
            with self._lock:
                self._checkpoints[event_index] = {
                    "event_index": event_index,
                    "snapshot": snapshot,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            return True
        except Exception:
            return False

    def get_checkpoint_before(self, event_index: int) -> typing.Optional[typing.Dict[str, typing.Any]]:
        """Get the nearest checkpoint at or before the given event index.

        Returns None if no checkpoint exists before the index.
        """
        with self._lock:
            if not self._checkpoints:
                return None
            candidates = [idx for idx in self._checkpoints.keys() if idx <= event_index]
            if not candidates:
                return None
            nearest = max(candidates)
            return self._checkpoints[nearest]

    def clear_checkpoints(self) -> None:
        """Remove all checkpoints."""
        with self._lock:
            self._checkpoints.clear()

    def checkpoint_count(self) -> int:
        """Return the number of stored checkpoints."""
        with self._lock:
            return len(self._checkpoints)

    def _maybe_create_checkpoint(self, event_index: int) -> None:
        """Create a checkpoint if the interval has been reached."""
        if self._checkpoint_interval <= 0:
            return
        if self._checkpoint_capture_fn is None:
            return
        if event_index > 0 and event_index % self._checkpoint_interval == 0:
            self.create_checkpoint(event_index)

    def get_events_from_checkpoint(
            self,
            target_event_index: int,
    ) -> typing.Tuple[typing.Optional[typing.Dict[str, typing.Any]], typing.List[typing.Dict[str, typing.Any]]]:
        """Get snapshot and events needed to reach target_event_index.

        Returns a tuple of (checkpoint_snapshot, events_to_replay).
        If no checkpoint exists before target, snapshot is None and all events
        up to target are returned.
        """
        with self._lock:
            checkpoint = self.get_checkpoint_before(target_event_index)
            if checkpoint is None:
                events = list(self._events[:target_event_index + 1])
                return None, events
            start_index = checkpoint["event_index"] + 1
            events = list(self._events[start_index:target_event_index + 1])
            return checkpoint["snapshot"], events

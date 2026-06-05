from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import uuid

from chisurf import typing


class OperationHistory:
    """Append-only operation history for traceable user actions."""

    # Version management
    HISTORY_VERSION = "1.0"  # Current history format version
    SUPPORTED_VERSIONS = ["1.0"]  # Versions we can read
    
    DEFAULT_CHECKPOINT_INTERVAL = 50

    def __init__(self, checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL):
        self._events: typing.List[typing.Dict[str, typing.Any]] = []
        self._lock = threading.RLock()
        self._subscribers: typing.List[typing.Callable[[typing.Dict[str, typing.Any]], None]] = []
        self._checkpoints: typing.Dict[int, typing.Dict[str, typing.Any]] = {}
        self._checkpoint_interval = max(1, int(checkpoint_interval))
        self._checkpoint_capture_fn: typing.Optional[typing.Callable[[], typing.Dict[str, typing.Any]]] = None
        self._recording_suppressed = False
        # Memory management settings
        self._max_events = 10000  # Maximum number of events to keep in memory
        self._auto_compact_threshold = 5000  # Compact when exceeding this number of events

    def record(
            self,
            action_type: str,
            summary: str,
            payload: typing.Optional[typing.Dict[str, typing.Any]] = None,
            source_uid: typing.Optional[str] = None,
            target_uid: typing.Optional[str] = None,
    ) -> typing.Dict[str, typing.Any]:
        with self._lock:
            if self._recording_suppressed:
                return {}
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

    from contextlib import contextmanager
    @contextmanager
    def suppress_recording(self):
        """Context manager to temporarily disable operation recording."""
        was_suppressed = self._recording_suppressed
        self._recording_suppressed = True
        try:
            yield
        finally:
            self._recording_suppressed = was_suppressed

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

    def save_jsonl(self, filename: typing.Union[str, Path], include_metadata: bool = True) -> Path:
        """Save history to JSONL file with optional metadata header."""
        path = Path(filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            rows = list(self._events)
            
            # Add metadata header with version information
            with path.open("w", encoding="utf-8") as fp:
                if include_metadata:
                    metadata = {
                        "history_version": self.HISTORY_VERSION,
                        "event_count": len(rows),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "checkpoint_count": len(self._checkpoints)
                    }
                    fp.write("# CHISURF HISTORY METADATA: " + json.dumps(metadata, sort_keys=True) + "\n")
                
                # Write events
                for row in rows:
                    fp.write(json.dumps(row, sort_keys=True) + "\n")
        
        return path

    def load_jsonl(self, filename: typing.Union[str, Path], replace: bool = True) -> typing.Dict[str, typing.Any]:
        """Load history from JSONL file with version validation and integrity checking."""
        path = Path(filename).resolve()
        if not path.exists():
            raise FileNotFoundError(f"History file not found: {path}")
        
        result = {
            "success": False,
            "loaded_events": 0,
            "file_version": None,
            "compatibility": "unknown",
            "errors": []
        }
        
        rows: typing.List[typing.Dict[str, typing.Any]] = []
        file_version = None
        
        try:
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    text = line.strip()
                    if not text:
                        continue
                    
                    # Check for metadata header
                    if text.startswith("# CHISURF HISTORY METADATA: "):
                        try:
                            metadata_json = text[len("# CHISURF HISTORY METADATA: "):]
                            metadata = json.loads(metadata_json)
                            file_version = metadata.get("history_version", "unknown")
                            result["file_version"] = file_version
                            
                            # Check version compatibility
                            if file_version not in self.SUPPORTED_VERSIONS:
                                result["compatibility"] = "incompatible"
                                result["errors"].append(f"Unsupported history version: {file_version}")
                            else:
                                result["compatibility"] = "compatible"
                        except json.JSONDecodeError:
                            result["errors"].append("Invalid metadata format")
                            continue
                    else:
                        # Regular event line
                        try:
                            event = json.loads(text)
                            rows.append(event)
                        except json.JSONDecodeError as e:
                            result["errors"].append(f"Invalid JSON at line: {str(e)}")
            
            # Validate all loaded events
            integrity_report = self.validate_history_integrity()
            if integrity_report["corruption_detected"]:
                result["errors"].append(f"History corruption detected: {len(integrity_report['invalid_events'])} invalid events")
                
                # Attempt automatic repair
                if integrity_report["invalid_events"]:
                    repair_report = self.repair_history()
                    if repair_report["repair_successful"]:
                        result["errors"].append(f"Automatically repaired: removed {repair_report['events_removed']} invalid events")
                        rows = list(self._events)  # Use repaired events
                    else:
                        result["errors"].append("Automatic repair failed")
            
            # Load events into history
            with self._lock:
                if replace:
                    self._events = rows
                    self._checkpoints.clear()
                else:
                    self._events.extend(rows)
            
            result["success"] = True
            result["loaded_events"] = len(rows)
            
        except Exception as e:
            result["errors"].append(f"Load failed: {str(e)}")
            # Don't leave history in partially loaded state
            if replace:
                with self._lock:
                    self._events = []
                    self._checkpoints.clear()
        
        return result

    def create_backup(self, backup_dir: typing.Optional[typing.Union[str, Path]] = None) -> typing.Dict[str, typing.Any]:
        """Create a backup of the current history."""
        result = {
            "success": False,
            "backup_path": None,
            "error": None
        }
        
        try:
            if backup_dir is None:
                backup_dir = Path("history_backups")
            else:
                backup_dir = Path(backup_dir) if isinstance(backup_dir, str) else backup_dir
            
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped backup filename
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_filename = f"history_backup_{timestamp}_{len(self._events)}_events.jsonl"
            backup_path = backup_dir / backup_filename
            
            # Save with full metadata
            self.save_jsonl(backup_path, include_metadata=True)
            
            result["success"] = True
            result["backup_path"] = str(backup_path)
            result["event_count"] = len(self._events)
            result["timestamp"] = timestamp
            
        except Exception as e:
            result["error"] = str(e)
        
        return result

    def restore_from_backup(self, backup_path: typing.Union[str, Path]) -> typing.Dict[str, typing.Any]:
        """Restore history from a backup file."""
        return self.load_jsonl(backup_path, replace=True)

    def get_history_stats(self) -> typing.Dict[str, typing.Any]:
        """Get statistics about the current history."""
        with self._lock:
            return {
                "event_count": len(self._events),
                "checkpoint_count": len(self._checkpoints),
                "oldest_event": self._events[0]["timestamp"] if self._events else None,
                "newest_event": self._events[-1]["timestamp"] if self._events else None,
                "action_types": list(set(event["action_type"] for event in self._events)) if self._events else []
            }

    def set_memory_limits(self, max_events: int = 10000, auto_compact_threshold: int = 5000) -> None:
        """Set memory management limits for history."""
        with self._lock:
            self._max_events = max(100, int(max_events))  # Minimum 100 events
            self._auto_compact_threshold = max(50, int(auto_compact_threshold))  # Minimum 50 events

    def get_memory_limits(self) -> typing.Dict[str, typing.Any]:
        """Get current memory management limits."""
        with self._lock:
            return {
                "max_events": self._max_events,
                "auto_compact_threshold": self._auto_compact_threshold,
                "current_event_count": len(self._events)
            }

    def compact_history(self, keep_recent: int = 500) -> typing.Dict[str, typing.Any]:
        """Compact history by removing older events while keeping recent ones."""
        report = {
            "events_before": 0,
            "events_after": 0,
            "events_removed": 0,
            "compaction_successful": False
        }
        
        with self._lock:
            report["events_before"] = len(self._events)
            
            if len(self._events) <= keep_recent:
                report["events_after"] = len(self._events)
                report["compaction_successful"] = False
                return report
            
            # Keep the most recent events
            keep_recent = max(10, min(keep_recent, len(self._events) - 1))  # Ensure we keep at least 10 events
            compacted_events = self._events[-keep_recent:]
            
            report["events_removed"] = len(self._events) - len(compacted_events)
            self._events = compacted_events
            
            # Also clean up checkpoints that are no longer relevant
            relevant_checkpoints = {
                idx: cp for idx, cp in self._checkpoints.items()
                if idx >= len(self._events) - keep_recent
            }
            report["checkpoints_removed"] = len(self._checkpoints) - len(relevant_checkpoints)
            self._checkpoints = relevant_checkpoints
            
            report["events_after"] = len(self._events)
            report["compaction_successful"] = True
        
        return report

    def auto_compact_if_needed(self) -> typing.Dict[str, typing.Any]:
        """Automatically compact history if it exceeds the auto-compact threshold."""
        with self._lock:
            if len(self._events) <= self._auto_compact_threshold:
                return {
                    "compaction_performed": False,
                    "current_event_count": len(self._events),
                    "threshold": self._auto_compact_threshold
                }
            
            # Perform compaction (keep half of auto_compact_threshold)
            keep_recent = self._auto_compact_threshold // 2
            return self.compact_history(keep_recent)

    def get_estimated_memory_usage(self) -> typing.Dict[str, typing.Any]:
        """Estimate memory usage of the history."""
        import sys
        
        with self._lock:
            # Estimate event sizes
            if self._events:
                sample_event = self._events[0]
                approx_event_size = sys.getsizeof(str(sample_event))
                estimated_events_size = len(self._events) * approx_event_size
            else:
                approx_event_size = 0
                estimated_events_size = 0
            
            # Estimate checkpoint sizes
            if self._checkpoints:
                sample_checkpoint = next(iter(self._checkpoints.values()))
                approx_checkpoint_size = sys.getsizeof(str(sample_checkpoint))
                estimated_checkpoints_size = len(self._checkpoints) * approx_checkpoint_size
            else:
                approx_checkpoint_size = 0
                estimated_checkpoints_size = 0
            
            return {
                "event_count": len(self._events),
                "approx_event_size_bytes": approx_event_size,
                "estimated_events_memory_bytes": estimated_events_size,
                "checkpoint_count": len(self._checkpoints),
                "approx_checkpoint_size_bytes": approx_checkpoint_size,
                "estimated_checkpoints_memory_bytes": estimated_checkpoints_size,
                "total_estimated_memory_bytes": estimated_events_size + estimated_checkpoints_size
            }

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

    def validate_event(self, event: typing.Dict[str, typing.Any]) -> bool:
        """Validate that an event has the required structure and fields."""
        required_fields = {"event_id", "timestamp", "action_type", "summary", "payload"}
        
        # Check all required fields are present
        if not required_fields.issubset(event.keys()):
            return False
        
        # Validate field types
        try:
            str(event["event_id"])
            str(event["timestamp"])
            str(event["action_type"])
            str(event["summary"])
            if not isinstance(event["payload"], dict):
                return False
            return True
        except (KeyError, ValueError, TypeError):
            return False

    def validate_history_integrity(self) -> typing.Dict[str, typing.Any]:
        """Validate the integrity of the entire history."""
        report = {
            "total_events": 0,
            "valid_events": 0,
            "invalid_events": [],
            "corruption_detected": False,
            "missing_event_ids": []
        }
        
        with self._lock:
            report["total_events"] = len(self._events)
            
            for idx, event in enumerate(self._events):
                if not self.validate_event(event):
                    report["invalid_events"].append(idx)
                    report["corruption_detected"] = True
                else:
                    report["valid_events"] += 1
            
            # Check for duplicate event IDs (only for valid events)
            event_ids = []
            for idx, event in enumerate(self._events):
                if idx not in report["invalid_events"]:  # Only check valid events
                    try:
                        eid = str(event.get("event_id", ""))
                        if eid in event_ids:
                            report["corruption_detected"] = True
                            report["invalid_events"].append(idx)
                        else:
                            event_ids.append(eid)
                    except Exception:
                        report["corruption_detected"] = True
                        report["invalid_events"].append(idx)
        
        return report

    def repair_history(self) -> typing.Dict[str, typing.Any]:
        """Attempt to repair corrupted history by removing invalid events."""
        report = {
            "events_before": 0,
            "events_after": 0,
            "events_removed": 0,
            "repair_successful": False
        }
        
        with self._lock:
            report["events_before"] = len(self._events)
            
            # Filter out invalid events
            valid_events = []
            removed_indices = []
            
            for idx, event in enumerate(self._events):
                if self.validate_event(event):
                    valid_events.append(event)
                else:
                    removed_indices.append(idx)
            
            if len(removed_indices) > 0:
                self._events = valid_events
                report["events_after"] = len(self._events)
                report["events_removed"] = len(removed_indices)
                report["repair_successful"] = True
                report["removed_indices"] = removed_indices
            else:
                report["events_after"] = report["events_before"]
                report["repair_successful"] = False
        
        return report

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

    def replay(
            self,
            handlers: typing.Dict[str, typing.Callable[[typing.Dict[str, typing.Any]], None]],
            stop_on_error: bool = False,
    ) -> typing.Dict[str, typing.Any]:
        """Replay history events using provided handlers.

        Args:
            handlers: Dict mapping action_type to handler callable.
            stop_on_error: If True, stop replaying on first handler error.

        Returns:
            Dict with:
                - total: total events processed
                - replayed: events with handlers called
                - skipped: events without handlers
                - errors: list of (event_index, error_message) tuples
        """
        with self._lock:
            total = len(self._events)
            replayed = 0
            skipped = 0
            errors: typing.List[typing.Tuple[int, str]] = []

            for i, event in enumerate(self._events):
                action_type = str(event.get("action_type", ""))
                handler = handlers.get(action_type)

                if handler is None:
                    skipped += 1
                    continue

                try:
                    handler(event)
                    replayed += 1
                except Exception as e:
                    error_msg = str(e)
                    errors.append((i, error_msg))
                    if stop_on_error:
                        break

            return {
                "total": total,
                "replayed": replayed,
                "skipped": skipped,
                "errors": errors,
            }

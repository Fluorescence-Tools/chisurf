from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import time

from chisurf import typing


@dataclass(frozen=True)
class ActionSpec:
    name: str
    schema: typing.Dict[str, typing.Any] = field(default_factory=dict)
    replayable: bool = True
    debounce_ms: int = 0
    debounce_keys: typing.Optional[typing.Tuple[str, ...]] = None
    side_effect_class: str = "state"
    handler: typing.Optional[typing.Callable] = None
    _handler_params: frozenset = field(init=False, repr=False)
    _handler_has_var_kw: bool = field(init=False, repr=False)

    def __post_init__(self):
        import inspect
        if self.handler is not None:
            try:
                sig = inspect.signature(self.handler)
                names = frozenset(sig.parameters.keys())
                has_var = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            except Exception:
                names = frozenset()
                has_var = True
        else:
            names = frozenset()
            has_var = True
        object.__setattr__(self, '_handler_params', names)
        object.__setattr__(self, '_handler_has_var_kw', has_var)

    def filter_payload(self, payload: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        if self._handler_has_var_kw:
            return payload
        return {k: v for k, v in payload.items() if k in self._handler_params}

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        schema_repr: typing.Dict[str, str] = {}
        for key, expected in self.schema.items():
            if expected is None:
                schema_repr[str(key)] = "any"
            elif isinstance(expected, tuple):
                schema_repr[str(key)] = "|".join(getattr(t, "__name__", str(t)) for t in expected)
            else:
                schema_repr[str(key)] = getattr(expected, "__name__", str(expected))
        return {
            "name": str(self.name),
            "schema": schema_repr,
            "replayable": bool(self.replayable),
            "debounce_ms": int(self.debounce_ms),
            "debounce_keys": list(self.debounce_keys) if self.debounce_keys else None,
            "side_effect_class": str(self.side_effect_class),
        }

    def validate_payload(self, payload: typing.Optional[typing.Dict[str, typing.Any]]) -> typing.Dict[str, typing.Any]:
        data = dict(payload or {})
        for key, expected in self.schema.items():
            if key not in data:
                raise ValueError(f"Action '{self.name}' missing required payload key '{key}'")
            if expected is None or expected is typing.Any:
                continue
            value = data.get(key)

            if isinstance(expected, str) or getattr(expected, "__origin__", None) is not None:
                continue

            try:
                if isinstance(expected, tuple):
                    valid_types = tuple(t for t in expected if isinstance(t, type))
                    if not valid_types:
                        continue
                    ok = isinstance(value, valid_types)
                else:
                    if not isinstance(expected, type):
                        continue
                    ok = isinstance(value, expected)
            except TypeError:
                continue

            if not ok:
                exp_name = (
                    "/".join(getattr(t, "__name__", str(t)) for t in expected)
                    if isinstance(expected, tuple)
                    else getattr(expected, "__name__", str(expected))
                )
                raise TypeError(
                    f"Action '{self.name}' payload key '{key}' expects {exp_name}, got {type(value).__name__}"
                )
        return data


class ActionRegistry:
    def __init__(self):
        self._specs: typing.Dict[str, ActionSpec] = {}
        self._lock = threading.RLock()

    def register(self, spec: ActionSpec) -> None:
        name = str(spec.name)
        with self._lock:
            self._specs[name] = spec

    def has(self, name: str) -> bool:
        with self._lock:
            return str(name) in self._specs

    def get(self, name: str) -> typing.Optional[ActionSpec]:
        with self._lock:
            key = str(name)
            return self._specs.get(key)

    def resolve_name(self, name: str) -> str:
        key = str(name)
        with self._lock:
            if key in self._specs:
                return key
            dotted = key.replace("_", ".")
            if dotted in self._specs:
                return dotted
            underscored = key.replace(".", "_")
            if underscored in self._specs:
                return underscored
        return key

    def list_actions(self) -> typing.List[str]:
        with self._lock:
            return sorted(self._specs.keys())

    def catalog(self) -> typing.List[typing.Dict[str, typing.Any]]:
        with self._lock:
            names = sorted(self._specs.keys())
            return [self._specs[n].to_dict() for n in names]


class ActionDispatcher:
    def __init__(
            self,
            registry: ActionRegistry,
            history_provider: typing.Callable[[], typing.Any],
            scheduler: typing.Optional[typing.Callable[..., typing.Any]] = None,
    ):
        self.registry = registry
        self._history_provider = history_provider
        self._scheduler = scheduler
        self._lock = threading.RLock()
        self._recent_fingerprints: typing.Dict[str, float] = {}
        self._pending_timers: typing.Dict[str, threading.Timer] = {}

    def set_scheduler(self, scheduler: typing.Optional[typing.Callable[..., typing.Any]]) -> None:
        self._scheduler = scheduler

    @staticmethod
    def _safe_json(value: typing.Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return repr(value)

    def _fingerprint(
            self,
            action_type: str,
            payload: typing.Dict[str, typing.Any],
            source_uid: typing.Optional[str],
            debounce_keys: typing.Optional[typing.Tuple[str, ...]] = None,
    ) -> str:
        if debounce_keys:
            identity_payload = {k: payload.get(k) for k in debounce_keys if k in payload}
        else:
            identity_payload = payload
        return f"{action_type}|{self._safe_json(identity_payload)}|{str(source_uid or '')}"

    def _is_within_debounce(self, fingerprint: str, debounce_ms: int) -> bool:
        if debounce_ms <= 0:
            return False
        now_ms = time.monotonic() * 1000.0
        with self._lock:
            old = self._recent_fingerprints.get(fingerprint)
            self._recent_fingerprints[fingerprint] = now_ms
            if old is None:
                return False
            result = (now_ms - old) <= debounce_ms
            return result

    def _cancel_pending(self, fingerprint: str) -> None:
        with self._lock:
            old_timer = self._pending_timers.pop(fingerprint, None)
            if old_timer:
                try:
                    old_timer.cancel()
                except Exception:
                    pass

    def _schedule_trailing_edge(
            self,
            spec: ActionSpec,
            name: str,
            payload: typing.Dict[str, typing.Any],
            summary: typing.Optional[str],
            source_uid: typing.Optional[str],
            target_uid: typing.Optional[str],
            fingerprint: str,
    ) -> None:
        with self._lock:
            self._cancel_pending(fingerprint)

            def delayed_execute():
                if self._scheduler is not None:
                    self._scheduler(
                        self.execute,
                        name=name,
                        payload=payload,
                        summary=summary,
                        source_uid=source_uid,
                        target_uid=target_uid,
                        _is_trailing_edge=True
                    )
                    return
                self.execute(
                    name=name,
                    payload=payload,
                    summary=summary,
                    source_uid=source_uid,
                    target_uid=target_uid,
                    _is_trailing_edge=True
                )

            timer = threading.Timer(spec.debounce_ms / 1000.0, delayed_execute)
            self._pending_timers[fingerprint] = timer
            timer.start()

    def execute(
            self,
            name: str,
            payload: typing.Optional[typing.Dict[str, typing.Any]] = None,
            summary: typing.Optional[str] = None,
            source_uid: typing.Optional[str] = None,
            target_uid: typing.Optional[str] = None,
            _is_trailing_edge: bool = False,
            **_kw
    ) -> typing.Optional[typing.Dict[str, typing.Any]]:
        canonical_name = self.registry.resolve_name(name)
        spec = self.registry.get(canonical_name)
        if spec is None:
            import logging
            logging.getLogger(__name__).warning("dispatch(%r): unknown action", name)
            return None
        normalized_payload = spec.validate_payload(payload)

        fingerprint = self._fingerprint(canonical_name, normalized_payload, source_uid, spec.debounce_keys)

        if not _is_trailing_edge:
            if self._is_within_debounce(fingerprint, spec.debounce_ms):
                self._schedule_trailing_edge(spec, name, normalized_payload, summary, source_uid, target_uid, fingerprint)
                return None

        self._cancel_pending(fingerprint)

        result = None
        if spec.handler is not None:
            handler_kw = spec.filter_payload(normalized_payload)
            result = spec.handler(**handler_kw)

        if isinstance(result, dict):
            source_uid = result.get("source_uid", source_uid)
            target_uid = result.get("target_uid", target_uid)

        history_obj = self._history_provider()
        event = None
        if history_obj is not None and hasattr(history_obj, "record"):
            final_summary = summary or str(canonical_name)
            if _is_trailing_edge:
                final_summary += " (auto-sync)"

            enriched_payload = dict(normalized_payload)
            enriched_payload.setdefault("replayable", bool(spec.replayable))
            enriched_payload.setdefault("side_effect_class", str(spec.side_effect_class))
            enriched_payload.setdefault("debounce_ms", int(spec.debounce_ms))

            event = history_obj.record(
                action_type=str(canonical_name),
                summary=str(final_summary),
                payload=enriched_payload,
                source_uid=source_uid,
                target_uid=target_uid,
            )

        return result if result is not None else event


def build_default_dispatcher(history_provider: typing.Callable[[], typing.Any]) -> ActionDispatcher:
    return ActionDispatcher(registry=ActionRegistry(), history_provider=history_provider)


def get_action_catalog() -> typing.List[typing.Dict[str, typing.Any]]:
    import chisurf

    reg = getattr(chisurf, "action_registry", None)
    if reg is not None and hasattr(reg, "catalog"):
        return reg.catalog()

    return []


def record_action(
        action_type: str,
        summary: str,
        payload: typing.Optional[typing.Dict[str, typing.Any]] = None,
        source_uid: typing.Optional[str] = None,
        target_uid: typing.Optional[str] = None,
) -> typing.Optional[typing.Dict[str, typing.Any]]:
    import chisurf

    payload_data = payload or {}
    history_obj = getattr(chisurf, "history", None)
    if history_obj is not None and hasattr(history_obj, "record"):
        return history_obj.record(
            action_type=str(action_type),
            summary=str(summary),
            payload=payload_data,
            source_uid=source_uid,
            target_uid=target_uid,
        )

    return None


def invoke_action(
        name: str,
        payload: typing.Optional[typing.Dict[str, typing.Any]] = None,
        summary: typing.Optional[str] = None,
        source_uid: typing.Optional[str] = None,
        target_uid: typing.Optional[str] = None,
) -> typing.Optional[typing.Dict[str, typing.Any]]:
    """Alias for chisurf.core.actions.dispatch. Prefer using that directly."""
    import chisurf
    return chisurf.core.actions.dispatch(name=str(name), payload=payload)

"""Ordered app startup service lifecycle manager.

This module provides a generalised startup-service framework for the ChiSurf
application. Services are declared in JSON config files, loaded in
lexicographic order, and executed according to dependency and phase rules.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol


class _EntrypointLoader(Protocol):
    """Protocol for objects that load entrypoint callables."""

    def load(self, entrypoint: str) -> Any:
        """Return the callable registered at *entrypoint*."""


class _DefaultEntrypointLoader:
    """Load entrypoints using the ``module:attribute`` convention."""

    def load(self, entrypoint: str) -> Any:
        """Return the callable registered at *entrypoint*."""
        module_name, attribute_name = entrypoint.rsplit(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, attribute_name)


@dataclass(frozen=True)
class _EnabledIf:
    """Declarative condition for enabling a startup service."""

    source_type: str
    source_key: str
    exists: bool | None = None
    equals: Any | None = None
    truthy: bool | None = None


@dataclass(frozen=True)
class AppStartupServiceSpec:
    """Declarative description of an app startup service."""

    id: str
    entrypoint: str
    order: int = 100
    depends_on: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    ready_timeout: float = 5.0
    surface: str = "app"
    phase: str = ""
    label: str = ""
    progress: int = 0
    thread: str = "main"
    requires: tuple[str, ...] = ()
    enabled_if: _EnabledIf | None = None
    run_if_dependency_skipped: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppStartupServiceSpec:
        """Create an app startup service spec from JSON data."""
        depends_on = data.get("depends_on", [])
        methods = data.get("methods", [])
        requires = data.get("requires", [])
        if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
            raise ValueError("field 'depends_on' must be a list of strings")
        if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
            raise ValueError("field 'methods' must be a list of strings")
        if not isinstance(requires, list) or not all(isinstance(item, str) for item in requires):
            raise ValueError("field 'requires' must be a list of strings")
        surface = data.get("surface", "app")
        if not isinstance(surface, str):
            raise ValueError("field 'surface' must be a string")
        phase = data.get("phase", "")
        if not isinstance(phase, str):
            raise ValueError("field 'phase' must be a string")
        label = data.get("label", "")
        if not isinstance(label, str):
            raise ValueError("field 'label' must be a string")
        progress = data.get("progress", 0)
        if not isinstance(progress, int) or not (0 <= progress <= 100):
            raise ValueError("field 'progress' must be an integer in range 0-100")
        if "background" in data and "thread" not in data:
            thread = "background" if data["background"] else "main"
        else:
            thread = data.get("thread", "main")
        if thread not in ("main", "background"):
            raise ValueError("field 'thread' must be 'main' or 'background'")
        run_if_dependency_skipped = data.get("run_if_dependency_skipped", False)
        if not isinstance(run_if_dependency_skipped, bool):
            raise ValueError("field 'run_if_dependency_skipped' must be a boolean")

        enabled_if = None
        enabled_if_data = data.get("enabled_if")
        if enabled_if_data is not None:
            if not isinstance(enabled_if_data, dict):
                raise ValueError("field 'enabled_if' must be an object")
            source_keys = [k for k in ("setting", "attribute", "env") if k in enabled_if_data]
            if len(source_keys) != 1:
                raise ValueError(
                    "field 'enabled_if' must have exactly one source key "
                    "(setting, attribute, env)"
                )
            source_type = source_keys[0]
            source_key = enabled_if_data[source_type]
            if not isinstance(source_key, str):
                raise ValueError(f"field 'enabled_if.{source_type}' must be a string")
            comparison_keys = [k for k in ("exists", "equals", "truthy") if k in enabled_if_data]
            if len(comparison_keys) == 0:
                raise ValueError(
                    "field 'enabled_if' must have at least one comparison key "
                    "(exists, equals, truthy)"
                )
            enabled_if = _EnabledIf(
                source_type=source_type,
                source_key=source_key,
                exists=enabled_if_data.get("exists"),
                equals=enabled_if_data.get("equals"),
                truthy=enabled_if_data.get("truthy"),
            )

        return cls(
            id=str(data["id"]),
            entrypoint=str(data["entrypoint"]),
            order=int(data.get("order", 100)),
            depends_on=tuple(depends_on),
            methods=tuple(methods),
            ready_timeout=float(data.get("ready_timeout", 5.0)),
            surface=surface,
            phase=phase,
            label=label,
            progress=progress,
            thread=thread,
            requires=tuple(requires),
            enabled_if=enabled_if,
            run_if_dependency_skipped=run_if_dependency_skipped,
        )


@dataclass
class AppStartupContext:
    """Runtime context passed to app startup service entrypoints."""

    dispatcher: Any = None
    state: Any = None
    event_bus: Any = None
    job_manager: Any = None
    stop_event: threading.Event | None = None
    dependencies: dict[str, Any] = field(default_factory=dict)
    ready_event: threading.Event = field(default_factory=threading.Event)

    app: Any = None
    qt_app: Any = None
    main_window: Any = None
    plugin_registry: Any = None
    splash: Any = None
    surface: str = ""
    phase: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_ready(self) -> None:
        """Mark this service as ready to accept requests."""
        self.ready_event.set()


@dataclass
class _BackgroundServiceState:
    """Runtime state for a background-started service."""

    thread: threading.Thread
    ready_event: threading.Event
    error: BaseException | None = None
    result: Any = None


class AppStartupError(RuntimeError):
    """Raised when app startup fails."""


def _resolve_dotted_setting(dotted_path: str) -> Any:
    """Resolve a dotted path into ``chisurf.core.settings.cs_settings``."""
    import chisurf.core.settings
    value = chisurf.core.settings.cs_settings
    for part in dotted_path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            try:
                value = getattr(value, part)
            except AttributeError:
                return None
    return value


def _resolve_dotted_attribute(obj: Any, dotted_path: str) -> Any:
    """Resolve a dotted path into an object hierarchy."""
    value = obj
    for part in dotted_path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            try:
                value = getattr(value, part)
            except AttributeError:
                return None
    return value


class AppStartupServiceManager:
    """Start app services in a JSON-defined dependency order.

    Services marked ``background=True`` run their entrypoint in a daemon thread.
    Startup still waits until each service marks itself ready.
    """

    def __init__(
        self,
        dispatcher: Any = None,
        state: Any = None,
        event_bus: Any = None,
        job_manager: Any = None,
        *,
        specs: Iterable[AppStartupServiceSpec] | None = None,
        config_path: str | Path | None = None,
        loader: _EntrypointLoader | None = None,
    ) -> None:
        """Initialise the app startup service manager."""
        self.dispatcher = dispatcher
        self.state = state
        self.event_bus = event_bus
        self.job_manager = job_manager
        self._loader = loader or _DefaultEntrypointLoader()
        self._stop_event = threading.Event()
        self._states: dict[str, _BackgroundServiceState] = {}
        self._results: dict[str, Any] = {}
        self._started_ids: set[str] = set()
        self._specs = tuple(specs) if specs is not None else load_app_startup_services(config_path)
        self._skipped: dict[str, tuple[AppStartupServiceSpec, str]] = {}

    @classmethod
    def from_specs(
        cls,
        specs: Iterable[AppStartupServiceSpec],
        dispatcher: Any = None,
        state: Any = None,
        event_bus: Any = None,
        job_manager: Any = None,
        **kwargs: Any,
    ) -> AppStartupServiceManager:
        """Create a manager from explicit service specs."""
        return cls(
            dispatcher=dispatcher,
            state=state,
            event_bus=event_bus,
            job_manager=job_manager,
            specs=specs,
            **kwargs,
        )

    @property
    def specs(self) -> tuple[AppStartupServiceSpec, ...]:
        """Return the configured startup service specs."""
        return self._specs

    @property
    def entrypoints(self) -> set[str]:
        """Return entrypoints owned by this startup configuration.

        All configured specs are returned regardless of skip status
        (configured-owner semantics) to prevent duplicate plugin registration.
        """
        return {spec.entrypoint for spec in self._specs}

    def get_skipped_services(self) -> dict[str, str]:
        """Return a dict of skipped service IDs to skip reasons."""
        return {sid: reason for sid, (_, reason) in self._skipped.items()}

    def ordered_specs(
        self,
        surface: str | None = None,
        phase: str | None = None,
    ) -> list[AppStartupServiceSpec]:
        """Return services sorted by dependencies and startup order.

        Parameters
        ----------
        surface : str, optional
            If set, only return specs matching this surface.
        phase : str, optional
            If set, only return specs matching this phase.
        """
        filtered = self._specs
        if surface is not None:
            filtered = [s for s in filtered if s.surface == surface]
        if phase is not None:
            filtered = [s for s in filtered if s.phase == phase]

        by_id = {spec.id: spec for spec in filtered}
        if len(by_id) != len(filtered):
            duplicate_ids = sorted(
                {
                    spec.id
                    for spec in filtered
                    if sum(1 for item in filtered if item.id == spec.id) > 1
                }
            )
            raise AppStartupError(f"duplicate service id(s): {', '.join(duplicate_ids)}")

        ordered: list[AppStartupServiceSpec] = []
        remaining = set(by_id)
        completed: set[str] = set()
        while remaining:
            candidates = [
                service_id
                for service_id in remaining
                if set(by_id[service_id].depends_on).issubset(completed)
            ]
            if not candidates:
                missing = sorted(
                    dep
                    for service_id in remaining
                    for dep in by_id[service_id].depends_on
                    if dep not in completed
                )
                raise AppStartupError(
                    f"service dependency cycle or missing dependency: {', '.join(missing)}"
                )
            service_id = sorted(candidates, key=lambda item: (by_id[item].order, item))[0]
            spec = by_id[service_id]
            ordered.append(spec)
            completed.add(service_id)
            remaining.remove(service_id)
        return ordered

    def resolve_enabled(
        self,
        surface: str | None = None,
        phase: str | None = None,
    ) -> list[AppStartupServiceSpec]:
        """Return enabled specs after evaluating conditions.

        Evaluates ``enabled_if`` conditions and dependency skipping without
        executing services. Results are cached so subsequent calls use the
        same skip decisions.
        """
        ordered = self.ordered_specs(surface=surface, phase=phase)
        self._evaluate_conditions(ordered)
        return [s for s in ordered if s.id not in self._skipped]

    def start(
        self,
        surface: str | None = None,
        phase: str | None = None,
        on_stage_start: Callable[[dict[str, Any]], None] | None = None,
        on_stage_finish: Callable[[dict[str, Any]], None] | None = None,
        on_stage_skip: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Start services in dependency order, optionally filtered.

        Parameters
        ----------
        surface : str, optional
            If set, only start services for this surface.
        phase : str, optional
            If set, only start services for this phase.
        on_stage_start : callable, optional
            Called before each service with a dict containing id, label,
            surface, phase, and progress.
        on_stage_finish : callable, optional
            Called after each service completes with the same payload dict
            plus ``skipped`` and ``skip_reason`` keys.
        on_stage_skip : callable, optional
            Called for skipped services with a payload dict containing id,
            label, surface, phase, progress, and skip_reason.
        """
        ordered = self.ordered_specs(surface=surface, phase=phase)
        self._evaluate_conditions(ordered)

        # Shared context accumulates state as services complete
        context = AppStartupContext(
            dispatcher=self.dispatcher,
            state=self.state,
            event_bus=self.event_bus,
            job_manager=self.job_manager,
            stop_event=self._stop_event,
            surface=surface or "",
            phase=phase or "",
        )

        for spec in ordered:
            if spec.id in self._started_ids:
                continue

            base_payload = {
                "id": spec.id,
                "label": spec.label,
                "surface": spec.surface,
                "phase": spec.phase,
                "progress": spec.progress,
            }

            if spec.id in self._skipped:
                _, skip_reason = self._skipped[spec.id]
                payload = {**base_payload, "skipped": True, "skip_reason": skip_reason}
                if on_stage_skip is not None:
                    on_stage_skip(payload)
                continue

            # Deferred attribute condition evaluation
            if spec.enabled_if and spec.enabled_if.source_type == "attribute":
                enabled, reason = self._evaluate_context_condition(spec, context)
                if not enabled:
                    self._skipped[spec.id] = (spec, reason or "disabled by enabled_if")
                    # Propagate to dependents still in this ordered list
                    for later in ordered:
                        if later.id in self._skipped or later.id in self._started_ids:
                            continue
                        if spec.id in later.depends_on and not later.run_if_dependency_skipped:
                            self._skipped[later.id] = (later, f"dependency '{spec.id}' is skipped")
                    _, skip_reason = self._skipped[spec.id]
                    if on_stage_skip is not None:
                        on_stage_skip({**base_payload, "skipped": True, "skip_reason": skip_reason})
                    continue

            if on_stage_start is not None:
                on_stage_start(base_payload)
            self._start_service(spec, context)
            self._started_ids.add(spec.id)
            self._update_context_from_result(context, spec.id)
            if on_stage_finish is not None:
                on_stage_finish({**base_payload, "skipped": False, "skip_reason": ""})

    def get_service_result(self, service_id: str) -> Any:
        """Return the result of a completed service, or None."""
        return self._results.get(service_id)

    def stop(self) -> None:
        """Request background services to stop and wait briefly for threads."""
        self._stop_event.set()
        for state in list(self._states.values()):
            if state.thread.is_alive():
                state.thread.join(timeout=1.0)

    def _evaluate_conditions(self, ordered: list[AppStartupServiceSpec]) -> None:
        """Evaluate ``enabled_if`` conditions and propagate dependency skips.

        Only evaluates specs not already in ``_skipped`` or ``_started_ids``.
        """
        # First pass: evaluate enabled_if for each spec
        for spec in ordered:
            if spec.id in self._skipped or spec.id in self._started_ids:
                continue
            enabled, reason = self._evaluate_enabled_if(spec)
            if not enabled:
                self._skipped[spec.id] = (spec, reason or "disabled by enabled_if")

        # Second pass: propagate skips to dependents (iterate until stable)
        changed = True
        while changed:
            changed = False
            for spec in ordered:
                if spec.id in self._skipped or spec.id in self._started_ids:
                    continue
                for dep in spec.depends_on:
                    if dep in self._skipped:
                        if not spec.run_if_dependency_skipped:
                            self._skipped[spec.id] = (
                                spec,
                                f"dependency '{dep}' is skipped",
                            )
                            changed = True
                            break

    def _evaluate_enabled_if(self, spec: AppStartupServiceSpec) -> tuple[bool, str]:
        """Evaluate a single spec's ``enabled_if`` condition.

        Returns ``(True, "")`` if enabled, or ``(False, reason)`` if disabled.
        """
        cond = spec.enabled_if
        if cond is None:
            return True, ""

        if cond.source_type == "setting":
            value = _resolve_dotted_setting(cond.source_key)
        elif cond.source_type == "attribute":
            # Deferred to execution time — context may not be ready
            return True, ""
        elif cond.source_type == "env":
            value = os.environ.get(cond.source_key)
        else:
            return False, f"unknown condition source: {cond.source_type}"

        if cond.exists is not None:
            exists = value is not None
            if cond.exists != exists:
                return False, (
                    f"setting '{cond.source_key}' exists={exists}, "
                    f"expected exists={cond.exists}"
                )

        if cond.equals is not None:
            if cond.source_type == "env":
                # Coerce env string to match comparison type
                if isinstance(cond.equals, bool):
                    matched = value is not None and value.lower() in (
                        ("true", "1", "yes") if cond.equals else ("false", "0", "no")
                    )
                elif isinstance(cond.equals, (int, float)):
                    try:
                        matched = type(cond.equals)(value) == cond.equals
                    except (ValueError, TypeError):
                        matched = False
                else:
                    matched = value == cond.equals
            else:
                matched = value == cond.equals
            if not matched:
                return False, (
                    f"condition '{cond.source_key}' equals {value!r}, "
                    f"expected {cond.equals!r}"
                )

        if cond.truthy is not None:
            truthy = bool(value)
            if cond.truthy != truthy:
                return False, (
                    f"setting '{cond.source_key}' truthy={truthy}, "
                    f"expected truthy={cond.truthy}"
                )

        return True, ""

    def _start_service(
        self,
        spec: AppStartupServiceSpec,
        context: AppStartupContext | None = None,
    ) -> None:
        """Start one app startup service."""
        if context is None:
            context = AppStartupContext(
                dispatcher=self.dispatcher,
                state=self.state,
                event_bus=self.event_bus,
                job_manager=self.job_manager,
                stop_event=self._stop_event,
                dependencies={dep: self._results.get(dep) for dep in spec.depends_on},
                surface=spec.surface,
                phase=spec.phase,
            )
        else:
            context.dependencies = {dep: self._results.get(dep) for dep in spec.depends_on}
        if spec.thread == "background":
            ready_event = threading.Event()
            context.ready_event = ready_event
            state = _BackgroundServiceState(
                thread=threading.Thread(
                    target=self._run_service,
                    args=(spec, context),
                    daemon=True,
                    name=f"chisurf-startup-{spec.id}",
                ),
                ready_event=ready_event,
            )
            self._states[spec.id] = state
            state.thread.start()
            if not ready_event.wait(timeout=spec.ready_timeout):
                raise AppStartupError(
                    f"service '{spec.id}' did not become ready within {spec.ready_timeout}s"
                )
            if state.error is not None:
                raise AppStartupError(f"service '{spec.id}' failed") from state.error
            self._results[spec.id] = state.result
            return

        result = self._run_service(spec, context)
        context.mark_ready()
        self._results[spec.id] = result

    def _evaluate_context_condition(
        self,
        spec: AppStartupServiceSpec,
        context: AppStartupContext,
    ) -> tuple[bool, str]:
        """Evaluate an ``enabled_if.attribute`` condition at execution time."""
        cond = spec.enabled_if
        if cond is None or cond.source_type != "attribute":
            return True, ""
        value = _resolve_dotted_attribute(context, cond.source_key)
        if cond.exists is not None:
            exists = value is not None
            if cond.exists != exists:
                return False, (
                    f"attribute '{cond.source_key}' exists={exists}, "
                    f"expected exists={cond.exists}"
                )
        if cond.equals is not None:
            if value != cond.equals:
                return False, (
                    f"attribute '{cond.source_key}' equals {value!r}, "
                    f"expected {cond.equals!r}"
                )
        if cond.truthy is not None:
            truthy = bool(value)
            if cond.truthy != truthy:
                return False, (
                    f"attribute '{cond.source_key}' truthy={truthy}, "
                    f"expected truthy={cond.truthy}"
                )
        return True, ""

    def _update_context_from_result(
        self,
        context: AppStartupContext,
        service_id: str,
    ) -> None:
        """Populate shared context fields from a completed service result."""
        result = self._results.get(service_id)
        if isinstance(result, dict):
            for key, value in result.items():
                if hasattr(context, key):
                    setattr(context, key, value)

    def _run_service(self, spec: AppStartupServiceSpec, context: AppStartupContext) -> Any:
        """Load and call one service entrypoint."""
        try:
            register_fn = getattr(self._loader, "load", self._loader)(spec.entrypoint)
            result = register_fn(context)
            context.mark_ready()
            return result
        except BaseException as exc:
            for state in self._states.values():
                if state.thread is threading.current_thread():
                    state.error = exc
                    state.ready_event.set()
            raise


def load_app_startup_services(
    config_path: str | Path | None = None,
) -> tuple[AppStartupServiceSpec, ...]:
    """Load ordered app startup services from Linux-style prefixed JSON configs."""
    paths = _startup_config_paths(config_path)
    specs: list[AppStartupServiceSpec] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"app startup config must be a JSON object: {path}")
        services = data.get("services", [])
        if not isinstance(services, list):
            raise ValueError(f"field 'services' must be a list in {path}")
        default_order = _order_from_filename(path.name)
        for item in services:
            if not isinstance(item, dict):
                raise ValueError(f"each service entry must be an object in {path}")
            if item.get("enabled", True) is False:
                continue
            merged = dict(item)
            merged.setdefault("order", default_order)
            specs.append(AppStartupServiceSpec.from_dict(merged))
    return tuple(specs)


def _startup_config_paths(config_path: str | Path | None) -> list[Path]:
    """Return startup config files sorted lexicographically by filename."""
    if config_path is None:
        directory = resources.files("chisurf.startup").joinpath("services.d")
        return sorted(
            (Path(str(path)) for path in directory.iterdir() if path.name.endswith(".json")),
            key=lambda item: item.name,
        )

    path = Path(config_path)
    if path.is_dir():
        return sorted(path.glob("*.json"), key=lambda item: item.name)
    if path.is_file():
        return [path]
    raise FileNotFoundError(path)


def _order_from_filename(filename: str) -> int:
    """Return the numeric prefix order from a config filename."""
    match = re.match(r"^(\d+)_", filename)
    if match is None:
        return 100
    return int(match.group(1))

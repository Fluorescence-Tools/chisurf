"""Ordered service startup configuration for the JSON-RPC server."""

from __future__ import annotations

import importlib
import json
import re
import threading
from collections.abc import Iterable
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
class StartupServiceSpec:
    """Declarative description of a server startup service."""

    id: str
    entrypoint: str
    order: int = 100
    background: bool = False
    depends_on: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    ready_timeout: float = 5.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StartupServiceSpec:
        """Create a startup service spec from JSON data."""
        depends_on = data.get("depends_on", [])
        methods = data.get("methods", [])
        if not isinstance(depends_on, list) or not all(isinstance(item, str) for item in depends_on):
            raise ValueError("field 'depends_on' must be a list of strings")
        if not isinstance(methods, list) or not all(isinstance(item, str) for item in methods):
            raise ValueError("field 'methods' must be a list of strings")
        return cls(
            id=str(data["id"]),
            entrypoint=str(data["entrypoint"]),
            order=int(data.get("order", 100)),
            background=bool(data.get("background", False)),
            depends_on=tuple(depends_on),
            methods=tuple(methods),
            ready_timeout=float(data.get("ready_timeout", 5.0)),
        )


@dataclass
class StartupServiceContext:
    """Runtime context passed to ordered startup service entrypoints."""

    dispatcher: Any
    state: Any
    event_bus: Any
    job_manager: Any
    stop_event: threading.Event
    dependencies: dict[str, Any] = field(default_factory=dict)
    ready_event: threading.Event = field(default_factory=threading.Event)

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


class ServiceStartupError(RuntimeError):
    """Raised when ordered service startup fails."""


class ServiceStartupManager:
    """Start server services in a JSON-defined dependency order.

    Services marked ``background=True`` run their entrypoint in a daemon thread.
    Startup still waits until each service marks itself ready, so the ZMQ server
    never accepts requests before required services are registered.
    """

    def __init__(
        self,
        dispatcher: Any,
        state: Any,
        event_bus: Any,
        job_manager: Any,
        *,
        specs: Iterable[StartupServiceSpec] | None = None,
        config_path: str | Path | None = None,
        loader: _EntrypointLoader | None = None,
    ) -> None:
        """Initialise the ordered service startup manager."""
        self.dispatcher = dispatcher
        self.state = state
        self.event_bus = event_bus
        self.job_manager = job_manager
        self._loader = loader or _DefaultEntrypointLoader()
        self._stop_event = threading.Event()
        self._states: dict[str, _BackgroundServiceState] = {}
        self._results: dict[str, Any] = {}
        self._started_ids: set[str] = set()
        self._specs = tuple(specs) if specs is not None else load_startup_services(config_path)

    @classmethod
    def from_specs(
        cls,
        specs: Iterable[StartupServiceSpec],
        dispatcher: Any,
        state: Any,
        event_bus: Any,
        job_manager: Any,
        **kwargs: Any,
    ) -> ServiceStartupManager:
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
    def specs(self) -> tuple[StartupServiceSpec, ...]:
        """Return the configured startup service specs."""
        return self._specs

    @property
    def entrypoints(self) -> set[str]:
        """Return entrypoints owned by this startup configuration."""
        return {spec.entrypoint for spec in self._specs}

    def ordered_specs(self) -> list[StartupServiceSpec]:
        """Return services sorted by dependencies and startup order."""
        by_id = {spec.id: spec for spec in self._specs}
        if len(by_id) != len(self._specs):
            duplicate_ids = sorted(
                {
                    spec.id
                    for spec in self._specs
                    if sum(1 for item in self._specs if item.id == spec.id) > 1
                }
            )
            raise ServiceStartupError(f"duplicate service id(s): {', '.join(duplicate_ids)}")

        ordered: list[StartupServiceSpec] = []
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
                raise ServiceStartupError(
                    f"service dependency cycle or missing dependency: {', '.join(missing)}"
                )
            service_id = sorted(candidates, key=lambda item: (by_id[item].order, item))[0]
            spec = by_id[service_id]
            ordered.append(spec)
            completed.add(service_id)
            remaining.remove(service_id)
        return ordered

    def start(self) -> None:
        """Start all configured services in dependency order."""
        if self._started_ids:
            return
        for spec in self.ordered_specs():
            self._start_service(spec)
            self._started_ids.add(spec.id)

    def stop(self) -> None:
        """Request background services to stop and wait briefly for threads."""
        self._stop_event.set()
        for state in list(self._states.values()):
            if state.thread.is_alive():
                state.thread.join(timeout=1.0)

    def _start_service(self, spec: StartupServiceSpec) -> None:
        """Start one startup service."""
        dependencies = {dep: self._results.get(dep) for dep in spec.depends_on}
        context = StartupServiceContext(
            dispatcher=self.dispatcher,
            state=self.state,
            event_bus=self.event_bus,
            job_manager=self.job_manager,
            stop_event=self._stop_event,
            dependencies=dependencies,
        )
        if spec.background:
            ready_event = threading.Event()
            context.ready_event = ready_event
            state = _BackgroundServiceState(
                thread=threading.Thread(
                    target=self._run_service,
                    args=(spec, context),
                    daemon=True,
                    name=f"chisurf-service-{spec.id}",
                ),
                ready_event=ready_event,
            )
            self._states[spec.id] = state
            state.thread.start()
            if not ready_event.wait(timeout=spec.ready_timeout):
                raise ServiceStartupError(
                    f"service '{spec.id}' did not become ready within {spec.ready_timeout}s"
                )
            if state.error is not None:
                raise ServiceStartupError(f"service '{spec.id}' failed") from state.error
            self._results[spec.id] = state.result
            return

        result = self._run_service(spec, context)
        context.mark_ready()
        self._results[spec.id] = result

    def _run_service(self, spec: StartupServiceSpec, context: StartupServiceContext) -> Any:
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


def load_startup_services(config_path: str | Path | None = None) -> tuple[StartupServiceSpec, ...]:
    """Load ordered startup services from Linux-style prefixed JSON configs."""
    paths = _startup_config_paths(config_path)
    specs: list[StartupServiceSpec] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"service startup config must be a JSON object: {path}")
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
            specs.append(StartupServiceSpec.from_dict(merged))
    return tuple(specs)


def _startup_config_paths(config_path: str | Path | None) -> list[Path]:
    """Return startup config files sorted lexicographically by filename."""
    if config_path is None:
        directory = resources.files("chisurf.server").joinpath("service_startup.d")
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

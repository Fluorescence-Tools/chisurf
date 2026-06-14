"""Tests for ordered JSON startup services."""

from __future__ import annotations

import pytest

from chisurf.server.service_startup import (
    ServiceStartupError,
    ServiceStartupManager,
    StartupServiceSpec,
    load_startup_services,
)


def _loader(calls: list[tuple[str, list[str]]]):
    """Return a fake entrypoint loader that records startup calls."""

    def load(entrypoint: str):
        """Return a fake service entrypoint."""

        def register(context):
            """Register the fake service."""
            calls.append((entrypoint, sorted(context.dependencies)))
            context.mark_ready()

        return register

    return load


class _FakeDispatcher:
    """Minimal dispatcher used by password service registration tests."""

    def __init__(self) -> None:
        """Create an empty handler registry."""
        self.handlers = {}

    def register(self, name: str, handler) -> None:
        """Register a handler by name."""
        self.handlers[name] = handler


def test_load_startup_services_uses_prefixed_config_order():
    """Prefixed config filenames define default startup order."""
    specs = load_startup_services()
    assert [spec.id for spec in specs] == ["mfdb"]
    assert specs[0].order == 10


def test_load_startup_services_explicit_order_overrides_prefix(tmp_path):
    """Explicit order overrides the numeric filename prefix."""
    (tmp_path / "10_b.json").write_text(
        '{"services": [{"id": "b", "entrypoint": "pkg:register"}]}',
        encoding="utf-8",
    )
    (tmp_path / "20_a.json").write_text(
        '{"services": [{"id": "a", "entrypoint": "pkg:register", "order": 5}]}',
        encoding="utf-8",
    )

    specs = load_startup_services(tmp_path)
    manager = ServiceStartupManager.from_specs(
        specs,
        dispatcher=object(),
        state=object(),
        event_bus=object(),
        job_manager=object(),
    )

    assert [spec.id for spec in manager.ordered_specs()] == ["a", "b"]
    assert specs[0].order == 10
    assert specs[1].order == 5


def test_service_startup_manager_orders_dependencies_before_order():
    """Dependencies take precedence over numeric order."""
    specs = [
        StartupServiceSpec(id="b", entrypoint="pkg:b", order=20, depends_on=("a",)),
        StartupServiceSpec(id="a", entrypoint="pkg:a", order=10),
    ]
    manager = ServiceStartupManager.from_specs(
        specs,
        dispatcher=object(),
        state=object(),
        event_bus=object(),
        job_manager=object(),
    )

    assert [spec.id for spec in manager.ordered_specs()] == ["a", "b"]


def test_mfdb_auth_service_registers_auth_methods():
    """The MFDB auth service exposes auth methods."""
    from chisurf.plugins.core.mfdb_admin.backend import auth_services

    dispatcher = _FakeDispatcher()

    auth_services.register_services(dispatcher)

    assert {
        "mfdb.auth.login",
        "mfdb.auth.logout",
        "mfdb.auth.me",
        "mfdb.auth.change_password",
    }.issubset(dispatcher.handlers)


def test_service_startup_manager_starts_background_services_in_order():
    """Background services start sequentially and receive dependency results."""
    calls: list[tuple[str, list[str]]] = []
    specs = [
        StartupServiceSpec(id="mfdb", entrypoint="pkg:mfdb", order=10, background=True),
        StartupServiceSpec(
            id="password",
            entrypoint="pkg:password",
            order=20,
            background=True,
            depends_on=("mfdb",),
        ),
    ]
    manager = ServiceStartupManager.from_specs(
        specs,
        dispatcher=object(),
        state=object(),
        event_bus=object(),
        job_manager=object(),
        loader=_loader(calls),
    )

    manager.start()

    assert calls == [("pkg:mfdb", []), ("pkg:password", ["mfdb"])]
    assert manager.entrypoints == {"pkg:mfdb", "pkg:password"}
    manager.stop()


def test_service_startup_manager_rejects_missing_dependency():
    """Missing dependencies fail startup early."""
    specs = [
        StartupServiceSpec(id="password", entrypoint="pkg:password", depends_on=("mfdb",)),
    ]
    manager = ServiceStartupManager.from_specs(
        specs,
        dispatcher=object(),
        state=object(),
        event_bus=object(),
        job_manager=object(),
    )

    with pytest.raises(ServiceStartupError, match="missing dependency"):
        manager.start()


def test_service_startup_manager_rejects_dependency_cycle():
    """Dependency cycles fail startup early."""
    specs = [
        StartupServiceSpec(id="a", entrypoint="pkg:a", depends_on=("b",)),
        StartupServiceSpec(id="b", entrypoint="pkg:b", depends_on=("a",)),
    ]
    manager = ServiceStartupManager.from_specs(
        specs,
        dispatcher=object(),
        state=object(),
        event_bus=object(),
        job_manager=object(),
    )

    with pytest.raises(ServiceStartupError, match="cycle"):
        manager.start()

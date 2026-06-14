"""Tests for GUI startup services."""

from __future__ import annotations

import pytest

from chisurf.startup.services import (
    AppStartupServiceManager,
    AppStartupServiceSpec,
    load_app_startup_services,
)


class _TestLoader:
    """Entrypoint loader that records which services run."""

    def __init__(self):
        self.calls = []

    def load(self, entrypoint: str):
        def register(context):
            context.mark_ready()
            self.calls.append((entrypoint, context.surface, context.phase))

        return register


def test_gui_splash_services_load_from_default_config():
    """GUI splash services are loaded from the default config."""
    specs = load_app_startup_services()
    splash_specs = [s for s in specs if s.surface == "gui" and s.phase == "splash"]
    assert len(splash_specs) > 0
    for spec in splash_specs:
        assert spec.surface == "gui"
        assert spec.phase == "splash"
        assert isinstance(spec.label, str) and spec.label
        assert isinstance(spec.progress, int) and 0 <= spec.progress <= 100


def test_gui_post_show_services_load_from_default_config():
    """GUI post-show services are loaded from the default config."""
    specs = load_app_startup_services()
    post_specs = [s for s in specs if s.surface == "gui" and s.phase == "post_gui_show"]
    assert len(post_specs) > 0
    for spec in post_specs:
        assert spec.surface == "gui"
        assert spec.phase == "post_gui_show"

    # Jupyter service should have enabled_if gate
    jupyter = next((s for s in post_specs if s.id == "gui.start_jupyter"), None)
    assert jupyter is not None
    assert jupyter.enabled_if is not None
    assert jupyter.enabled_if.source_type == "setting"
    assert "gui.start_jupyter_on_startup" in jupyter.enabled_if.source_key


def test_splash_services_execute_in_order():
    """Splash phase services execute in dependency/order sequence."""
    loader = _TestLoader()
    specs = load_app_startup_services()
    splash_specs = [s for s in specs if s.surface == "gui" and s.phase == "splash"]
    manager = AppStartupServiceManager.from_specs(splash_specs, loader=loader)
    manager.start(surface="gui", phase="splash")

    assert len(loader.calls) == len(splash_specs)
    entrypoints_run = [call[0] for call in loader.calls]
    for spec in splash_specs:
        assert spec.entrypoint in entrypoints_run


def test_splash_services_have_progress_labels():
    """Stage progress payloads contain labels and progress values."""
    specs = load_app_startup_services()
    splash_specs = [s for s in specs if s.surface == "gui" and s.phase == "splash"]
    for spec in splash_specs:
        assert spec.label, f"Service '{spec.id}' has no label"
        assert 0 <= spec.progress <= 100, f"Service '{spec.id}' has invalid progress {spec.progress}"


def test_startup_interface_service_produces_main_window():
    """The startup_interface service sets the main window result."""
    def fake_startup_interface(context):
        context.mark_ready()
        return {"type": "main_window"}

    class _SingleLoader:
        def load(self, entrypoint):
            return fake_startup_interface

    spec = AppStartupServiceSpec(
        id="startup_interface",
        entrypoint="chisurf.startup.gui_services:startup_interface",
        surface="gui",
        phase="splash",
    )
    manager = AppStartupServiceManager.from_specs([spec], loader=_SingleLoader())
    manager.start()

    result = manager.get_service_result("startup_interface")
    assert result is not None
    assert result["type"] == "main_window"


def test_post_show_services_filtered_separately_from_splash():
    """Post-show services can be filtered separately from splash services."""
    specs = [
        AppStartupServiceSpec(id="splash_a", entrypoint="pkg:a", surface="gui", phase="splash"),
        AppStartupServiceSpec(id="post_b", entrypoint="pkg:b", surface="gui", phase="post_gui_show"),
    ]
    manager = AppStartupServiceManager.from_specs(specs)

    splash_ordered = manager.ordered_specs(surface="gui", phase="splash")
    post_ordered = manager.ordered_specs(surface="gui", phase="post_gui_show")

    assert len(splash_ordered) == 1
    assert splash_ordered[0].id == "splash_a"
    assert len(post_ordered) == 1
    assert post_ordered[0].id == "post_b"


def test_disabled_jupyter_service_does_not_block():
    """Jupyter service is skipped via enabled_if when setting is falsy."""
    import chisurf.core.settings
    if "gui" not in chisurf.core.settings.cs_settings:
        chisurf.core.settings.cs_settings["gui"] = {}
    chisurf.core.settings.cs_settings["gui"]["start_jupyter_on_startup"] = False

    specs = load_app_startup_services()
    post_specs = [s for s in specs if s.surface == "gui" and s.phase == "post_gui_show"]

    class _FakeLoader:
        def load(self, entrypoint):
            def svc(ctx):
                ctx.mark_ready()
            return svc

    manager = AppStartupServiceManager.from_specs(post_specs, loader=_FakeLoader())
    enabled = manager.resolve_enabled()

    enabled_ids = {s.id for s in enabled}
    assert "gui.start_jupyter" not in enabled_ids
    assert "gui.populate_notebooks" not in enabled_ids

    del chisurf.core.settings.cs_settings["gui"]["start_jupyter_on_startup"]


def test_enabled_jupyter_service_runs():
    """Jupyter service runs via enabled_if when setting is truthy."""
    import chisurf.core.settings
    if "gui" not in chisurf.core.settings.cs_settings:
        chisurf.core.settings.cs_settings["gui"] = {}
    chisurf.core.settings.cs_settings["gui"]["start_jupyter_on_startup"] = True

    specs = load_app_startup_services()
    post_specs = [s for s in specs if s.surface == "gui" and s.phase == "post_gui_show"]

    class _FakeLoader:
        def load(self, entrypoint):
            def svc(ctx):
                ctx.mark_ready()
            return svc

    manager = AppStartupServiceManager.from_specs(post_specs, loader=_FakeLoader())
    enabled = manager.resolve_enabled()

    enabled_ids = {s.id for s in enabled}
    assert "gui.start_jupyter" in enabled_ids
    assert "gui.populate_notebooks" in enabled_ids

    del chisurf.core.settings.cs_settings["gui"]["start_jupyter_on_startup"]

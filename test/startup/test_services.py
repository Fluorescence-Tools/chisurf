"""Tests for app-level ordered startup services."""

from __future__ import annotations

import pytest

from chisurf.startup.services import (
    AppStartupError,
    AppStartupServiceManager,
    AppStartupServiceSpec,
    _EnabledIf,
    load_app_startup_services,
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


def test_load_app_startup_services_uses_prefixed_config_order():
    """Prefixed config filenames define default startup order."""
    specs = load_app_startup_services()
    assert [spec.id for spec in specs] == ["mfdb", "gui_imports", "setup_ipython", "startup_interface", "setup_logging", "init_setups", "restore_setup_defaults", "define_actions", "load_tools", "init_executors", "arrange_widgets", "setup_style", "deferred_gui_imports", "populate_plugins", "check_updates", "gui.start_jupyter", "gui.populate_notebooks", "warmup_imports"]
    mfdb_spec = next(s for s in specs if s.id == "mfdb")
    assert mfdb_spec.surface == "server"
    assert mfdb_spec.phase == "pre_server_listen"
    assert mfdb_spec.order == 10


def test_load_app_startup_services_explicit_order_overrides_prefix(tmp_path):
    """Explicit order overrides the numeric filename prefix."""
    (tmp_path / "10_b.json").write_text(
        '{"services": [{"id": "b", "entrypoint": "pkg:register"}]}',
        encoding="utf-8",
    )
    (tmp_path / "20_a.json").write_text(
        '{"services": [{"id": "a", "entrypoint": "pkg:register", "order": 5}]}',
        encoding="utf-8",
    )

    specs = load_app_startup_services(tmp_path)
    manager = AppStartupServiceManager.from_specs(
        specs,
    )

    assert [spec.id for spec in manager.ordered_specs()] == ["a", "b"]
    assert specs[0].order == 10
    assert specs[1].order == 5


def test_app_startup_manager_orders_dependencies_before_order():
    """Dependencies take precedence over numeric order."""
    specs = [
        AppStartupServiceSpec(id="b", entrypoint="pkg:b", order=20, depends_on=("a",)),
        AppStartupServiceSpec(id="a", entrypoint="pkg:a", order=10),
    ]
    manager = AppStartupServiceManager.from_specs(specs)

    assert [spec.id for spec in manager.ordered_specs()] == ["a", "b"]


def test_mfdb_auth_service_registers_auth_methods():
    """The MFDB auth service exposes auth methods."""
    from mfdb.admin.backend import auth_services

    dispatcher = _FakeDispatcher()

    auth_services.register_services(dispatcher)

    assert {
        "mfdb.auth.login",
        "mfdb.auth.logout",
        "mfdb.auth.me",
        "mfdb.auth.change_password",
    }.issubset(dispatcher.handlers)


def test_app_startup_manager_starts_background_services_in_order():
    """Background services start sequentially and receive dependency results."""
    calls: list[tuple[str, list[str]]] = []
    specs = [
        AppStartupServiceSpec(id="mfdb", entrypoint="pkg:mfdb", order=10, thread="background"),
        AppStartupServiceSpec(
            id="password",
            entrypoint="pkg:password",
            order=20,
            thread="background",
            depends_on=("mfdb",),
        ),
    ]
    manager = AppStartupServiceManager.from_specs(
        specs,
        loader=_loader(calls),
    )

    manager.start()

    assert calls == [("pkg:mfdb", []), ("pkg:password", ["mfdb"])]
    assert manager.entrypoints == {"pkg:mfdb", "pkg:password"}
    manager.stop()


def test_app_startup_manager_rejects_missing_dependency():
    """Missing dependencies fail startup early."""
    specs = [
        AppStartupServiceSpec(id="password", entrypoint="pkg:password", depends_on=("mfdb",)),
    ]
    manager = AppStartupServiceManager.from_specs(specs)

    with pytest.raises(AppStartupError, match="missing dependency"):
        manager.start()


def test_app_startup_manager_rejects_dependency_cycle():
    """Dependency cycles fail startup early."""
    specs = [
        AppStartupServiceSpec(id="a", entrypoint="pkg:a", depends_on=("b",)),
        AppStartupServiceSpec(id="b", entrypoint="pkg:b", depends_on=("a",)),
    ]
    manager = AppStartupServiceManager.from_specs(specs)

    with pytest.raises(AppStartupError, match="cycle"):
        manager.start()


def test_app_startup_manager_filters_by_surface():
    """Filtering by surface returns only matching specs."""
    specs = [
        AppStartupServiceSpec(id="server_svc", entrypoint="pkg:server", surface="server"),
        AppStartupServiceSpec(id="gui_svc", entrypoint="pkg:gui", surface="gui"),
        AppStartupServiceSpec(id="app_svc", entrypoint="pkg:app", surface="app"),
    ]
    manager = AppStartupServiceManager.from_specs(specs)

    ordered = manager.ordered_specs(surface="gui")
    assert [s.id for s in ordered] == ["gui_svc"]


def test_app_startup_manager_filters_by_phase():
    """Filtering by phase returns only matching specs."""
    specs = [
        AppStartupServiceSpec(id="splash", entrypoint="pkg:splash", surface="gui", phase="splash"),
        AppStartupServiceSpec(id="post", entrypoint="pkg:post", surface="gui", phase="post_gui_show"),
    ]
    manager = AppStartupServiceManager.from_specs(specs)

    ordered = manager.ordered_specs(phase="splash")
    assert [s.id for s in ordered] == ["splash"]


def test_app_startup_manager_filters_by_surface_and_phase():
    """Filtering by both surface and phase returns only matching specs."""
    specs = [
        AppStartupServiceSpec(id="a", entrypoint="pkg:a", surface="gui", phase="splash"),
        AppStartupServiceSpec(id="b", entrypoint="pkg:b", surface="server", phase="pre_server_listen"),
        AppStartupServiceSpec(id="c", entrypoint="pkg:c", surface="gui", phase="post_gui_show"),
    ]
    manager = AppStartupServiceManager.from_specs(specs)

    ordered = manager.ordered_specs(surface="gui", phase="splash")
    assert [s.id for s in ordered] == ["a"]


def test_app_startup_spec_from_dict_validates_surface():
    """surface must be a string when provided."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "surface": "gui",
    })
    assert spec.surface == "gui"

    with pytest.raises(ValueError, match="surface"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test", "surface": 42,
        })


def test_app_startup_spec_from_dict_validates_phase():
    """phase must be a string when provided."""
    with pytest.raises(ValueError, match="phase"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test", "phase": 42,
        })


def test_app_startup_spec_from_dict_validates_progress():
    """progress must be an integer in range 0-100."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "progress": 50,
    })
    assert spec.progress == 50

    with pytest.raises(ValueError, match="progress"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test", "progress": 150,
        })

    with pytest.raises(ValueError, match="progress"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test", "progress": "high",
        })


def test_app_startup_spec_from_dict_validates_thread():
    """thread must be 'main' or 'background'."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "thread": "background",
    })
    assert spec.thread == "background"

    with pytest.raises(ValueError, match="thread"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test", "thread": "other",
        })


def test_app_startup_spec_from_dict_validates_label():
    """label must be a string when provided."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "label": "My Service",
    })
    assert spec.label == "My Service"


def test_app_startup_spec_from_dict_validates_requires():
    """requires must be a list of strings."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "requires": ["main_window"],
    })
    assert spec.requires == ("main_window",)

    with pytest.raises(ValueError, match="requires"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test", "requires": "not_a_list",
        })


def test_app_startup_manager_start_respects_stage_callbacks():
    """start() invokes on_stage_start and on_stage_finish for each service."""
    calls = []
    specs = [
        AppStartupServiceSpec(id="a", entrypoint="pkg:a", label="Service A", progress=25),
        AppStartupServiceSpec(id="b", entrypoint="pkg:b", label="Service B", progress=50),
    ]

    def on_start(payload):
        calls.append(("start", payload["id"], payload["label"], payload["progress"]))

    def on_finish(payload):
        calls.append(("finish", payload["id"]))

    class _FakeLoader:
        def load(self, entrypoint):
            def fake_service(context):
                context.mark_ready()
            return fake_service

    manager = AppStartupServiceManager.from_specs(specs, loader=_FakeLoader())
    manager.start(on_stage_start=on_start, on_stage_finish=on_finish)

    assert calls == [
        ("start", "a", "Service A", 25),
        ("finish", "a"),
        ("start", "b", "Service B", 50),
        ("finish", "b"),
    ]


def test_app_startup_manager_get_service_result():
    """get_service_result returns the result of a completed service."""
    def make_service(value):
        def svc(context):
            context.mark_ready()
            return value
        return svc

    class _TestLoader:
        _values = iter(["result_a", "result_b"])
        def load(self, entrypoint):
            val = next(self._values)
            return lambda ctx: val

    specs = [
        AppStartupServiceSpec(id="a", entrypoint="pkg:a"),
        AppStartupServiceSpec(id="b", entrypoint="pkg:b"),
    ]
    manager = AppStartupServiceManager.from_specs(specs, loader=_TestLoader())
    manager.start()

    assert manager.get_service_result("a") == "result_a"
    assert manager.get_service_result("b") == "result_b"
    assert manager.get_service_result("nonexistent") is None


# ── enabled_if validation tests ──────────────────────────────────────────

def test_enabled_if_from_dict_setting_source():
    """enabled_if with a setting source parses correctly."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test",
        "entrypoint": "pkg:test",
        "enabled_if": {"setting": "gui.start_jupyter_on_startup", "equals": True},
    })
    assert spec.enabled_if is not None
    assert spec.enabled_if.source_type == "setting"
    assert spec.enabled_if.source_key == "gui.start_jupyter_on_startup"
    assert spec.enabled_if.equals is True


def test_enabled_if_from_dict_env_source():
    """enabled_if with an env source parses correctly."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test",
        "entrypoint": "pkg:test",
        "enabled_if": {"env": "CHISURF_FEATURE_X", "exists": True},
    })
    assert spec.enabled_if is not None
    assert spec.enabled_if.source_type == "env"
    assert spec.enabled_if.source_key == "CHISURF_FEATURE_X"
    assert spec.enabled_if.exists is True


def test_enabled_if_from_dict_attribute_source():
    """enabled_if with an attribute source parses correctly."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test",
        "entrypoint": "pkg:test",
        "enabled_if": {"attribute": "metadata.experimental", "truthy": True},
    })
    assert spec.enabled_if is not None
    assert spec.enabled_if.source_type == "attribute"
    assert spec.enabled_if.source_key == "metadata.experimental"
    assert spec.enabled_if.truthy is True


def test_enabled_if_from_dict_missing_source_key():
    """enabled_if without exactly one source key raises."""
    with pytest.raises(ValueError, match="exactly one source key"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test",
            "enabled_if": {"equals": True},
        })


def test_enabled_if_from_dict_missing_comparison_key():
    """enabled_if without at least one comparison key raises."""
    with pytest.raises(ValueError, match="at least one comparison"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test",
            "enabled_if": {"setting": "x"},
        })


def test_enabled_if_from_dict_multiple_source_keys():
    """enabled_if with multiple source keys raises."""
    with pytest.raises(ValueError, match="exactly one source key"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test",
            "enabled_if": {"setting": "x", "env": "Y"},
        })


def test_enabled_if_from_dict_non_object():
    """enabled_if must be an object."""
    with pytest.raises(ValueError, match="enabled_if.*object"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test", "enabled_if": "yes",
        })


def test_run_if_dependency_skipped_validation():
    """run_if_dependency_skipped must be a boolean."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "run_if_dependency_skipped": True,
    })
    assert spec.run_if_dependency_skipped is True

    with pytest.raises(ValueError, match="boolean"):
        AppStartupServiceSpec.from_dict({
            "id": "test", "entrypoint": "pkg:test", "run_if_dependency_skipped": "yes",
        })


# ── enabled_if evaluation tests ──────────────────────────────────────────

def test_enabled_if_setting_equals_true_skips_when_false():
    """A service with setting equals true is skipped when the setting is falsy."""
    import chisurf.core.settings
    chisurf.core.settings.cs_settings["test_feature"] = False

    spec = AppStartupServiceSpec(
        id="feat",
        entrypoint="pkg:feat",
        enabled_if=_EnabledIf("setting", "test_feature", equals=True),
    )
    manager = AppStartupServiceManager.from_specs([spec])
    enabled = manager.resolve_enabled()
    assert len(enabled) == 0
    skipped = manager.get_skipped_services()
    assert "feat" in skipped
    assert "equals" in skipped["feat"]

    # Clean up
    del chisurf.core.settings.cs_settings["test_feature"]


def test_enabled_if_setting_equals_true_runs_when_true():
    """A service with setting equals true is enabled when the setting is truthy."""
    import chisurf.core.settings
    chisurf.core.settings.cs_settings["test_feature"] = True

    spec = AppStartupServiceSpec(
        id="feat",
        entrypoint="pkg:feat",
        enabled_if=_EnabledIf("setting", "test_feature", equals=True),
    )

    class _FakeLoader:
        def load(self, entrypoint):
            def svc(ctx):
                ctx.mark_ready()
            return svc

    manager = AppStartupServiceManager.from_specs([spec], loader=_FakeLoader())
    enabled = manager.resolve_enabled()
    assert len(enabled) == 1
    assert enabled[0].id == "feat"
    assert manager.get_skipped_services() == {}

    del chisurf.core.settings.cs_settings["test_feature"]


def test_enabled_if_env_exists_skips_when_missing():
    """A service with env exists is skipped when the env var is not set."""
    spec = AppStartupServiceSpec(
        id="feat",
        entrypoint="pkg:feat",
        enabled_if=_EnabledIf("env", "CHISURF_TEST_FEATURE", exists=True),
    )
    manager = AppStartupServiceManager.from_specs([spec])
    enabled = manager.resolve_enabled()
    assert len(enabled) == 0


def test_enabled_if_env_exists_runs_when_set():
    """A service with env exists runs when the env var is set."""
    import os
    os.environ["CHISURF_TEST_FEATURE"] = "1"

    spec = AppStartupServiceSpec(
        id="feat",
        entrypoint="pkg:feat",
        enabled_if=_EnabledIf("env", "CHISURF_TEST_FEATURE", exists=True),
    )

    class _FakeLoader:
        def load(self, entrypoint):
            def svc(ctx):
                ctx.mark_ready()
            return svc

    manager = AppStartupServiceManager.from_specs([spec], loader=_FakeLoader())
    enabled = manager.resolve_enabled()
    assert len(enabled) == 1

    del os.environ["CHISURF_TEST_FEATURE"]


def test_enabled_if_not_exists_skips_when_var_is_set():
    """A service with env exists=false is skipped when the var IS set."""
    import os
    os.environ["CHISURF_TEST_SKIP"] = "1"

    spec = AppStartupServiceSpec(
        id="feat",
        entrypoint="pkg:feat",
        enabled_if=_EnabledIf("env", "CHISURF_TEST_SKIP", exists=False),
    )
    manager = AppStartupServiceManager.from_specs([spec])
    enabled = manager.resolve_enabled()
    assert len(enabled) == 0

    del os.environ["CHISURF_TEST_SKIP"]


# ── Dependency skipping tests ────────────────────────────────────────────

def test_dependency_skipped_skips_dependent():
    """A dependent of a skipped service is also skipped."""
    specs = [
        AppStartupServiceSpec(
            id="parent",
            entrypoint="pkg:parent",
            enabled_if=_EnabledIf("env", "NONEXISTENT_VAR_FOR_TEST", exists=True),
        ),
        AppStartupServiceSpec(
            id="child",
            entrypoint="pkg:child",
            depends_on=("parent",),
        ),
    ]
    manager = AppStartupServiceManager.from_specs(specs)
    enabled = manager.resolve_enabled()
    assert len(enabled) == 0
    skipped = manager.get_skipped_services()
    assert "parent" in skipped
    assert "child" in skipped
    assert "dependency" in skipped["child"]


def test_run_if_dependency_skipped_allows_independent():
    """run_if_dependency_skipped=True allows a service to run even if its dependency is skipped."""
    specs = [
        AppStartupServiceSpec(
            id="parent",
            entrypoint="pkg:parent",
            enabled_if=_EnabledIf("env", "NONEXISTENT_VAR_FOR_TEST", exists=True),
        ),
        AppStartupServiceSpec(
            id="child",
            entrypoint="pkg:child",
            depends_on=("parent",),
            run_if_dependency_skipped=True,
        ),
    ]

    class _FakeLoader:
        def load(self, entrypoint):
            def svc(ctx):
                ctx.mark_ready()
            return svc

    manager = AppStartupServiceManager.from_specs(specs, loader=_FakeLoader())
    enabled = manager.resolve_enabled()
    assert len(enabled) == 1
    assert enabled[0].id == "child"


def test_entrypoints_always_returns_all_configured():
    """entrypoints returns all configured entrypoints (configured-owner semantics)."""
    import chisurf.core.settings
    chisurf.core.settings.cs_settings["test_flag"] = False

    specs = [
        AppStartupServiceSpec(
            id="enabled_svc",
            entrypoint="pkg:enabled",
        ),
        AppStartupServiceSpec(
            id="disabled_svc",
            entrypoint="pkg:disabled",
            enabled_if=_EnabledIf("setting", "test_flag", equals=True),
        ),
    ]

    class _FakeLoader:
        def load(self, entrypoint):
            def svc(ctx):
                ctx.mark_ready()
            return svc

    manager = AppStartupServiceManager.from_specs(specs, loader=_FakeLoader())
    # Before start: all entrypoints present
    assert manager.entrypoints == {"pkg:enabled", "pkg:disabled"}

    # After start with evaluation: still all entrypoints (configured-owner)
    manager.start()
    assert manager.entrypoints == {"pkg:enabled", "pkg:disabled"}

    del chisurf.core.settings.cs_settings["test_flag"]


def test_on_stage_skip_callback_invoked():
    """start() invokes on_stage_skip for each skipped service."""
    skip_calls = []

    specs = [
        AppStartupServiceSpec(
            id="skip_me",
            entrypoint="pkg:skip",
            label="Will Skip",
            progress=50,
            enabled_if=_EnabledIf("env", "NONEXISTENT_VAR_FOR_TEST", exists=True),
        ),
    ]

    class _FakeLoader:
        def load(self, entrypoint):
            def svc(ctx):
                ctx.mark_ready()
            return svc

    def on_skip(payload):
        skip_calls.append(payload)

    manager = AppStartupServiceManager.from_specs(specs, loader=_FakeLoader())
    manager.start(on_stage_skip=on_skip)

    assert len(skip_calls) == 1
    p = skip_calls[0]
    assert p["id"] == "skip_me"
    assert p["skipped"] is True
    assert "skip_reason" in p
    assert p["label"] == "Will Skip"
    assert p["progress"] == 50


def test_get_skipped_services_returns_reasons():
    """get_skipped_services returns id->reason mapping."""
    specs = [
        AppStartupServiceSpec(
            id="skip_me",
            entrypoint="pkg:skip",
            enabled_if=_EnabledIf("env", "NONEXISTENT_VAR_FOR_TEST", exists=True),
        ),
    ]
    manager = AppStartupServiceManager.from_specs(specs)
    manager.resolve_enabled()

    skipped = manager.get_skipped_services()
    assert "skip_me" in skipped
    assert isinstance(skipped["skip_me"], str)
    assert len(skipped["skip_me"]) > 0


def test_resolve_enabled_returns_only_enabled():
    """resolve_enabled returns only specs that pass their conditions."""
    specs = [
        AppStartupServiceSpec(id="a", entrypoint="pkg:a"),
        AppStartupServiceSpec(
            id="b",
            entrypoint="pkg:b",
            enabled_if=_EnabledIf("env", "NONEXISTENT_VAR_FOR_TEST", exists=True),
        ),
        AppStartupServiceSpec(id="c", entrypoint="pkg:c"),
    ]
    manager = AppStartupServiceManager.from_specs(specs)
    enabled = manager.resolve_enabled()
    assert [s.id for s in enabled] == ["a", "c"]


def test_config_has_jupyter_enabled_if():
    """The Jupyter service in the default config has an enabled_if setting gate."""
    specs = load_app_startup_services()
    jupyter = next((s for s in specs if s.id == "gui.start_jupyter"), None)
    assert jupyter is not None, "gui.start_jupyter not found in config"
    assert jupyter.enabled_if is not None
    assert jupyter.enabled_if.source_type == "setting"
    assert jupyter.enabled_if.source_key == "gui.start_jupyter_on_startup"
    assert jupyter.enabled_if.equals is True


# ── Deprecated background alias ─────────────────────────────────────────

def test_background_deprecated_alias_maps_to_thread():
    """'background' in from_dict maps to thread field."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "background": True,
    })
    assert spec.thread == "background"

    spec_false = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "background": False,
    })
    assert spec_false.thread == "main"


def test_thread_field_takes_precedence_over_background():
    """'thread' takes precedence when both thread and background are set."""
    spec = AppStartupServiceSpec.from_dict({
        "id": "test", "entrypoint": "pkg:test", "thread": "main", "background": True,
    })
    assert spec.thread == "main"


# ── Env equals coercion ────────────────────────────────────────────────

def test_env_equals_boolean_coercion():
    """Env equals with bool coerces common truthy/falsy strings."""
    import os
    os.environ["CHISURF_TEST_BOOL"] = "true"

    spec_true = AppStartupServiceSpec(
        id="feat",
        entrypoint="pkg:feat",
        enabled_if=_EnabledIf("env", "CHISURF_TEST_BOOL", equals=True),
    )
    manager = AppStartupServiceManager.from_specs([spec_true])
    enabled = manager.resolve_enabled()
    assert len(enabled) == 1

    del os.environ["CHISURF_TEST_BOOL"]


def test_env_equals_boolean_false_coercion():
    """Env equals false with '0' string evaluates correctly."""
    import os
    os.environ["CHISURF_TEST_BOOL"] = "0"

    spec_false = AppStartupServiceSpec(
        id="feat",
        entrypoint="pkg:feat",
        enabled_if=_EnabledIf("env", "CHISURF_TEST_BOOL", equals=False),
    )
    manager = AppStartupServiceManager.from_specs([spec_false])
    enabled = manager.resolve_enabled()
    assert len(enabled) == 1

    del os.environ["CHISURF_TEST_BOOL"]


def test_env_equals_string_no_coercion():
    """Env equals with string compares literally."""
    import os
    os.environ["CHISURF_TEST_STR"] = "production"

    spec = AppStartupServiceSpec(
        id="feat",
        entrypoint="pkg:feat",
        enabled_if=_EnabledIf("env", "CHISURF_TEST_STR", equals="production"),
    )
    manager = AppStartupServiceManager.from_specs([spec])
    enabled = manager.resolve_enabled()
    assert len(enabled) == 1

    del os.environ["CHISURF_TEST_STR"]


# ── Attribute condition deferred evaluation ────────────────────────────

def test_attribute_condition_deferred_to_execution():
    """Attribute conditions are not evaluated upfront by resolve_enabled()."""
    spec = AppStartupServiceSpec(
        id="attr_svc",
        entrypoint="pkg:attr",
        enabled_if=_EnabledIf("attribute", "main_window.visible", truthy=True),
    )
    manager = AppStartupServiceManager.from_specs([spec])
    # resolve_enabled includes attribute-gated services (can't evaluate yet)
    enabled = manager.resolve_enabled()
    assert len(enabled) == 1
    assert enabled[0].id == "attr_svc"


def test_attribute_condition_skips_at_runtime_when_context_lacks_attr():
    """Attribute condition skips the service at runtime when the context lacks the attribute."""
    spec = AppStartupServiceSpec(
        id="attr_svc",
        entrypoint="pkg:attr",
        enabled_if=_EnabledIf("attribute", "main_window.nonexistent", truthy=True),
    )

    class _FakeLoader:
        def load(self, entrypoint):
            def svc(ctx):
                ctx.mark_ready()
            return svc

    manager = AppStartupServiceManager.from_specs([spec], loader=_FakeLoader())
    manager.start()
    skipped = manager.get_skipped_services()
    assert "attr_svc" in skipped


def test_attribute_condition_runs_when_context_has_attr():
    """Attribute condition allows the service when the shared context has the expected attribute."""
    class _Window:
        visible = True

    started = []

    class _Loader:
        def load(self, entrypoint):
            def svc(ctx):
                started.append(entrypoint)
                ctx.main_window = _Window()
                ctx.mark_ready()
            return svc

    # Service that sets main_window
    producer = AppStartupServiceSpec(
        id="startup_interface",
        entrypoint="pkg:startup_interface",
    )
    # Service that checks main_window attribute
    consumer = AppStartupServiceSpec(
        id="attr_svc",
        entrypoint="pkg:attr",
        enabled_if=_EnabledIf("attribute", "main_window.visible", truthy=True),
        depends_on=("startup_interface",),
    )

    manager = AppStartupServiceManager.from_specs([producer, consumer], loader=_Loader())
    manager.start()
    skipped = manager.get_skipped_services()
    assert "attr_svc" not in skipped
    assert "pkg:attr" in started

"""Headless tests for the AutoForm-based AI Settings plugin."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Redirect the AI settings file to a tmp path so auto-save never touches ~/.chisurf."""
    from chisurf.core.settings import ai_settings

    monkeypatch.setattr(
        ai_settings, "_get_settings_path", lambda: tmp_path / "ai_api_settings.json"
    )


# --- backing-model tests (no Qt) -----------------------------------------------
class TestAISettingsModel:
    """The GUI-free AISettingsModel: provider/network/persistence behaviour."""

    def test_view_spec_has_collapsible_panels(self):
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        view = AISettingsModel().view_spec()
        titles = [getattr(s, "title", None) for s in view.sections]
        assert "API Configuration" in titles
        assert "Models" in titles
        assert "Generation Settings" in titles

    def test_set_provider_loads_provider_defaults(self):
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        model = AISettingsModel()
        model.set_provider("mistral")
        assert model.provider == "mistral"
        assert "mistral" in model.base_url

    def test_save_separates_text_and_image_models(self, monkeypatch):
        from chisurf.core.settings import ai_settings
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        saved: dict = {}
        monkeypatch.setattr(
            ai_settings, "save_api_settings", lambda settings: saved.update(settings) or True
        )

        model = AISettingsModel()
        model.text_model = "text-model"
        model.image_model = "image-model"
        model.save()

        assert saved["text_model"] == "text-model"
        assert saved["image_model"] == "image-model"

    def test_reset_restores_defaults(self, monkeypatch):
        from chisurf.core.settings import ai_settings
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        monkeypatch.setattr(ai_settings, "save_api_settings", lambda settings: True)
        model = AISettingsModel()
        model.set_provider("openai")
        model.api_key = "new-key"
        model.base_url = "http://custom.url"
        model.reset()
        assert model.api_key == ""
        assert model.base_url == "https://api.openai.com/v1"

    def test_fetch_models_populates_option_sources(self, monkeypatch):
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        class Response:
            status_code = 200

            def json(self):
                return {"data": [{"id": "chat-model"}, {"id": "image-model"}]}

        monkeypatch.setitem(
            sys.modules, "requests", SimpleNamespace(get=lambda *a, **k: Response())
        )

        model = AISettingsModel()
        model.base_url = "https://example.invalid/v1"
        model.fetch_models()

        assert "chat-model" in model.available_text_models()
        assert "image-model" in model.available_image_models()

    def test_on_change_fires_on_status_update(self, monkeypatch):
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        model = AISettingsModel()
        calls = []
        model.on_change = lambda: calls.append(1)
        model.set_provider("local")
        assert calls  # provider switch notified the view

    def test_editing_a_field_auto_persists(self):
        from chisurf.core.settings import ai_settings
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        model = AISettingsModel()
        model.set_provider("openai")
        model.api_key = "sk-pasted-token"  # a plain edit, no Save click
        # It is on disk immediately, under the current provider.
        assert ai_settings.get_api_settings("openai")["api_key"] == "sk-pasted-token"

    def test_apply_token_tests_connection(self, monkeypatch):
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        monkeypatch.setattr(
            AISettingsModel, "_get_models", staticmethod(lambda *a, **k: {"data": []})
        )
        model = AISettingsModel()
        model.set_provider("openai")
        model.api_key = "sk-token"
        model.apply_token()
        assert "successful" in model.status_html.lower()

    def test_provider_switch_does_not_carry_key_across_providers(self):
        from chisurf.core.settings import ai_settings
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        model = AISettingsModel()
        model.set_provider("openai")
        model.api_key = "openai-only-key"  # auto-saved under openai
        model.set_provider("mistral")  # loads mistral; must not inherit the key
        assert model.api_key == ""
        assert ai_settings.get_api_settings("mistral")["api_key"] != "openai-only-key"

    def test_async_runner_is_used_for_network(self, monkeypatch):
        from chisurf.plugins.ai_settings.gui.model import AISettingsModel

        monkeypatch.setattr(
            AISettingsModel, "_get_models", staticmethod(lambda *a, **k: {"data": []})
        )
        model = AISettingsModel()
        model.set_provider("openai")

        used = []

        def runner(work, done):
            used.append(True)
            done(work())  # synchronous stand-in for the real off-thread runner

        model.async_runner = runner
        model.test_connection()
        assert used == [True]
        assert "successful" in model.status_html.lower()


# --- widget/AutoForm rendering tests -------------------------------------------
class TestAISettingsWidget:
    """The AutoForm-mounted widget renders the secret + editable controls."""

    def test_creation_mounts_autoform(self, qapp, qtbot):
        from chisurf.gui.autoform import AutoForm
        from chisurf.plugins.ai_settings.gui.tool import AISettingsWidget

        widget = AISettingsWidget()
        qtbot.addWidget(widget)
        assert isinstance(widget.form, AutoForm)
        assert widget.model is not None

    def test_secret_api_key_field_masked_with_reveal(self, qapp, qtbot):
        from qtpy import QtWidgets

        from chisurf.gui.autoform.sections.builtin import ValueWidget
        from chisurf.plugins.ai_settings.gui.tool import AISettingsWidget

        widget = AISettingsWidget()
        qtbot.addWidget(widget)
        key_field = next(
            vw
            for vw in widget.form.findChildren(ValueWidget)
            if getattr(vw._section, "attr", None) == "api_key"
        )
        assert key_field.editor.echoMode() == QtWidgets.QLineEdit.Password
        key_field.reveal.setChecked(True)
        assert key_field.editor.echoMode() == QtWidgets.QLineEdit.Normal

    def test_model_combos_are_editable(self, qapp, qtbot):
        from chisurf.gui.autoform.sections.builtin import ChoiceWidget
        from chisurf.plugins.ai_settings.gui.tool import AISettingsWidget

        widget = AISettingsWidget()
        qtbot.addWidget(widget)
        editable_attrs = {
            cw._section.attr
            for cw in widget.form.findChildren(ChoiceWidget)
            if cw.combo is not None and cw.combo.isEditable()
        }
        assert {"text_model", "image_model"} <= editable_attrs

    def test_connection_runs_off_the_ui_thread(self, qapp, qtbot, monkeypatch):
        import threading

        from chisurf.plugins.ai_settings.gui.model import AISettingsModel
        from chisurf.plugins.ai_settings.gui.tool import AISettingsWidget

        seen = {}

        def fake_get_models(base_url, api_key):
            seen["worker"] = threading.current_thread()
            return {"data": []}

        monkeypatch.setattr(AISettingsModel, "_get_models", staticmethod(fake_get_models))

        widget = AISettingsWidget()
        qtbot.addWidget(widget)
        widget.model.set_provider("openai")
        widget.model.api_key = "sk-x"
        widget.model.test_connection()

        qtbot.waitUntil(lambda: "successful" in widget.model.status_html.lower(), timeout=3000)
        # The blocking request ran on a background thread, not the GUI thread.
        assert seen["worker"] is not threading.main_thread()

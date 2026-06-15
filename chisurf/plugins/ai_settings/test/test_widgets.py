class TestAISettingsWidget:
    def test_creation(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        widget = AISettingsWidget()
        assert widget is not None

    def test_has_provider_combo(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from qtpy.QtWidgets import QComboBox
        widget = AISettingsWidget()
        assert isinstance(widget.provider_combo, QComboBox)

    def test_has_base_url_input(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from qtpy.QtWidgets import QLineEdit
        widget = AISettingsWidget()
        assert isinstance(widget.base_url_input, QLineEdit)

    def test_has_model_combo(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from qtpy.QtWidgets import QComboBox
        widget = AISettingsWidget()
        assert isinstance(widget.model_combo, QComboBox)

    def test_has_image_model_combo(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from qtpy.QtWidgets import QComboBox
        widget = AISettingsWidget()
        assert isinstance(widget.image_model_input, QComboBox)

    def test_has_api_key_input(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from qtpy.QtWidgets import QLineEdit
        widget = AISettingsWidget()
        assert isinstance(widget.api_key_input, QLineEdit)

    def test_has_signin_button(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from qtpy.QtWidgets import QPushButton
        widget = AISettingsWidget()
        assert isinstance(widget.signin_button, QPushButton)

    def test_provider_local_disables_api_key(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        widget = AISettingsWidget()
        # Select "Local" provider
        idx = widget.provider_combo.findData("local")
        if idx >= 0:
            widget.provider_combo.setCurrentIndex(idx)
            assert not widget.api_key_input.isEnabled()
            assert not widget.signin_button.isEnabled()

    def test_provider_openai_enables_api_key(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        widget = AISettingsWidget()
        idx = widget.provider_combo.findData("openai")
        if idx >= 0:
            widget.provider_combo.setCurrentIndex(idx)
            assert widget.api_key_input.isEnabled()
            assert widget.signin_button.isEnabled()

    def test_reset_to_defaults(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from chisurf.core.settings import ai_settings
        # Set up a known initial state
        ai_settings.save_api_settings({
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4",
            "api_key": "test-key",
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048,
        })
        widget = AISettingsWidget()
        widget.api_key_input.setText("new-key")
        widget.base_url_input.setText("http://custom.url")
        widget.reset_to_defaults()
        assert widget.api_key_input.text() == ""
        assert widget.base_url_input.text() == "https://api.openai.com/v1"

    def test_save_settings_separates_text_and_image_models(self, qapp, monkeypatch):
        from chisurf.core.settings import ai_settings
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget

        saved = {}
        monkeypatch.setattr(ai_settings, "save_api_settings", lambda settings: saved.update(settings) or True)

        widget = AISettingsWidget()
        widget.model_combo.setEditText("text-model")
        widget.image_model_input.setEditText("image-model")
        widget.save_settings()

        assert saved["text_model"] == "text-model"
        assert saved["image_model"] == "image-model"
        assert "model" not in saved

    def test_fetch_models_populates_text_and_image_model_combos(self, qapp, monkeypatch):
        import sys
        from types import SimpleNamespace

        from chisurf.plugins.ai_settings.plugin import AISettingsWidget

        class Response:
            status_code = 200

            def json(self):
                return {"data": [{"id": "chat-model"}, {"id": "image-model"}]}

        monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=lambda *args, **kwargs: Response()))

        widget = AISettingsWidget()
        widget.base_url_input.setText("https://example.invalid/v1")
        widget.model_combo.setEditText("chat-model")
        widget.image_model_input.setEditText("image-model")

        widget._fetch_models()

        assert widget.model_combo.findText("chat-model") >= 0
        assert widget.image_model_input.findText("image-model") >= 0

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

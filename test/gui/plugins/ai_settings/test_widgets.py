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

    def test_has_chat_model_combo(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from qtpy.QtWidgets import QComboBox
        widget = AISettingsWidget()
        assert isinstance(widget.chat_model_combo, QComboBox)

    def test_has_api_key_input(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        from qtpy.QtWidgets import QLineEdit
        widget = AISettingsWidget()
        assert isinstance(widget.api_key_input, QLineEdit)

    def test_reset_to_defaults(self, qapp):
        from chisurf.plugins.ai_settings.plugin import AISettingsWidget
        widget = AISettingsWidget()
        widget.api_key_input.setText("test-key")
        widget.base_url_input.setText("http://custom.url")
        widget.reset_to_defaults()
        assert widget.api_key_input.text() == ""
        assert widget.base_url_input.text() == "https://api.mistral.ai/v1"

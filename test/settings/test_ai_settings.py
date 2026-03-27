import os
import pathlib
import tempfile
import json
import pytest
from unittest.mock import patch, MagicMock
from qtpy import QtWidgets

from chisurf.settings import ai_settings
from chisurf.plugins.ai_settings.plugin import AISettingsWidget, PROVIDER_OPTIONS, DEFAULT_BASE_URLS, DEFAULT_MODELS


class TestAISettingsModule:
    """Tests for the ai_settings module."""

    def test_default_settings_structure(self):
        """Test that default settings have the correct structure."""
        defaults = ai_settings.DEFAULT_SETTINGS
        assert "provider" in defaults
        assert "base_url" in defaults
        assert "model" in defaults
        assert "api_key" in defaults

    def test_get_provider_returns_valid_provider(self):
        """Test that get_provider returns a valid provider string."""
        valid_providers = ["openai", "mistral", "anthropic", "local"]
        provider = ai_settings.get_provider()
        assert provider in valid_providers

    def test_get_base_url_returns_string(self):
        """Test that get_base_url returns a string."""
        url = ai_settings.get_base_url()
        assert isinstance(url, str)

    def test_get_model_returns_string(self):
        """Test that get_model returns a string."""
        model = ai_settings.get_model()
        assert isinstance(model, str)

    def test_get_api_settings_returns_dict(self):
        """Test that get_api_settings returns a dictionary."""
        settings = ai_settings.get_api_settings()
        assert isinstance(settings, dict)
        assert "provider" in settings
        assert "base_url" in settings
        assert "model" in settings
        assert "api_key" in settings


class TestAISettingsSaveLoad:
    """Tests for save/load functionality with temp directory."""

    def test_save_settings_creates_file(self):
        """Test that save_settings creates a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)

            with patch.object(ai_settings, 'get_path', return_value=tmp_path):
                test_settings = {
                    "provider": "mistral",
                    "base_url": "https://api.mistral.ai/v1",
                    "model": "mistral-large",
                    "api_key": "test-key-123",
                }
                result = ai_settings.save_api_settings(test_settings)
                assert result is True

                expected_file = tmp_path / "ai_api_settings.json"
                assert expected_file.exists()

    def test_load_settings_reads_file(self):
        """Test that load_settings reads from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            test_file = tmp_path / "ai_api_settings.json"

            test_settings = {
                "provider": "anthropic",
                "base_url": "https://api.anthropic.com",
                "model": "claude-3",
                "api_key": "test-anthropic-key",
            }
            with open(test_file, 'w') as f:
                json.dump(test_settings, f)

            with patch.object(ai_settings, 'get_path', return_value=tmp_path):
                loaded = ai_settings.get_api_settings()
                assert loaded["provider"] == "anthropic"
                assert loaded["base_url"] == "https://api.anthropic.com"
                assert loaded["model"] == "claude-3"
                assert loaded["api_key"] == "test-anthropic-key"

    def test_save_and_load_roundtrip(self):
        """Test save and load together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)

            original_get_path = ai_settings.get_path

            def mock_path(name):
                if name == 'settings':
                    return tmp_path
                return original_get_path(name)

            with patch.object(ai_settings, 'get_path', side_effect=mock_path):
                test_settings = {
                    "provider": "local",
                    "base_url": "http://localhost:1234",
                    "model": "llama-2",
                    "api_key": "local-key",
                }
                ai_settings.save_api_settings(test_settings)
                loaded = ai_settings.get_api_settings()

                assert loaded["provider"] == "local"
                assert loaded["base_url"] == "http://localhost:1234"
                assert loaded["model"] == "llama-2"
                assert loaded["api_key"] == "local-key"


class TestAISettingsWidget:
    """Tests for the AI Settings widget."""

    @pytest.fixture
    def app(self):
        """Create QApplication instance for Qt tests."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])
        yield app

    def test_widget_creation(self, app):
        """Test that the widget can be created."""
        widget = AISettingsWidget()
        assert widget is not None

    def test_widget_has_required_fields(self, app):
        """Test that widget has all required input fields."""
        widget = AISettingsWidget()
        assert hasattr(widget, 'provider_combo')
        assert hasattr(widget, 'base_url_input')
        assert hasattr(widget, 'model_input')
        assert hasattr(widget, 'api_key_input')

    def test_provider_options_populated(self, app):
        """Test that provider options are populated."""
        widget = AISettingsWidget()
        assert widget.provider_combo.count() == len(PROVIDER_OPTIONS)

    def test_default_base_urls_defined(self):
        """Test default base URLs for all providers."""
        assert "openai" in DEFAULT_BASE_URLS
        assert "mistral" in DEFAULT_BASE_URLS
        assert "anthropic" in DEFAULT_BASE_URLS
        assert "local" in DEFAULT_BASE_URLS

    def test_default_models_defined(self):
        """Test default models for all providers."""
        assert "openai" in DEFAULT_MODELS
        assert "mistral" in DEFAULT_MODELS
        assert "anthropic" in DEFAULT_MODELS
        assert "local" in DEFAULT_MODELS

    def test_provider_change_emits_signal(self, app):
        """Test that provider change triggers callback."""
        widget = AISettingsWidget()
        called = []

        def on_change(index):
            called.append(index)

        widget.provider_combo.currentIndexChanged.connect(on_change)
        widget.provider_combo.setCurrentIndex(1)
        widget.provider_combo.setCurrentIndex(2)

        assert len(called) >= 1

    def test_reset_clears_inputs(self, app):
        """Test reset to defaults clears all inputs."""
        widget = AISettingsWidget()

        widget.provider_combo.setCurrentIndex(1)
        widget.base_url_input.setText("test-url")
        widget.model_input.setText("test-model")
        widget.api_key_input.setText("test-key")

        widget.reset_to_defaults()

        assert widget.base_url_input.text() == ""
        assert widget.model_input.text() == ""
        assert widget.api_key_input.text() == ""


class TestAISettingsIntegration:
    """Integration tests for AI settings plugin."""

    def test_plugin_load_function_returns_widget(self):
        """Test that plugin load returns a widget."""
        from chisurf.plugins.ai_settings import load
        widget = load()
        assert isinstance(widget, AISettingsWidget)

    def test_plugin_name_defined(self):
        """Test that plugin name is defined."""
        from chisurf.plugins import ai_settings
        assert hasattr(ai_settings, 'name')
        assert ai_settings.name == "Tools:AI Settings"

    def test_plugin_icon_defined(self):
        """Test that plugin icon is defined."""
        from chisurf.plugins import ai_settings
        assert hasattr(ai_settings, 'icon')
        assert ai_settings.icon == "🤖"

    def test_plugin_exports_widget(self):
        """Test that widget is exported."""
        from chisurf.plugins.ai_settings import AISettingsWidget as ImportedWidget
        assert ImportedWidget is AISettingsWidget

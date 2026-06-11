from __future__ import annotations

import logging
import webbrowser

from qtpy import QtCore, QtWidgets

from chisurf.core.settings import ai_settings
from chisurf.core.settings.ai_settings import PROVIDERS

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    def persist_plugin_state(name):
        """Fallback no-op decorator."""
        def decorator(cls):
            return cls
        return decorator


_LOG = logging.getLogger(__name__)


@persist_plugin_state("ai_settings")
class AISettingsWidget(QtWidgets.QWidget):
    """Widget for configuring AI LLM API settings."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setup_ui()
        self.load_settings()

    def setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        title_label = QtWidgets.QLabel("<h2>AI Settings</h2>")
        layout.addWidget(title_label)

        info_label = QtWidgets.QLabel(
            "Configure remote LLM endpoints. "
            "Select a provider, sign in to get an API key, or enter a custom endpoint."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # API Configuration Group
        api_group = QtWidgets.QGroupBox("API Configuration")
        api_layout = QtWidgets.QFormLayout(api_group)
        api_layout.setSpacing(10)

        # Provider
        self.provider_combo = QtWidgets.QComboBox()
        for display_name, (key, _url, _api_url, _env) in PROVIDERS.items():
            self.provider_combo.addItem(display_name, key)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        api_layout.addRow("Provider:", self.provider_combo)

        # Sign in via browser button
        self.signin_button = QtWidgets.QPushButton("Sign in via Browser")
        self.signin_button.setToolTip("Open the provider's console to get an API key")
        self.signin_button.clicked.connect(self._open_browser_signin)
        api_layout.addRow("", self.signin_button)

        # Base URL
        self.base_url_input = QtWidgets.QLineEdit()
        self.base_url_input.setPlaceholderText("https://api.openai.com/v1")
        api_layout.addRow("Base URL:", self.base_url_input)

        # API Key
        self.api_key_input = QtWidgets.QLineEdit()
        self.api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Enter API key (optional for local endpoints)")
        api_layout.addRow("API Key:", self.api_key_input)

        # Show/Hide API key button
        self.toggle_key_visibility = QtWidgets.QPushButton("Show")
        self.toggle_key_visibility.setFixedWidth(50)
        self.toggle_key_visibility.setCheckable(True)
        self.toggle_key_visibility.toggled.connect(self._toggle_key_visibility)
        api_layout.addRow("", self.toggle_key_visibility)

        layout.addWidget(api_group)

        # Model Selection Group
        model_group = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QFormLayout(model_group)
        model_layout.setSpacing(10)

        # Model Combo
        model_row_layout = QtWidgets.QHBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setEditable(True)
        model_row_layout.addWidget(self.model_combo)

        self.fetch_models_btn = QtWidgets.QPushButton("Fetch Models")
        self.fetch_models_btn.clicked.connect(self._fetch_models)
        model_row_layout.addWidget(self.fetch_models_btn)

        model_layout.addRow("Model:", model_row_layout)

        layout.addWidget(model_group)

        # Generation Settings Group
        gen_group = QtWidgets.QGroupBox("Generation Settings")
        gen_layout = QtWidgets.QFormLayout(gen_group)
        gen_layout.setSpacing(10)

        # Temperature
        self.temperature = QtWidgets.QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.3)
        gen_layout.addRow("Temperature:", self.temperature)

        # Top-p
        self.top_p = QtWidgets.QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(0.9)
        gen_layout.addRow("Top-p:", self.top_p)

        # Max Tokens
        self.max_tokens = QtWidgets.QSpinBox()
        self.max_tokens.setRange(1, 1000000)
        self.max_tokens.setValue(4096)
        gen_layout.addRow("Max Tokens:", self.max_tokens)

        layout.addWidget(gen_group)

        # Test Connection Button
        self.test_button = QtWidgets.QPushButton("Test Connection")
        self.test_button.clicked.connect(self.test_connection)
        layout.addWidget(self.test_button)

        # Status Label
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Save Button
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.save_button = QtWidgets.QPushButton("Save Settings")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_button)

        self.reset_button = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)
        layout.addStretch()

    def _on_provider_changed(self, index: int) -> None:
        """Update base URL and visibility when provider changes."""
        provider_key = self.provider_combo.currentData()
        # Find the provider info
        for _display, (key, default_url, _api_url, _env) in PROVIDERS.items():
            if key == provider_key:
                self.base_url_input.setText(default_url)
                # Local providers don't need API key
                is_local = provider_key in ("local", "custom")
                self.api_key_input.setEnabled(not is_local)
                self.signin_button.setEnabled(not is_local)
                break

    def _toggle_key_visibility(self, checked: bool) -> None:
        """Toggle API key visibility."""
        if checked:
            self.api_key_input.setEchoMode(QtWidgets.QLineEdit.Normal)
            self.toggle_key_visibility.setText("Hide")
        else:
            self.api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
            self.toggle_key_visibility.setText("Show")

    def _open_browser_signin(self) -> None:
        """Open the provider's console/API key page in the default browser."""
        provider_key = self.provider_combo.currentData()
        for _display, (key, _url, api_url, _env) in PROVIDERS.items():
            if key == provider_key and api_url:
                webbrowser.open(api_url)
                self.status_label.setText(
                    f"<span style='color: blue;'>Opened {api_url} in browser</span>"
                )
                return
        self.status_label.setText(
            "<span style='color: orange;'>No browser sign-in available for this provider</span>"
        )

    def load_settings(self) -> None:
        """Load settings from the ai_settings module."""
        settings = ai_settings.get_api_settings()

        # Provider
        provider = settings.get("provider", "openai")
        idx = self.provider_combo.findData(provider)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)

        # Base URL
        self.base_url_input.setText(settings.get("base_url", ""))

        # API Key
        self.api_key_input.setText(settings.get("api_key", ""))

        # Model
        model = settings.get("model", "")
        if model:
            self.model_combo.setEditText(model)

        # Generation Settings
        self.temperature.setValue(float(settings.get("temperature", 0.3)))
        self.top_p.setValue(float(settings.get("top_p", 0.9)))
        self.max_tokens.setValue(int(settings.get("max_tokens", 4096)))

    def save_settings(self) -> None:
        """Save settings to the ai_settings module."""
        base_url = self.base_url_input.text().strip()
        provider_key = self.provider_combo.currentData()

        # Find default URL for non-custom providers
        for _display, (key, default_url, _api_url, _env) in PROVIDERS.items():
            if key == provider_key:
                if provider_key != "custom" and not base_url:
                    base_url = default_url
                break

        settings = {
            "provider": provider_key,
            "base_url": base_url,
            "api_key": self.api_key_input.text().strip(),
            "model": self.model_combo.currentText().strip(),
            "temperature": self.temperature.value(),
            "top_p": self.top_p.value(),
            "max_tokens": self.max_tokens.value(),
        }

        success = ai_settings.save_api_settings(settings)

        if success:
            self.status_label.setText("<span style='color: green;'>Settings saved successfully!</span>")
        else:
            self.status_label.setText("<span style='color: red;'>Failed to save settings.</span>")

    def reset_to_defaults(self) -> None:
        """Reset settings to defaults."""
        self.provider_combo.setCurrentIndex(0)
        self.base_url_input.setText("https://api.openai.com/v1")
        self.api_key_input.clear()
        self.model_combo.clearEditText()
        self.temperature.setValue(0.3)
        self.top_p.setValue(0.9)
        self.max_tokens.setValue(4096)
        self.status_label.setText("<span style='color: blue;'>Settings reset to defaults (not saved).</span>")

    def test_connection(self) -> None:
        """Test the API connection."""
        self.status_label.setText("<span style='color: blue;'>Testing connection...</span>")
        self.test_button.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        base_url = self.base_url_input.text().strip()
        api_key = self.api_key_input.text().strip()

        if not base_url:
            self.status_label.setText("<span style='color: red;'>No base URL provided.</span>")
            self.test_button.setEnabled(True)
            return

        try:
            self._test_connection(base_url, api_key)
        except Exception as e:
            self.status_label.setText(f"<span style='color: red;'>Connection failed: {str(e)}</span>")
        finally:
            self.test_button.setEnabled(True)

    def _test_connection(self, base_url: str, api_key: str) -> None:
        """Test API connection using the /v1/models endpoint."""
        import requests

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        url = base_url.rstrip('/') + "/models"
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            self.status_label.setText("<span style='color: green;'>Connection successful!</span>")
        else:
            self.status_label.setText(
                f"<span style='color: red;'>API error: {response.status_code} - {response.text[:100]}</span>"
            )

    def _fetch_models(self) -> None:
        """Fetch available models from the API endpoint."""
        base_url = self.base_url_input.text().strip()
        api_key = self.api_key_input.text().strip()

        if not base_url:
            self.status_label.setText("<span style='color: red;'>Enter a base URL first.</span>")
            return

        self.fetch_models_btn.setEnabled(False)
        self.status_label.setText("<span style='color: blue;'>Fetching models...</span>")
        QtWidgets.QApplication.processEvents()

        try:
            import requests

            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            url = base_url.rstrip('/') + "/models"
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 200:
                data = response.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                models.sort()

                current_model = self.model_combo.currentText()
                self.model_combo.clear()
                for model_id in models:
                    self.model_combo.addItem(model_id, model_id)

                # Restore previous selection if still available
                idx = self.model_combo.findText(current_model)
                if idx >= 0:
                    self.model_combo.setCurrentIndex(idx)
                elif current_model:
                    self.model_combo.setEditText(current_model)

                self.status_label.setText(
                    f"<span style='color: green;'>Found {len(models)} models.</span>"
                )
            else:
                self.status_label.setText(
                    f"<span style='color: red;'>Error: {response.status_code}</span>"
                )
        except Exception as e:
            self.status_label.setText(f"<span style='color: red;'>Failed: {str(e)}</span>")
        finally:
            self.fetch_models_btn.setEnabled(True)

from __future__ import annotations

import logging
from typing import Optional, List

from qtpy import QtWidgets, QtCore

from chisurf.core.settings import ai_settings
from chisurf.core.settings.ai_settings import MISTRAL_CHAT_MODELS, MISTRAL_EMBED_MODELS

_LOG = logging.getLogger(__name__)


class AISettingsWidget(QtWidgets.QWidget):
    """Widget for configuring Mistral AI API settings - unified for Chato and coding agent."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._advanced_widgets: List[QtWidgets.QWidget] = []
        self._advanced_labels: List[QtWidgets.QLabel] = []
        self.setup_ui()
        self.load_settings()

    def setup_ui(self) -> None:
        """Set up the user interface with Mistral-only settings."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        title_label = QtWidgets.QLabel("<h2>Mistral AI Settings</h2>")
        layout.addWidget(title_label)

        info_label = QtWidgets.QLabel(
            "Configure Mistral AI for chat and embeddings. "
            "Get your API key from <a href='https://console.mistral.ai/'>console.mistral.ai</a>"
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setSpacing(10)

        # API Key (required)
        self.api_key_input = QtWidgets.QLineEdit()
        self.api_key_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.api_key_input.setPlaceholderText("Your Mistral API key")
        form_layout.addRow("API Key:", self.api_key_input)

        # Base URL
        self.base_url_input = QtWidgets.QLineEdit()
        self.base_url_input.setPlaceholderText("https://api.mistral.ai/v1")
        form_layout.addRow("Base URL:", self.base_url_input)

        # Provider
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItem("Local (llama.cpp)", "local_llm")
        self.provider_combo.addItem("Mistral API", "mistral_api")
        self.provider_combo.addItem("OpenAI API", "openai_api")
        form_layout.addRow("LLM Provider:", self.provider_combo)

        # Chat Model
        self.chat_model_combo = QtWidgets.QComboBox()
        self.chat_model_combo.setEditable(True)
        for model in MISTRAL_CHAT_MODELS:
            self.chat_model_combo.addItem(model, model)
        form_layout.addRow("Chat Model:", self.chat_model_combo)

        # Local LLM settings
        self.hf_repo_input = QtWidgets.QLineEdit()
        self.hf_repo_input.setPlaceholderText("e.g. paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF")
        form_layout.addRow("Local Model Repo:", self.hf_repo_input)

        self.hf_file_input = QtWidgets.QLineEdit()
        self.hf_file_input.setPlaceholderText("e.g. qwen2.5-7b-instruct-q4_k_m.gguf")
        form_layout.addRow("Local Model File:", self.hf_file_input)

        # Temperature
        self.temperature = QtWidgets.QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.1)
        self.temperature.setValue(0.3)
        form_layout.addRow("Temperature:", self.temperature)

        # Top-p
        self.top_p = QtWidgets.QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(0.9)
        form_layout.addRow("Top-p:", self.top_p)

        # Max Tokens
        self.max_tokens = QtWidgets.QSpinBox()
        self.max_tokens.setRange(1, 100000)
        self.max_tokens.setValue(4096)
        form_layout.addRow("Max Tokens:", self.max_tokens)

        # Embedding Models Section
        embed_label = QtWidgets.QLabel("<h3>Embedding Models</h3>")
        layout.addWidget(embed_label)

        embed_layout = QtWidgets.QFormLayout()
        embed_layout.setSpacing(10)

        # Text Embedding Model
        self.text_embed_combo = QtWidgets.QComboBox()
        for model in MISTRAL_EMBED_MODELS.get("text", []):
            self.text_embed_combo.addItem(model, model)
        embed_layout.addRow("Text Embedding:", self.text_embed_combo)

        # Code Embedding Model  
        self.code_embed_combo = QtWidgets.QComboBox()
        for model in MISTRAL_EMBED_MODELS.get("code", []):
            self.code_embed_combo.addItem(model, model)
        embed_layout.addRow("Code Embedding:", self.code_embed_combo)

        layout.addLayout(form_layout)
        layout.addLayout(embed_layout)

        # RAG Settings Section
        rag_label = QtWidgets.QLabel("<h3>RAG & Agent Settings</h3>")
        layout.addWidget(rag_label)

        rag_layout = QtWidgets.QFormLayout()
        rag_layout.setSpacing(10)

        self.persistent_ingest_cb = QtWidgets.QCheckBox("Persistent Index")
        self.persistent_ingest_cb.setChecked(True)
        rag_layout.addRow("Storage:", self.persistent_ingest_cb)

        self.graphrag_enable_cb = QtWidgets.QCheckBox("Enable GraphRAG")
        self.graphrag_enable_cb.setChecked(True)
        rag_layout.addRow("GraphRAG:", self.graphrag_enable_cb)

        self.graphrag_mode_combo = QtWidgets.QComboBox()
        self.graphrag_mode_combo.addItem("Per Chunk", "per_chunk")
        self.graphrag_mode_combo.addItem("Per File", "per_file")
        rag_layout.addRow("GraphRAG Mode:", self.graphrag_mode_combo)

        self.rag_debug_cb = QtWidgets.QCheckBox("Enable RAG Debugging")
        rag_layout.addRow("Debug:", self.rag_debug_cb)

        # Advanced options layout (hidden by default)
        self.advanced_container = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QFormLayout(self.advanced_container)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        
        self.log_level_combo = QtWidgets.QComboBox()
        self.log_level_combo.addItem("DEBUG", logging.DEBUG)
        self.log_level_combo.addItem("INFO", logging.INFO)
        self.log_level_combo.addItem("WARNING", logging.WARNING)
        self.log_level_combo.addItem("ERROR", logging.ERROR)
        advanced_layout.addRow("Log Level:", self.log_level_combo)

        self.rag_max_hits_spin = QtWidgets.QSpinBox()
        self.rag_max_hits_spin.setRange(1, 100)
        self.rag_max_hits_spin.setValue(8)
        advanced_layout.addRow("RAG Debug Max Hits:", self.rag_max_hits_spin)

        self.rag_preview_spin = QtWidgets.QSpinBox()
        self.rag_preview_spin.setRange(10, 2000)
        self.rag_preview_spin.setValue(220)
        advanced_layout.addRow("RAG Preview Chars:", self.rag_preview_spin)
        
        rag_layout.addRow("", self.advanced_container)

        self.advanced_toggle = QtWidgets.QPushButton("Show Advanced")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        rag_layout.addRow("", self.advanced_toggle)
        self.advanced_container.setVisible(False)

        layout.addLayout(rag_layout)

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

    def load_settings(self) -> None:
        """Load settings from the ai_settings module."""
        settings = ai_settings.get_api_settings()

        self.base_url_input.setText(settings.get("base_url", ""))
        
        model = settings.get("model", "mistral-small-latest")
        idx = self.chat_model_combo.findData(model)
        if idx >= 0:
            self.chat_model_combo.setCurrentIndex(idx)
        
        self.api_key_input.setText(settings.get("api_key", ""))
        
        text_model = settings.get("text_embed_model", "mistral-embed")
        idx = self.text_embed_combo.findData(text_model)
        if idx >= 0:
            self.text_embed_combo.setCurrentIndex(idx)
        
        code_model = settings.get("code_embed_model", "codestral-embed")
        idx = self.code_embed_combo.findData(code_model)
        if idx >= 0:
            self.code_embed_combo.setCurrentIndex(idx)
        
        self.temperature.setValue(float(settings.get("temperature", 0.3)))
        self.top_p.setValue(float(settings.get("top_p", 0.9)))
        self.max_tokens.setValue(int(settings.get("max_tokens", 4096)))

        provider = settings.get("provider", "local_llm")
        idx = self.provider_combo.findData(provider)
        if idx >= 0:
            self.provider_combo.setCurrentIndex(idx)

        self.hf_repo_input.setText(settings.get("hf_repo", ""))
        self.hf_file_input.setText(settings.get("hf_file", ""))
        self.persistent_ingest_cb.setChecked(bool(settings.get("persistent_ingest", True)))
        self.graphrag_enable_cb.setChecked(bool(settings.get("graphrag_enable", True)))
        
        gmode = settings.get("graphrag_mode", "per_chunk")
        idx = self.graphrag_mode_combo.findData(gmode)
        if idx >= 0:
            self.graphrag_mode_combo.setCurrentIndex(idx)
            
        self.rag_debug_cb.setChecked(bool(settings.get("rag_debug", False)))
        
        log_lvl = settings.get("log_level", logging.INFO)
        idx = self.log_level_combo.findData(int(log_lvl))
        if idx >= 0:
            self.log_level_combo.setCurrentIndex(idx)
            
        self.rag_max_hits_spin.setValue(int(settings.get("rag_debug_max_hits", 8)))
        self.rag_preview_spin.setValue(int(settings.get("rag_debug_preview_chars", 220)))

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_container.setVisible(checked)
        self.advanced_toggle.setText("Hide Advanced" if checked else "Show Advanced")

    def save_settings(self) -> None:
        """Save settings to the ai_settings module."""
        base_url = self.base_url_input.text().strip()
        if not base_url:
            base_url = "https://api.mistral.ai/v1"

        api_key = self.api_key_input.text().strip()

        settings = {
            "base_url": base_url,
            "model": self.chat_model_combo.currentText(),
            "api_key": api_key,
            "text_embed_model": self.text_embed_combo.currentData(),
            "code_embed_model": self.code_embed_combo.currentData(),
            "temperature": self.temperature.value(),
            "top_p": self.top_p.value(),
            "max_tokens": self.max_tokens.value(),
            "provider": self.provider_combo.currentData(),
            "hf_repo": self.hf_repo_input.text().strip(),
            "hf_file": self.hf_file_input.text().strip(),
            "persistent_ingest": self.persistent_ingest_cb.isChecked(),
            "graphrag_enable": self.graphrag_enable_cb.isChecked(),
            "graphrag_mode": self.graphrag_mode_combo.currentData(),
            "rag_debug": self.rag_debug_cb.isChecked(),
            "log_level": self.log_level_combo.currentData(),
            "rag_debug_max_hits": self.rag_max_hits_spin.value(),
            "rag_debug_preview_chars": self.rag_preview_spin.value(),
        }

        success = ai_settings.save_api_settings(settings)

        if success:
            self.status_label.setText("<span style='color: green;'>Settings saved successfully!</span>")
        else:
            self.status_label.setText("<span style='color: red;'>Failed to save settings.</span>")

    def reset_to_defaults(self) -> None:
        """Reset settings to defaults."""
        self.base_url_input.setText("https://api.mistral.ai/v1")
        self.chat_model_combo.setCurrentText("mistral-small-latest")
        self.api_key_input.clear()
        self.text_embed_combo.setCurrentIndex(0)
        self.code_embed_combo.setCurrentIndex(0)
        self.temperature.setValue(0.3)
        self.top_p.setValue(0.9)
        self.max_tokens.setValue(4096)
        
        self.provider_combo.setCurrentIndex(0)
        self.hf_repo_input.clear()
        self.hf_file_input.clear()
        self.persistent_ingest_cb.setChecked(True)
        self.graphrag_enable_cb.setChecked(True)
        self.graphrag_mode_combo.setCurrentIndex(0)
        self.rag_debug_cb.setChecked(False)
        self.log_level_combo.setCurrentIndex(1) # INFO
        self.rag_max_hits_spin.setValue(8)
        self.rag_preview_spin.setValue(220)
        
        self.status_label.setText("<span style='color: blue;'>Settings reset to defaults (not saved).</span>")

    def test_connection(self) -> None:
        """Test the Mistral API connection."""
        self.status_label.setText("<span style='color: blue;'>Testing connection...</span>")
        self.test_button.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        base_url = self.base_url_input.text().strip()
        if not base_url:
            base_url = "https://api.mistral.ai/v1"

        api_key = self.api_key_input.text().strip()
        if not api_key:
            api_key = ai_settings.get_api_key()

        if not api_key:
            self.status_label.setText("<span style='color: red;'>No API key provided. Please enter your Mistral API key.</span>")
            self.test_button.setEnabled(True)
            return

        try:
            self._test_mistral(base_url, api_key)
        except Exception as e:
            self.status_label.setText(f"<span style='color: red;'>Connection failed: {str(e)}</span>")
        finally:
            self.test_button.setEnabled(True)

    def _test_mistral(self, base_url: str, api_key: str) -> None:
        """Test Mistral API connection."""
        import requests

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Test chat completion
        response = requests.post(
            base_url.rstrip('/') + "/chat/completions",
            headers=headers,
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1
            },
            timeout=30,
        )

        if response.status_code == 200:
            self.status_label.setText("<span style='color: green;'>Connection successful!</span>")
        else:
            self.status_label.setText(f"<span style='color: red;'>API error: {response.status_code} - {response.text[:100]}</span>")

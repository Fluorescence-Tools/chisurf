from __future__ import annotations

import json
import logging
import pathlib
from typing import Optional, Dict, List

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.core.settings.path_utils import get_path
from chisurf.gui.widgets.general import EnterAwarePlainTextEdit


_LOG = logging.getLogger(__name__)


def _get_history_path() -> pathlib.Path:
    """Return path to agent history JSON file."""
    settings_dir = get_path('settings')
    return settings_dir / 'agent_history.json'


def _load_history() -> List[Dict[str, str]]:
    """Load chat history from file."""
    history_path = _get_history_path()
    if history_path.is_file():
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[-50:]
        except Exception:
            pass
    return []


def _save_history(history: List[Dict[str, str]]) -> None:
    """Save chat history to file."""
    history_path = _get_history_path()
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        _LOG.warning(f"Could not save agent history: {e}")


class AgentPanelWidget(QtWidgets.QWidget):
    """AI coding agent panel for the code editor."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        get_context_callback: Optional[callable] = None,
    ):
        super().__init__(parent)
        self._get_context_callback = get_context_callback
        self._chat_history: list = _load_history()
        self._system_prompt = "You are a helpful coding assistant for ChiSurf, a fluorescence spectroscopy analysis application. Help the user with their code questions."
        self.setup_ui()
        self._restore_history()

    def setup_ui(self) -> None:
        """Set up the user interface matching Chato's chat widget."""
        from chisurf.plugins._dev.chato.frontend.transcript import TranscriptRenderer
        chat_available = bool(TranscriptRenderer)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.transcript = QtWidgets.QTextBrowser(self)
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(200)
        layout.addWidget(self.transcript)

        if chat_available and TranscriptRenderer:
            self._render = TranscriptRenderer(self.transcript, logger=_LOG)
        else:
            # Fallback renderer or just a placeholder for the object
            class DummyRenderer:
                def __init__(self, browser, logger):
                    self.browser = browser
                def append_sys(self, text): self.browser.append(f"<i>{text}</i>")
                def append_user(self, text): self.browser.append(f"<b>User:</b> {text}")
                def append_assistant(self, text): self.browser.append(f"<b>AI:</b> {text}")
            self._render = DummyRenderer(self.transcript, _LOG)

        self.input = EnterAwarePlainTextEdit(self)
        self.input.setPlaceholderText("Ask something... (Enter to send, Shift+Enter for newline)")
        self.input.setMaximumHeight(80)
        self.input.textChanged.connect(self._on_input_changed)
        self.input.sendRequested.connect(self._on_send)
        layout.addWidget(self.input)

        button_layout = QtWidgets.QHBoxLayout()

        self.restart_btn = QtWidgets.QPushButton("Restart")
        self.restart_btn.clicked.connect(self.clear_history)
        button_layout.addWidget(self.restart_btn)

        button_layout.addStretch()

        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.setEnabled(False)
        self.send_btn.clicked.connect(self._on_send)
        button_layout.addWidget(self.send_btn)

        layout.addLayout(button_layout)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt; padding: 2px 8px;")
        layout.addWidget(self.status_label)

        self._render.append_sys(
            "I'm an AI coding assistant. Select code in the editor and ask me to:\n"
            "• Explain the selected code\n"
            "• Refactor or improve it\n"
            "• Find bugs or issues\n"
            "• Write tests\n"
            "• Add documentation\n\n"
            "Current file context will be included with your question."
        )

    def _on_input_changed(self) -> None:
        """Enable/disable send button based on input."""
        self.send_btn.setEnabled(bool(self.input.toPlainText().strip()))

    def _on_send(self) -> None:
        """Handle send button click or enter key."""
        text = self.input.toPlainText().strip()
        if not text:
            return

        self.input.clear()
        self._render.append_user(text)
        self._process_message(text)

    def _process_message(self, text: str) -> None:
        """Process the user's message and generate a response."""
        context = self._get_context() if self._get_context_callback else ""

        full_prompt = context + ("\n\n" if context else "") + f"User: {text}"

        self.status_label.setText("Thinking...")
        QtWidgets.QApplication.processEvents()

        try:
            response = self._call_agent(full_prompt)
            self._render.append_assistant(response)
            self._chat_history.append({"role": "user", "content": text})
            self._chat_history.append({"role": "assistant", "content": response})
            _save_history(self._chat_history)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self._render.append_sys(error_msg)
            _LOG.error(f"Agent error: {e}")
        finally:
            self.status_label.setText("")

    def _call_agent(self, user_text: str) -> str:
        """Call the Chato backend to get a response."""
        try:
            from chisurf.plugins._dev.chato.backend.langchain import chat_langchain
            if not chat_langchain:
                return "Chato not available: Development plugins are excluded from this build."

            from chisurf.core.settings import ai_settings

            api_key = ai_settings.get_api_key()
            provider = ai_settings.get_provider()
            base_url = ai_settings.get_base_url()
            model = ai_settings.get_model()

            if not base_url:
                return "No base URL configured. Please set your AI API settings in Tools > AI Settings."
            if not model:
                return "No model configured. Please set your AI API settings in Tools > AI Settings."
            if not api_key and provider != "local":
                return "No API key configured. Please set your AI API key in Tools > AI Settings."

            provider_map = {
                "openai": "openai",
                "anthropic": "anthropic",
                "mistral": "mistral_api",
                "local": "local_llm",
            }
            chato_provider = provider_map.get(provider, "openai")

            history: List[Dict[str, str]] = []
            for msg in self._chat_history[-10:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant"):
                    history.append({"role": role, "content": content})

            result = chat_langchain(
                base_url=base_url,
                chat_model=model,
                history=history,
                user_text=user_text,
                system_prompt=self._system_prompt,
                api_key=api_key,
                provider=chato_provider,
            )

            if isinstance(result, dict):
                return result.get("response", str(result))
            return str(result)
        except Exception as e:
            return f"Error calling agent: {e}"

    def _get_context(self) -> str:
        """Get the current editor context."""
        if self._get_context_callback:
            return self._get_context_callback()
        return ""

    def set_context_callback(self, callback: callable) -> None:
        """Set the callback to get editor context."""
        self._get_context_callback = callback


    def _restore_history(self) -> None:
        """Restore chat history from saved file."""
        for msg in self._chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                self._render.append_user(content)
            elif role == "assistant":
                self._render.append_assistant(content)

    def clear_history(self) -> None:
        """Clear the chat history."""
        self._chat_history = []
        _save_history([])
        self.transcript.clear()
        self._render.append_sys("History cleared. Current file context will be included with your next question.")

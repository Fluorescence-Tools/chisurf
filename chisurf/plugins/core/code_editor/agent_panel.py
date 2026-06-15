from __future__ import annotations

import json
import logging as std_logging
import pathlib
import re
import threading
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from chisurf.plugins.core.code_editor.context_retriever import retrieve_context
from chisurf.plugins.core.code_editor.validation import validate_writes
from chisurf.plugins.core.code_editor.wiki_indexer import build_api_index

try:
    from chisurf.gui.widgets.general import EnterAwarePlainTextEdit
    from chisurf.settings import ai_settings
    from chisurf.settings.path_utils import get_path
except ImportError:
    from chisurf.core.settings import ai_settings
    from chisurf.core.settings.path_utils import get_path
    from chisurf.gui.widgets.general import EnterAwarePlainTextEdit  # noqa: E402

from chisurf.plugins.core.code_editor.agent_runtime import (
    ALLOWED_TOOLS,
    AgentMode,
    AgentRunConfig,
    AgentRuntime,
    AgentToolRegistry,
    default_tool_registry,
)

_PROVIDER_NAMES: dict[str, str] = {
    "openai": "OpenAI (ChatGPT)",
    "mistral": "Mistral",
    "local": "Local (Ollama, LMStudio...)",
    "custom": "Custom",
}


def _get_history_path() -> pathlib.Path:
    """Return path to agent history JSON file."""
    settings_dir = get_path('settings')
    return settings_dir / 'agent_history.json'


def _load_history() -> list[dict[str, str]]:
    """Load chat history from file."""
    history_path = _get_history_path()
    if history_path.is_file():
        try:
            with open(history_path, encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[-50:]
        except Exception:
            pass
    return []


def _save_history(history: list[dict[str, str]]) -> None:
    """Save chat history to file."""
    history_path = _get_history_path()
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        std_logging.warning(f"Could not save agent history: {e}")


def _get_input_history_path() -> pathlib.Path:
    """Return path to agent input history JSON file."""
    return get_path('settings') / 'agent_input_history.json'


def _load_input_history() -> list[str]:
    """Load agent input history from file."""
    history_path = _get_input_history_path()
    if history_path.is_file():
        try:
            with open(history_path, encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[-100:]
        except Exception:
            pass
    return []


def _save_input_history(history: list[str]) -> None:
    """Save agent input history to file."""
    history_path = _get_input_history_path()
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history[-100:], f, indent=2)
    except Exception as e:
        std_logging.warning(f"Could not save agent input history: {e}")


class WikiDialog(QtWidgets.QDialog):
    """Dialog for viewing and querying the LLM Wiki."""

    def __init__(self, parent: QtWidgets.QWidget | None = None, populate_callback: callable | None = None, progress_callback: callable | None = None):
        super().__init__(parent)
        self._populate_callback = populate_callback
        self._progress_callback = progress_callback
        self.setWindowTitle("LLM Wiki")
        self.resize(700, 500)
        self.setup_ui()
        self.load_wiki_index()

    def setup_ui(self) -> None:
        """Set up the wiki dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)

        search_layout = QtWidgets.QHBoxLayout()
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search wiki...")
        self.search_input.textChanged.connect(self._filter_pages)
        search_layout.addWidget(self.search_input)

        self.refresh_btn = QtWidgets.QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.load_wiki_index)
        search_layout.addWidget(self.refresh_btn)

        self.populate_btn = QtWidgets.QPushButton("📡 Populate Wiki")
        self.populate_btn.setToolTip("Feed current codebase to LLM Wiki")
        if self._populate_callback:
            self.populate_btn.clicked.connect(self._on_populate_clicked)
        else:
            self.populate_btn.clicked.connect(self._populate_wiki_from_codebase)
        search_layout.addWidget(self.populate_btn)

        layout.addLayout(search_layout)

        self.page_list = QtWidgets.QListWidget()
        self.page_list.itemClicked.connect(self._on_page_selected)
        layout.addWidget(self.page_list)

        self.page_browser = QtWidgets.QTextBrowser()
        self.page_browser.setOpenExternalLinks(True)
        layout.addWidget(self.page_browser)

        self.progress_label = QtWidgets.QLabel("")
        self.progress_label.setStyleSheet("color: #666; font-size: 9pt; padding: 2px 8px;")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def load_wiki_index(self) -> None:
        """Load the wiki index and populate the page list."""
        self.page_list.clear()

        wiki_dir = pathlib.Path(__file__).resolve().parents[4] / "llm-wiki" / "wiki"
        if not wiki_dir.exists():
            self.page_browser.setPlainText("LLM Wiki not found.")
            return

        index_path = wiki_dir / "index.md"
        if index_path.exists():
            with open(index_path, encoding='utf-8') as f:
                content = f.read()
            self.page_browser.setPlainText(content)
            self._populate_page_list(wiki_dir)
        else:
            self.page_browser.setPlainText("Wiki index.md not found.")

    def _populate_page_list(self, wiki_dir: pathlib.Path) -> None:
        """Populate the page list with wiki pages."""
        self._pages = {}

        for section in ["entities", "concepts", "sources", "synthesis"]:
            section_dir = wiki_dir / section
            if not section_dir.exists():
                continue

            for page_path in section_dir.glob("*.md"):
                title = self._extract_title(page_path)
                self._pages[title] = page_path
                self.page_list.addItem(f"{section}: {title}")

    def _extract_title(self, page_path: pathlib.Path) -> str:
        """Extract title from a wiki page."""
        with open(page_path, encoding='utf-8') as f:
            content = f.read()

        lines = content.split("\n")
        if len(lines) > 1 and lines[0] == "---":
            for line in lines[1:]:
                if line == "---":
                    break
                if line.startswith("title:"):
                    return line.split(":", 1)[1].strip()

        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()

        return page_path.stem

    def _filter_pages(self, text: str) -> None:
        """Filter the page list based on search text."""
        text = text.lower()
        for i in range(self.page_list.count()):
            item = self.page_list.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    def _on_page_selected(self, item: QtWidgets.QListWidgetItem) -> None:
        """Handle page selection."""
        title = item.text().split(": ", 1)[1]
        page_path = self._pages.get(title)
        if page_path and page_path.exists():
            with open(page_path, encoding="utf-8") as f:
                self.page_browser.setPlainText(f.read())

    def _on_populate_clicked(self) -> None:
        """Handle populate wiki button click."""
        self.populate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setText("Populating LLM Wiki from codebase...")
        if self._progress_callback:
            self._progress_callback(self._set_progress)
        if self._populate_callback:
            self._populate_callback()
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Wiki population complete.")
        self.populate_btn.setEnabled(True)
        self.load_wiki_index()

    def _set_progress(self, message: str, value: int | None = None) -> None:
        """Update the progress display."""
        self.progress_label.setText(message)
        if value is not None:
            self.progress_bar.setValue(value)

    def _populate_wiki_from_codebase(self) -> None:
        """Fallback method for populating the wiki."""
        self.page_browser.setPlainText("Please use the agent panel's Populate Wiki button.")


class AgentPanelWidget(QtWidgets.QWidget):
    """AI coding agent panel for the code editor."""

    responseReceived = QtCore.Signal(str)
    errorReceived = QtCore.Signal(str)
    fixResponseReceived = QtCore.Signal(str, int)
    validationReceived = QtCore.Signal(int, object, str)

    runtimeEventReceived = QtCore.Signal(str, dict)
    runtimeStarted = QtCore.Signal()
    runtimeFinished = QtCore.Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        get_context_callback: callable | None = None,
    ):
        super().__init__(parent)
        self.editor = None
        self._get_context_callback = get_context_callback
        self._chat_history: list = _load_history()
        self._input_history: list[str] = _load_input_history()
        self._input_history_index = -1
        self._system_prompt = (
            "You are a helpful coding assistant for ChiSurf, a fluorescence "
            "spectroscopy analysis application. Help the user with their code "
            "questions. Be concise and direct. "
            "You have access to the LLM Wiki and verified ChiSurf API context. "
            "When writing ChiSurf scripts, prefer existing APIs from the verified context. "
            "Do not invent function, class, or method names that are not present in the retrieved context or current editor. "
            "If the requested API is missing, say that you need to inspect the codebase before suggesting code. "
            "Format your responses with markdown: use **bold** for emphasis, "
            "`code` for inline code, ```python for code blocks, and LaTeX "
            "\\[ ... \\] for equations. Use numbered lists for steps.\n\n"
            "If the user asks you to write, edit, create, or insert code into a document/file (e.g., 'write to current open document', 'write to foo.py'), "
            "you MUST output a complete replacement code block with a special header comment on the very first line:\n"
            "```python\n"
            "# WRITE_FILE: <filename_or_current>\n"
            "<your complete code here>\n"
            "```\n"
            "Use '# WRITE_FILE: current' to write to the active editor document, or specify the file name/path. "
            "The editor will automatically capture this block and update the corresponding document tab. "
            "After code is written, it will be checked with py_compile and ruff; if diagnostics remain, output another complete corrected WRITE_FILE block."
        )
        self.setup_ui()
        self._restore_history()

    def setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(4, 4, 4, 2)

        header_label = QtWidgets.QLabel("🤖 AI Assistant")
        header_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.setMinimumWidth(160)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        header_layout.addWidget(self.provider_combo)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setMinimumWidth(130)
        self.mode_combo.addItem("💬 Chat only", AgentMode.CHAT_ONLY.value)
        self.mode_combo.addItem("🔧 ChiSurf tools", AgentMode.CHISURF_TOOLS.value)
        self.mode_combo.addItem("🤖 Autonomous fit", AgentMode.AUTONOMOUS_FIT.value)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        header_layout.addWidget(self.mode_combo)

        self.rpc_status_label = QtWidgets.QLabel("⬤")
        self.rpc_status_label.setToolTip("ChiSurf RPC status: unknown")
        self.rpc_status_label.setStyleSheet("color: #888; font-size: 10pt; margin-right: 4px;")
        header_layout.addWidget(self.rpc_status_label)

        self.wiki_btn = QtWidgets.QPushButton("📚 Wiki")
        self.wiki_btn.setToolTip("Open LLM Wiki")
        self.wiki_btn.clicked.connect(self._open_wiki)
        header_layout.addWidget(self.wiki_btn)

        layout.addLayout(header_layout)

        self.transcript = QtWidgets.QTextBrowser(self)
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(200)
        layout.addWidget(self.transcript)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #333; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #7aa2f7, stop:1 #9ece6a); border-radius: 3px; }"
        )
        layout.addWidget(self.progress_bar)

        self.input = EnterAwarePlainTextEdit(self)
        self.input.setPlaceholderText("Ask something... (Enter to send, Shift+Enter for newline)")
        self.input.setMaximumHeight(80)
        self.input.textChanged.connect(self._on_input_changed)
        self.input.sendRequested.connect(self._on_send)
        self.input.historyPrevRequested.connect(self._history_previous)
        self.input.historyNextRequested.connect(self._history_next)
        layout.addWidget(self.input)

        button_layout = QtWidgets.QHBoxLayout()

        self.restart_btn = QtWidgets.QPushButton("🔄 Restart")
        self.restart_btn.setToolTip("Clear chat history")
        self.restart_btn.clicked.connect(self.clear_history)
        button_layout.addWidget(self.restart_btn)

        button_layout.addStretch()

        self.send_btn = QtWidgets.QPushButton("➡ Send")
        self.send_btn.setEnabled(False)
        self.send_btn.setToolTip("Send message (Enter)")
        self.send_btn.clicked.connect(self._on_send)
        button_layout.addWidget(self.send_btn)

        self.cancel_btn = QtWidgets.QPushButton("✕ Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("Cancel running agent operation")
        self.cancel_btn.setStyleSheet("color: #e06c75; font-weight: bold;")
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt; padding: 2px 8px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._runtime: Optional[AgentRuntime] = None
        self._runtime_thread: Optional[threading.Thread] = None
        self._rpc_available: bool = False

        self.responseReceived.connect(self._on_response_received)
        self.errorReceived.connect(self._on_error_received)
        self.fixResponseReceived.connect(self._on_fix_response_received)
        self.validationReceived.connect(self._on_validation_received)
        self.runtimeEventReceived.connect(self._on_runtime_event)
        self.runtimeStarted.connect(self._on_runtime_started)
        self.runtimeFinished.connect(self._on_runtime_finished)

        self._populate_providers()
        self._append_sys(
            "👋 I'm an AI coding assistant. Select code in the editor and ask me to:\n"
            "• 🔍 Explain the selected code\n"
            "• 🔧 Refactor or improve it\n"
            "• 🐛 Find bugs or issues\n"
            "• 🧪 Write tests\n"
            "• 📝 Add documentation\n\n"
            "📚 Use 'Wiki' to feed the current codebase to the LLM Wiki."
        )

    def _populate_providers(self) -> None:
        """Populate the provider combo with configured endpoints."""
        self.provider_combo.blockSignals(True)
        self.provider_combo.clear()

        active_provider = ai_settings.get_provider()
        active_idx = 0

        for i, (key, name) in enumerate(_PROVIDER_NAMES.items()):
            settings = ai_settings.get_api_settings(key)
            has_key = bool(settings.get("api_key"))
            model = settings.get("text_model", settings.get("model", ""))

            label = name
            if has_key:
                label += " ✓"
            if model:
                label += f"  [{model}]"

            self.provider_combo.addItem(label, key)
            if key == active_provider:
                active_idx = i

        self.provider_combo.setCurrentIndex(active_idx)
        self.provider_combo.blockSignals(False)

    def _on_provider_changed(self, index: int) -> None:
        """Handle provider selection change."""
        provider_key = self.provider_combo.currentData()
        if provider_key:
            settings = ai_settings.get_api_settings(provider_key)
            ai_settings.save_api_settings(settings, provider=provider_key)
            self._update_provider_label(provider_key)

    def _current_mode(self) -> AgentMode:
        """Return the currently selected agent mode."""
        val = self.mode_combo.currentData()
        try:
            return AgentMode(val)
        except ValueError:
            return AgentMode.CHAT_ONLY

    def _on_mode_changed(self, index: int) -> None:
        """Handle agent mode change."""
        mode = self._current_mode()
        self._update_mode_status(mode)
        if mode in (AgentMode.CHISURF_TOOLS, AgentMode.AUTONOMOUS_FIT):
            self._check_rpc_status()
        else:
            self.rpc_status_label.setStyleSheet("color: #888; font-size: 10pt; margin-right: 4px;")
            self.rpc_status_label.setToolTip("RPC not needed in chat-only mode")

    def _update_mode_status(self, mode: AgentMode) -> None:
        """Update the status label with current mode info."""
        labels = {
            AgentMode.CHAT_ONLY: "💬 Chat only",
            AgentMode.CHISURF_TOOLS: "🔧 Tool mode active",
            AgentMode.AUTONOMOUS_FIT: "🤖 Autonomous fit mode",
        }
        self.status_label.setText(labels.get(mode, ""))

    def _check_rpc_status(self) -> None:
        """Check ChiSurf RPC availability and update the status indicator."""
        try:
            from chisurf.plugins.core.code_editor.settings import get_editor_settings
            from chisurf.server.startup import rpc_is_available
            ed = get_editor_settings()
            host = str(ed.get("agent_chisurf_rpc_host", "127.0.0.1"))
            port = int(ed.get("agent_chisurf_rpc_cmd_port", 8765))
            self._rpc_available = rpc_is_available(host, port, timeout_ms=300)
            if self._rpc_available:
                self.rpc_status_label.setStyleSheet("color: #98c379; font-size: 10pt; margin-right: 4px;")
                self.rpc_status_label.setToolTip(f"ChiSurf RPC connected: {host}:{port}")
            else:
                self.rpc_status_label.setStyleSheet("color: #e06c75; font-size: 10pt; margin-right: 4px;")
                self.rpc_status_label.setToolTip(f"ChiSurf RPC unreachable: {host}:{port}")
        except Exception:
            self.rpc_status_label.setStyleSheet("color: #d19a66; font-size: 10pt; margin-right: 4px;")
            self.rpc_status_label.setToolTip("RPC status check failed")

    def _on_cancel(self) -> None:
        """Cancel the running agent operation."""
        if self._runtime is not None:
            self._runtime.cancel()
            self._append_sys("⏹️ Cancellation requested...")
        self.cancel_btn.setEnabled(False)

    def _open_wiki(self) -> None:
        """Open the LLM Wiki dialog."""
        dialog = WikiDialog(
            self,
            populate_callback=self._populate_wiki_from_codebase,
            progress_callback=self._set_wiki_progress,
        )
        dialog.exec()

    def _set_wiki_progress(self, callback: callable) -> None:
        """Set the wiki progress callback."""
        self._wiki_progress_callback = callback

    def _populate_wiki_from_codebase(self) -> None:
        """Feed the current codebase into the LLM Wiki."""
        self.status_label.setText("Populating LLM Wiki from codebase...")
        if hasattr(self, "_wiki_progress_callback"):
            self._wiki_progress_callback("Populating LLM Wiki from codebase...")
        QtWidgets.QApplication.processEvents()

        try:
            count = self._write_codebase_to_wiki()
            build_api_index(pathlib.Path(__file__).resolve().parents[4])
            self.status_label.setText(f"LLM Wiki populated with {count} codebase files.")
            if hasattr(self, "_wiki_progress_callback"):
                self._wiki_progress_callback(f"LLM Wiki populated with {count} codebase files.")
            self._append_sys(f"LLM Wiki populated with {count} codebase files.")
        except Exception as e:
            self.status_label.setText(f"Failed to populate wiki: {e}")
            if hasattr(self, "_wiki_progress_callback"):
                self._wiki_progress_callback(f"Failed to populate wiki: {e}")
            std_logging.error(f"Failed to populate wiki: {e}")

    def _write_codebase_to_wiki(self) -> int:
        """Write codebase files to the LLM Wiki in meaningful chunks."""
        repo_root = pathlib.Path(__file__).resolve().parents[4]
        wiki_dir = repo_root / "llm-wiki" / "wiki"
        sources_dir = wiki_dir / "sources"
        concepts_dir = wiki_dir / "concepts"
        synthesis_dir = wiki_dir / "synthesis"

        sources_dir.mkdir(parents=True, exist_ok=True)
        concepts_dir.mkdir(parents=True, exist_ok=True)
        synthesis_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        pages = []
        py_files = [
            py_file
            for py_file in (repo_root / "chisurf").rglob("*.py")
            if "__pycache__" not in py_file.parts and "test" not in py_file.parts
        ]

        # Group files by package/module for meaningful chunks
        grouped_files: dict[str, list[pathlib.Path]] = {}
        for py_file in py_files:
            rel_parts = py_file.relative_to(repo_root / "chisurf").parts
            if len(rel_parts) >= 2:
                group_key = "/".join(rel_parts[:2])
            else:
                group_key = rel_parts[0] if rel_parts else "root"
            grouped_files.setdefault(group_key, []).append(py_file)

        for idx, (group_key, files) in enumerate(grouped_files.items(), start=1):
            title = group_key.replace("/", "_")
            title = title[:80]
            content_parts = []
            file_paths = []

            for py_file in files:
                rel_path = py_file.relative_to(repo_root)
                file_paths.append(rel_path.as_posix())
                try:
                    content = py_file.read_text(encoding="utf-8")
                except Exception:
                    continue
                content_parts.append(f"\n\n## File: `{rel_path.as_posix()}`\n\n```python\n{content[:4000]}\n```")

            combined_content = "\n".join(content_parts)
            source_path = sources_dir / f"{title}.md"
            source_path.write_text(
                self._create_source_page(title, ", ".join(file_paths), combined_content),
                encoding="utf-8",
            )

            # Extract larger concepts (classes with methods)
            for item in self._extract_larger_concepts(files):
                item["path"] = ", ".join(file_paths[:5])
                concept_path = concepts_dir / f"{title}__{item['name']}.md"
                concept_path.write_text(
                    self._create_concept_page(title, item),
                    encoding="utf-8",
                )
                pages.append(f"[[{title}__{item['name']}]]")

            pages.append(f"[[{title}]]")
            count += 1

            if hasattr(self, "_wiki_progress_callback"):
                self._wiki_progress_callback(f"Processed {idx}/{len(grouped_files)}: {group_key}")
            QtWidgets.QApplication.processEvents()

        self._update_wiki_index(wiki_dir, pages)
        self._append_wiki_log(wiki_dir, count)

        return count

    def _create_source_page(self, title: str, path: str, content: str) -> str:
        """Create a wiki source page for a code file."""
        safe_content = content[:5000]
        return f"""---
title: {title}
type: source
sources: [{path}]
related: []
created: 2026-06-09
updated: 2026-06-09
---

# {title}

Source file: `{path}`

## Code Content

```python
{safe_content}
```
"""

    def _create_concept_page(self, title: str, item: dict[str, str]) -> str:
        """Create a wiki concept page for a class or function."""
        return f"""---
title: {title}__{item['name']}
type: concept
sources: [{item['path']}]
related: [[{title}]]
created: 2026-06-09
updated: 2026-06-09
---

# {item['name']}

Type: {item['type']}

Source: `{item['path']}`

## Definition

```python
{item['definition']}
```
"""

    def _extract_larger_concepts(self, files: list[pathlib.Path]) -> list[dict[str, str]]:
        """Extract larger concepts (classes with methods) from code files."""
        concepts = []
        for py_file in files:
            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            lines = content.split("\n")
            current_class = None
            class_start = 0
            class_lines = []

            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("class "):
                    if current_class and class_lines:
                        concepts.append(
                            {
                                "name": current_class,
                                "type": "class",
                                "definition": "\n".join(class_lines)[:3000],
                            }
                        )
                    current_class = stripped.split("(", 1)[0].split(" ", 1)[1].strip()
                    class_start = i
                    class_lines = lines[i : min(i + 80, len(lines))]
                elif current_class and stripped.startswith("def ") and i > class_start + 5:
                    continue
                elif current_class and i > class_start + 80:
                    concepts.append(
                        {
                            "name": current_class,
                            "type": "class",
                            "definition": "\n".join(class_lines)[:3000],
                        }
                    )
                    current_class = None
                    class_lines = []

            if current_class and class_lines:
                concepts.append(
                    {
                        "name": current_class,
                        "type": "class",
                        "definition": "\n".join(class_lines)[:3000],
                    }
                )

        return concepts[:20]

    def _extract_code_items(self, content: str) -> list[dict[str, str]]:
        """Extract classes and functions from code content."""
        items = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("class ") or stripped.startswith("def "):
                item_type = "class" if stripped.startswith("class ") else "function"
                name = stripped.split("(", 1)[0].split(" ", 1)[1].strip()
                definition = "\n".join(lines[i : min(i + 20, len(lines))])
                items.append(
                    {
                        "name": name,
                        "type": item_type,
                        "definition": definition[:1000],
                    }
                )

        return items[:10]

    def _update_wiki_index(self, wiki_dir: pathlib.Path, pages: list[str]) -> None:
        """Update the wiki index with new pages."""
        index_path = wiki_dir / "index.md"
        unique_pages = sorted(set(pages))

        content = """---
title: Wiki Index
type: index
updated: 2026-06-09
---

# Wiki Index

> Catalog of all wiki pages for the ChiSurf codebase. Updated by the agent on every ingest.

## Codebase Sources

"""
        for page in unique_pages[:100]:
            content += f"- [[{page}]]\n"

        index_path.write_text(content, encoding="utf-8")

    def _append_wiki_log(self, wiki_dir: pathlib.Path, count: int) -> None:
        """Append an entry to the wiki log."""
        log_path = wiki_dir / "log.md"
        entry = f"\n## [2026-06-09] populate | Fed {count} codebase files into LLM Wiki\n"

        if log_path.exists():
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry)
        else:
            log_path.write_text(
                "---\ntitle: Activity Log\ntype: log\n---\n\n# Activity Log\n" + entry,
                encoding="utf-8",
            )

    def _append_sys(self, text: str) -> None:
        """Append a system message to the transcript."""
        self.transcript.append(f'<div style="color: #aaa; font-style: italic; margin: 4px 0 12px 0;">{text}</div>')

    def _append_user(self, text: str) -> None:
        """Append a user message to the transcript."""
        self.transcript.append('<div style="margin: 4px 0 2px 0;"></div>')
        self.transcript.append('<span style="color: #7aa2f7; font-weight: bold;">User:</span>')
        self.transcript.append(f'<div style="margin-left: 12px; margin-bottom: 4px;">{text}</div>')

    def _append_assistant(self, text: str) -> None:
        """Append an assistant message to the transcript."""
        self.transcript.append('<span style="color: #9ece6a; font-weight: bold;">AI:</span>')
        formatted = self._format_message(text)
        self.transcript.append(f'<div style="margin-left: 12px; margin-bottom: 4px;">{formatted}</div>')

    def _format_message(self, text: str) -> str:
        """Format assistant message with markdown, code blocks, and equations."""
        import re

        import markdown

        try:
            from latex2mathml.converter import convert as latex_to_mathml
        except ImportError:
            latex_to_mathml = None

        def format_code_block(match: re.Match[str]) -> str:
            language = match.group(1) or ""
            code = match.group(2)
            escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return (
                f'<div style="background: #1e1e1e; border: 1px solid #444; '
                f'border-radius: 4px; padding: 8px; margin: 6px 0; '
                f'font-family: monospace; font-size: 9pt; white-space: pre-wrap;">'
                f'<div style="color: #777; margin-bottom: 4px;">{language}</div>'
                f'{escaped}'
                f'</div>'
            )

        def format_display_equation(match: re.Match[str]) -> str:
            equation = match.group(1).strip()
            if latex_to_mathml:
                try:
                    mathml = latex_to_mathml(equation)
                    return (
                        f'<div style="background: #252526; border: 1px solid #555; '
                        f'border-radius: 4px; padding: 8px; margin: 6px 0; '
                        f'text-align: center; color: #d4d4d4;">{mathml}</div>'
                    )
                except Exception:
                    pass
            return f'<div style="background: #252526; border: 1px solid #555; border-radius: 4px; padding: 8px; margin: 6px 0; text-align: center; font-family: monospace;">{equation}</div>'

        def format_inline_equation(match: re.Match[str]) -> str:
            equation = match.group(1).strip()
            if latex_to_mathml:
                try:
                    return latex_to_mathml(equation)
                except Exception:
                    pass
            return equation

        def format_standalone_equation(line: str) -> str:
            """Format standalone LaTeX equations."""
            stripped = line.strip()
            if not stripped:
                return line
            if "\\" in stripped and any(cmd in stripped for cmd in ["\\frac", "\\left", "\\right", "\\sqrt", "\\sum", "\\int"]):
                equation = stripped
                # Handle equations like "E = \frac{...}"
                if "=" in equation:
                    equation = equation.split("=", 1)[1].strip()
                if latex_to_mathml:
                    try:
                        latex_to_mathml(equation)
                        # Extract text representation from MathML for Qt rendering
                        text_eq = equation.replace("\\frac", "frac").replace("\\left", "").replace("\\right", "")
                        text_eq = text_eq.replace("\\_", "_").replace("\\,", " ")
                        return (
                            f'<div style="background: #252526; border: 1px solid #555; '
                            f'border-radius: 4px; padding: 6px 8px; margin: 4px 0; '
                            f'text-align: center; color: #d4d4d4; font-family: "Times New Roman", serif; '
                            f'font-size: 11pt;">{text_eq}</div>'
                        )
                    except Exception:
                        pass
                return f'<div style="background: #252526; border: 1px solid #555; border-radius: 4px; padding: 6px 8px; margin: 4px 0; text-align: center; font-family: monospace;">{stripped}</div>'
            return line

        formatted = re.sub(r"```(\w*)\n(.*?)```", format_code_block, text, flags=re.DOTALL)
        formatted = re.sub(r"\\\[(.*?)\\\]", format_display_equation, formatted, flags=re.DOTALL)
        formatted = re.sub(r"\\\((.*?)\\\)", format_inline_equation, formatted)
        formatted = "\n".join(format_standalone_equation(line) for line in formatted.split("\n"))
        formatted = markdown.markdown(
            formatted,
            extensions=["extra", "sane_lists", "smarty"],
            output_format="html5",
        )
        return formatted

    def _on_input_changed(self) -> None:
        """Enable/disable send button based on input."""
        self.send_btn.setEnabled(bool(self.input.toPlainText().strip()))

    def _history_previous(self) -> None:
        """Navigate to previous input in history."""
        if not self._input_history:
            return
        if self._input_history_index < 0:
            self._input_history_index = len(self._input_history) - 1
        else:
            self._input_history_index -= 1
        self.input.setPlainText(self._input_history[self._input_history_index])
        self.input.moveCursor(QtGui.QTextCursor.End)

    def _history_next(self) -> None:
        """Navigate to next input in history."""
        if not self._input_history:
            return
        if self._input_history_index < len(self._input_history) - 1:
            self._input_history_index += 1
            self.input.setPlainText(self._input_history[self._input_history_index])
        else:
            self._input_history_index = -1
            self.input.clear()
        self.input.moveCursor(QtGui.QTextCursor.End)

    def _on_send(self) -> None:
        """Handle send button click or enter key."""
        text = self.input.toPlainText().strip()
        if not text:
            return

        self.input.clear()
        if text not in self._input_history:
            self._input_history.append(text)
            _save_input_history(self._input_history)
        self._input_history_index = -1
        self._append_user(text)

        mode = self._current_mode()
        if mode in (AgentMode.CHISURF_TOOLS, AgentMode.AUTONOMOUS_FIT):
            self._start_runtime(text, mode)
        else:
            self._process_message(text)

    def _process_message(self, text: str) -> None:
        """Process the user's message asynchronously without blocking the UI."""
        context = self._get_context() if self._get_context_callback else ""
        self.status_label.setText("🔎 Retrieving ChiSurf API context...")
        self.progress_bar.setVisible(True)
        self.send_btn.setEnabled(False)
        self.input.setEnabled(False)
        self._chat_history.append({"role": "user", "content": text})

        def _run():
            try:
                wiki_context = self._get_wiki_context(text, current_context=context)
                full_prompt = context + ("\n\n" if context else "")
                if wiki_context:
                    full_prompt += f"LLM Wiki Context:\n{wiki_context}\n\n"
                full_prompt += f"User: {text}"
                request = self._build_agent_request(full_prompt)
                result = self._call_agent_request(request)
                self.responseReceived.emit(result)
            except Exception as e:
                self.errorReceived.emit(f"{e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _on_response_received(self, response: str) -> None:
        """Handle a successful agent response on the main thread."""
        self.progress_bar.setVisible(True)
        self.status_label.setText("")

        self._append_assistant(response)
        self._chat_history.append({"role": "assistant", "content": response})

        writes = self._apply_file_writes(response)
        if writes:
            self._append_sys("📝 Applying generated code to the editor...")
            self._start_validation(writes, iteration=0, label="Initial code")
        else:
            self._append_sys("💬 No code was written; no compile or ruff checks run.")
            self.send_btn.setEnabled(True)
            self.input.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("")

    def _on_error_received(self, error_msg: str) -> None:
        """Handle an agent error on the main thread."""
        self.progress_bar.setVisible(False)
        self.send_btn.setEnabled(True)
        self.input.setEnabled(True)
        self.status_label.setText("⚠️ Error")

        self._append_sys(f"Error: {error_msg}")
        std_logging.error(f"Agent error: {error_msg}")

    def _start_runtime(self, text: str, mode: AgentMode) -> None:
        """Start the agent runtime in a background thread."""
        from chisurf.plugins.core.code_editor.settings import get_editor_settings
        ed_settings = get_editor_settings()
        config = AgentRunConfig(
            mode=mode,
            max_tool_iterations=int(ed_settings.get("agent_max_tool_iterations", 25)),
            tool_timeout_ms=int(ed_settings.get("agent_tool_timeout_ms", 30000)),
            code_run_enabled=bool(ed_settings.get("agent_code_run_enabled", False)),
            chisurf_rpc_host=str(ed_settings.get("agent_chisurf_rpc_host", "127.0.0.1")),
            chisurf_rpc_cmd_port=int(ed_settings.get("agent_chisurf_rpc_cmd_port", 8765)),
            chisurf_rpc_pub_port=int(ed_settings.get("agent_chisurf_rpc_pub_port", 8766)),
            editor_rpc_host=str(ed_settings.get("agent_editor_rpc_host", "127.0.0.1")),
            editor_rpc_cmd_port=int(ed_settings.get("agent_editor_rpc_cmd_port", 8775)),
            editor_rpc_pub_port=int(ed_settings.get("agent_editor_rpc_pub_port", 8776)),
            code_timeout_ms=int(ed_settings.get("agent_code_timeout_ms", 5000)),
            output_max_chars=int(ed_settings.get("agent_output_max_chars", 20000)),
        )

        try:
            from chisurf.server.startup import (
                ensure_embedded_chisurf_rpc_server,
                session_state_from_live_chisurf,
            )
            state = session_state_from_live_chisurf()
            available = ensure_embedded_chisurf_rpc_server(
                host=config.chisurf_rpc_host,
                cmd_port=config.chisurf_rpc_cmd_port,
                pub_port=config.chisurf_rpc_pub_port,
                timeout_s=3.0,
                state=state,
            )
            if not available:
                self._append_sys("⚠️ ChiSurf RPC server not available. Tool execution disabled.")
                self.send_btn.setEnabled(True)
                self.input.setEnabled(True)
                return
            registry = default_tool_registry(config)
        except Exception as e:
            self._append_sys(f"⚠️ Could not connect to ChiSurf RPC: {e}")
            self.send_btn.setEnabled(True)
            self.input.setEnabled(True)
            return

        def _llm_call(messages: List[Dict[str, Any]]) -> str:
            """Call the LLM using existing agent infrastructure."""
            provider_key = self.provider_combo.currentData()
            if not provider_key:
                return json.dumps({"type": "message", "content": "No provider selected."})
            settings = ai_settings.get_api_settings(provider_key)
            api_key = settings.get("api_key", "")
            base_url = settings.get("base_url", "").strip().rstrip("/")
            model = settings.get("text_model", settings.get("model", ""))
            if not api_key or not base_url or not model:
                return json.dumps({"type": "message", "content": "AI provider not fully configured."})

            import requests
            payload = {
                "model": model,
                "messages": messages,
                "temperature": settings.get("temperature", 0.3),
                "max_tokens": settings.get("max_tokens", 4096),
            }
            try:
                resp = requests.post(
                    base_url + "/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
                if resp.status_code != 200:
                    return json.dumps({"type": "message", "content": f"API error {resp.status_code}"})
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return json.dumps({"type": "message", "content": "No response from API."})
            except Exception as e:
                return json.dumps({"type": "message", "content": f"LLM call failed: {e}"})

        def system_prompt_fn() -> str:
            """Build the system prompt for tool-using modes."""
            base = self._system_prompt
            allowed = "\n".join(sorted(ALLOWED_TOOLS))
            safety = (
                "\n\n## Safety Rules\n"
                "- Do not invent RPC method names. Only use tools from the allowed list.\n"
                "- For fitting, always take a parameter snapshot before modifying parameters.\n"
                "- If the target fit is ambiguous or missing, ask the user for clarification.\n"
                "- Do not skip fit.parameter_snapshot before calling parameter.set_value.\n"
            )
            if mode == AgentMode.AUTONOMOUS_FIT:
                base += (
                    "\n\nYou are in AUTONOMOUS FIT mode. Follow this protocol:\n"
                    "1. Call session.describe to inspect the session.\n"
                    "2. Call fit.get on the target fit.\n"
                    "3. Call fit.parameter_snapshot before any changes.\n"
                    "4. Propose parameter changes (use parameter.set_value etc.).\n"
                    "5. Call fit.run.\n"
                    "6. Call fit.diagnostics.\n"
                    "7. Iterate to improve fit quality.\n"
                    "8. When done, emit final with a detailed summary.\n"
                    + safety +
                    "Available tools:\n" + allowed
                )
            else:
                base += (
                    "\n\nYou are in ChiSurf TOOLS mode. You may call individual "
                    "tools when the user asks. Do not run multi-step autonomous "
                    "loops unless the user explicitly requests it.\n"
                    + safety +
                    "Available tools:\n" + allowed
                )
            return base

        self._runtime = AgentRuntime(
            config=config,
            tool_registry=registry,
            event_callback=lambda ev, data: self.runtimeEventReceived.emit(ev, data),
            llm_call_fn=_llm_call,
            system_prompt_fn=system_prompt_fn,
        )

        self.runtimeStarted.emit()

        def _run():
            try:
                self._runtime.start(text)
            except Exception as e:
                std_logging.error(f"Runtime error: {e}")
            finally:
                self.runtimeFinished.emit()

        self._runtime_thread = threading.Thread(target=_run, daemon=True)
        self._runtime_thread.start()

    def _on_runtime_started(self) -> None:
        """Handle runtime start on the main thread."""
        self.send_btn.setEnabled(False)
        self.input.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.mode_combo.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("🤖 Agent running...")
        self._check_rpc_status()

    def _on_runtime_finished(self) -> None:
        """Handle runtime completion on the main thread."""
        self.send_btn.setEnabled(True)
        self.input.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.mode_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("")
        self._runtime = None
        self._runtime_thread = None
        _save_history(self._chat_history)

    def _on_runtime_event(self, event: str, data: Dict[str, Any]) -> None:
        """Handle a runtime event on the main thread."""
        if event == "message.started":
            pass
        elif event == "message.completed":
            content = data.get("content", "")
            if content:
                self._append_assistant(content)
                self._chat_history.append({"role": "assistant", "content": content})
        elif event == "tool.started":
            tool = data.get("tool", "?")
            params = data.get("params", {})
            compact = json.dumps(params, default=str)[:200]
            self._append_sys(f"🔧 Calling <b>{tool}</b> params: {compact}")
            self.status_label.setText(f"🔧 Executing {tool}...")
        elif event == "tool.completed":
            tool = data.get("tool", "?")
            ok = data.get("ok", False)
            elapsed = data.get("elapsed_ms", 0)
            status = "✅" if ok else "❌"
            self._append_sys(f"{status} <b>{tool}</b> ({elapsed}ms)")
        elif event == "tool.failed":
            error = data.get("error", "unknown error")
            self._append_sys(f"❌ Tool failed: {error}")
        elif event == "fit.iteration.started":
            self._append_sys("🏃 Fit run starting...")
            self.status_label.setText("🏃 Running fit...")
        elif event == "fit.iteration.completed":
            ok = data.get("ok", False)
            reason = data.get("reason", "")
            metrics = data.get("metrics", {})
            worsening = data.get("worsening_count", 0)
            if metrics:
                chi2r = metrics.get("chi2r", "?")
                chi2 = metrics.get("chi2", "?")
                self._append_sys(
                    f"📊 chi2r={chi2r} chi2={chi2} "
                    f"{'✅ improved' if ok else '❌ worsened'} "
                    f"(worsening streak: {worsening})"
                )
            else:
                self._append_sys(f"📊 Fit iteration: {'✅' if ok else '❌'} {reason}")
        elif event == "fit.rollback.completed":
            self._append_sys("⏪ Restored best-known parameter snapshot")
        elif event == "agent.cancelled":
            self._append_sys("⏹️ Agent cancelled by user")
        elif event == "agent.failed":
            error = data.get("error", "unknown error")
            self._append_sys(f"⚠️ Agent failed: {error}")
        elif event == "agent.completed":
            summary = data
            fit_runs = summary.get("fit_runs", 0)
            iterations = summary.get("iterations", 0)
            rollback = summary.get("rollback_done", False)
            initial = summary.get("initial_metrics", {})
            best = summary.get("best_metrics", {})
            lines = [
                "---",
                "**Agent run complete**",
                f"- Iterations: {iterations}",
                f"- Fit runs: {fit_runs}",
                f"- Rollback: {'yes' if rollback else 'no'}",
            ]
            if initial:
                lines.append(f"- Initial chi2r: {initial.get('chi2r', '?')}  chi2: {initial.get('chi2', '?')}")
            if best:
                lines.append(f"- Best chi2r: {best.get('chi2r', '?')}  chi2: {best.get('chi2', '?')}")
            lines.append("---")
            self._append_sys("<br>".join(lines))

    def _start_fix_loop(self, issues: list[tuple[str, list[dict]]], iteration: int = 0) -> None:
        """Send validation issues back to the agent for auto-fixing."""
        max_iterations = 5
        if iteration >= max_iterations:
            self.send_btn.setEnabled(True)
            self.input.setEnabled(True)
            self._append_sys(f"⚠️ Reached max {max_iterations} fix iterations, some issues remain.")
            _save_history(self._chat_history)
            self.progress_bar.setVisible(False)
            self.status_label.setText("")
            return

        total_issues = sum(len(d) for _, d in issues)
        kind = self._issue_kind(issues)
        fix_lines = [
            "Fix the following validation issues in the written code. "
            "Output only a complete corrected code block with the WRITE_FILE header.",
        ]
        for fname, diags in issues:
            fix_lines.append(f"\nFile: `{fname}`")
            for diagnostic in diags[:15]:
                line = diagnostic.get("line", "?")
                code = diagnostic.get("code", "?")
                msg = diagnostic.get("message", "?")
                fix_lines.append(f"  L{line} {code}: {msg}")
        fix_prompt = "\n".join(fix_lines)

        self._append_sys(
            f"🔄 Auto-fix round {iteration + 1}: {total_issues} {kind} issue(s) found — asking agent to fix..."
        )
        self.status_label.setText(f"🔄 Fixing ({kind})... round {iteration + 1}/{max_iterations}")
        self.progress_bar.setVisible(True)

        def _run():
            try:
                request = self._build_agent_request(fix_prompt)
                result = self._call_agent_request(request)
                self.fixResponseReceived.emit(result, iteration)
            except Exception as e:
                self.errorReceived.emit(f"{e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _on_fix_response_received(self, response: str, iteration: int) -> None:
        """Apply a fix response and start validation off the UI thread."""
        writes = self._apply_file_writes(response)
        if writes:
            self._append_sys(f"📝 Applying auto-fix round {iteration + 1} to the editor...")
            self._start_validation(writes, iteration + 1, f"Fix round {iteration + 1}")
        else:
            self._append_sys("⚠️ Fix response did not contain WRITE_FILE code; leaving current diagnostics visible.")
            self.send_btn.setEnabled(True)
            self.input.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("")

    def _start_validation(self, writes: list[tuple[str, str]], iteration: int, label: str) -> None:
        """Validate generated writes in a worker thread."""
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"🐍 Validating {label}...")
        self._append_sys(f"🐍 Running py_compile and ruff for {label}...")

        def _run():
            try:
                issues = validate_writes(writes, editor=self.editor)
                self.validationReceived.emit(iteration, issues, label)
            except Exception as e:
                self.errorReceived.emit(f"{e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _on_validation_received(self, iteration: int, issues: list[tuple[str, list[dict]]], label: str) -> None:
        """Handle validation results on the main thread."""
        if issues:
            total = sum(len(d) for _, d in issues)
            kind = self._issue_kind(issues)
            self._append_sys(f"🔍 Validation found {total} {kind} issue(s) in {label}.")
            QtCore.QTimer.singleShot(0, lambda: self._start_fix_loop(issues, iteration))
            return

        self._append_sys("✅ All checks pass — code is clean.")
        self.send_btn.setEnabled(True)
        self.input.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("")
        _save_history(self._chat_history)

    @staticmethod
    def _issue_kind(issues: list[tuple[str, list[dict]]]) -> str:
        """Return the dominant validation issue kind."""
        if any(diagnostic.get("code") == "E999" for _, diagnostics in issues for diagnostic in diagnostics):
            return "compile"
        return "ruff"

    def _apply_file_writes(self, response: str) -> list[tuple[str, str]]:
        """Parse and apply file writes from the assistant.

        Returns
        -------
        list[tuple[str, str]]
            Filename and content pairs found in the response.
        """
        writes = []

        # 1. Match XML-like tags: <write_file filename="xyz">content</write_file>
        xml_pattern = re.compile(r"<write_file\s+filename=\"([^\"]+)\"\s*>(.*?)</write_file>", re.DOTALL)
        for filename, content in xml_pattern.findall(response):
            writes.append((filename.strip(), content.strip()))

        # 2. Match markdown code blocks containing a # WRITE_FILE header
        block_pattern = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)\n```", re.DOTALL)
        for block in block_pattern.findall(response):
            match = re.match(r"^\s*(?:#|//)\s*WRITE_FILE:?\s*([^\r\n]+)\r?\n(.*)$", block, re.DOTALL)
            if match:
                writes.append((match.group(1).strip(), match.group(2)))

        for filename, content in writes:
            self._write_to_editor_document(filename, content)

        return writes

    def _write_to_editor_document(self, filename: str, content: str) -> None:
        """Write content to the matching open editor document or create a new one."""
        if self.editor is None:
            std_logging.warning("No editor reference in AgentPanelWidget")
            return

        filename = filename.strip()
        target_editor = None
        is_current = filename.lower() in (
            "current",
            "active",
            "current open document",
            "current_document",
            "current file",
            "untitled",
            "untitled document",
        )

        if is_current:
            target_editor = self.editor._get_current_editor()
            if target_editor is None:
                self.editor._add_new_editor_tab()
                target_editor = self.editor._get_current_editor()
        else:
            open_files = self.editor._open_files
            # Exact path match
            for path, widget in open_files.items():
                if path == filename:
                    target_editor = widget
                    break
                try:
                    same_path = pathlib.Path(path).resolve() == pathlib.Path(filename).resolve()
                except (OSError, RuntimeError, ValueError):
                    same_path = False
                if same_path:
                    target_editor = widget
                    break

            # Base name match
            if target_editor is None:
                for path, widget in open_files.items():
                    if pathlib.Path(path).name == pathlib.Path(filename).name:
                        target_editor = widget
                        break

            # Tab title match
            if target_editor is None:
                for index in range(self.editor.tab_widget.count()):
                    widget = self.editor.tab_widget.widget(index)
                    if widget is not self.editor.agent_panel and widget is not None:
                        tab_text = self.editor.tab_widget.tabText(index)
                        clean = tab_text[:-2] if tab_text.endswith(" *") else tab_text
                        if clean == filename or pathlib.Path(clean).name == pathlib.Path(filename).name:
                            target_editor = widget
                            break

        if target_editor is not None:
            target_editor.blockSignals(True)
            target_editor.setPlainText(content)
            target_editor.blockSignals(False)
            target_editor.document().setModified(True)
            idx = self.editor.tab_widget.indexOf(target_editor)
            if idx != -1:
                self.editor.tab_widget.setCurrentIndex(idx)
            self.editor._sync_editor_document(target_editor)
        else:
            if self.editor._is_real_file(filename) or "/" in filename or "\\" in filename:
                try:
                    path = pathlib.Path(filename).resolve()
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    self.editor.load_file(str(path))
                except Exception as e:
                    std_logging.error(f"Failed to write file to disk: {e}")
                    editor, idx = self.editor._create_editor_tab(filename=filename)
                    editor.blockSignals(True)
                    editor.setPlainText(content)
                    editor.blockSignals(False)
                    editor.document().setModified(True)
                    self.editor.tab_widget.setCurrentIndex(idx)
            else:
                editor, idx = self.editor._create_editor_tab(filename=filename)
                editor.blockSignals(True)
                editor.setPlainText(content)
                editor.blockSignals(False)
                editor.document().setModified(True)
                self.editor.tab_widget.setCurrentIndex(idx)

    def _build_agent_request(self, user_text: str) -> dict:
        """Build an LLM request from the selected provider and chat history."""
        provider_key = self.provider_combo.currentData()
        if not provider_key:
            return {"error": "No provider selected."}

        settings = ai_settings.get_api_settings(provider_key)
        api_key = settings.get("api_key", "")
        base_url = settings.get("base_url", "").strip().rstrip("/")
        model = settings.get("text_model", settings.get("model", ""))

        if not base_url:
            return {
                "error": "No base URL configured. Please set your AI API settings in Tools > AI Settings."
            }
        if not model:
            return {
                "error": "No model configured. Please set your AI API settings in Tools > AI Settings."
            }
        if not api_key:
            return {
                "error": "No API key configured. Please set your AI API key in Tools > AI Settings."
            }

        messages = [{"role": "system", "content": self._system_prompt}]
        for msg in self._chat_history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        return {
            "url": base_url + "/chat/completions",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            "json": {
                "model": model,
                "messages": messages,
                "temperature": settings.get("temperature", 0.3),
                "max_tokens": settings.get("max_tokens", 4096),
            },
        }

    @staticmethod
    def _call_agent_request(request: dict) -> str:
        """Call the LLM API using a prebuilt request."""
        import requests

        error = request.get("error")
        if error:
            return str(error)
        response = requests.post(
            request["url"],
            headers=request.get("headers", {}),
            json=request.get("json", {}),
            timeout=60,
        )
        if response.status_code != 200:
            return f"API error {response.status_code}: {response.text[:200]}"
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "No response content.")
        return "No response from API."

    def _call_agent(self, user_text: str) -> str:
        """Call the LLM API directly using settings from AI Settings."""
        return self._call_agent_request(self._build_agent_request(user_text))

    def _get_context(self) -> str:
        """Get the current editor context."""
        if self._get_context_callback:
            return self._get_context_callback()
        return ""

    def set_context_callback(self, callback: callable) -> None:
        """Set the callback to get editor context."""
        self._get_context_callback = callback

    def set_editor_font(self, font: QtGui.QFont) -> None:
        """Apply font to the transcript and input widgets."""
        self.transcript.setFont(font)
        self.input.setFont(font)

    def _restore_history(self) -> None:
        """Restore chat history from saved file."""
        for msg in self._chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                self._append_user(content)
            elif role == "assistant":
                self._append_assistant(content)

    def clear_history(self) -> None:
        """Clear the chat history."""
        self._chat_history = []
        _save_history([])
        self.transcript.clear()
        self._append_sys("🗑️ History cleared. Current file context will be included with your next question.")

    def _update_provider_label(self, provider_key: str | None = None) -> None:
        """Update the provider status label."""
        if provider_key is None:
            provider_key = ai_settings.get_provider()
        model = ai_settings.get_model()
        name = _PROVIDER_NAMES.get(provider_key, provider_key)
        if model:
            self.status_label.setText(f"🔌 Active: {name} | 🧠 Model: {model}")
        else:
            self.status_label.setText(f"🔌 Active: {name}")

    def _get_wiki_context(self, text: str, current_context: str = "") -> str:
        """Get verified ChiSurf API context for the user's question."""
        repo_root = pathlib.Path(__file__).resolve().parents[4]
        return retrieve_context(text, current_context, repo_root=repo_root, limit=8)

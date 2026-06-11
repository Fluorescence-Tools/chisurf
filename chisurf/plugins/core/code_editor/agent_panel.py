from __future__ import annotations

import json
import logging
import pathlib

from qtpy import QtGui, QtWidgets

try:
    from chisurf.gui.widgets.general import EnterAwarePlainTextEdit
    from chisurf.settings import ai_settings
    from chisurf.settings.path_utils import get_path
except ImportError:
    from chisurf.core.settings import ai_settings
    from chisurf.core.settings.path_utils import get_path
    from chisurf.gui.widgets.general import EnterAwarePlainTextEdit  # noqa: E402

_LOG = logging.getLogger(__name__)

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
        _LOG.warning(f"Could not save agent history: {e}")


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
        _LOG.warning(f"Could not save agent input history: {e}")


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

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_wiki_index)
        search_layout.addWidget(self.refresh_btn)

        self.populate_btn = QtWidgets.QPushButton("Populate Wiki")
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

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        get_context_callback: callable | None = None,
    ):
        super().__init__(parent)
        self._get_context_callback = get_context_callback
        self._chat_history: list = _load_history()
        self._input_history: list[str] = _load_input_history()
        self._input_history_index = -1
        self._system_prompt = (
            "You are a helpful coding assistant for ChiSurf, a fluorescence "
            "spectroscopy analysis application. Help the user with their code "
            "questions. Be concise and direct. "
            "You have access to the LLM Wiki for codebase context. "
            "Format your responses with markdown: use **bold** for emphasis, "
            "`code` for inline code, ```python for code blocks, and LaTeX "
            "\\[ ... \\] for equations. Use numbered lists for steps."
        )
        self.setup_ui()
        self._restore_history()

    def setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(4, 4, 4, 2)

        header_label = QtWidgets.QLabel("AI Assistant")
        header_label.setStyleSheet("font-weight: bold; font-size: 10pt;")
        header_layout.addWidget(header_label)

        header_layout.addStretch()

        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.setMinimumWidth(160)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        header_layout.addWidget(self.provider_combo)

        self.wiki_btn = QtWidgets.QPushButton("Wiki")
        self.wiki_btn.setToolTip("Open LLM Wiki")
        self.wiki_btn.clicked.connect(self._open_wiki)
        header_layout.addWidget(self.wiki_btn)

        layout.addLayout(header_layout)

        self.transcript = QtWidgets.QTextBrowser(self)
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setReadOnly(True)
        self.transcript.setMinimumHeight(200)
        layout.addWidget(self.transcript)

        self.input = EnterAwarePlainTextEdit(self)
        self.input.setPlaceholderText("Ask something... (Enter to send, Shift+Enter for newline)")
        self.input.setMaximumHeight(80)
        self.input.textChanged.connect(self._on_input_changed)
        self.input.sendRequested.connect(self._on_send)
        self.input.historyPrevRequested.connect(self._history_previous)
        self.input.historyNextRequested.connect(self._history_next)
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
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self._populate_providers()
        self._append_sys(
            "I'm an AI coding assistant. Select code in the editor and ask me to:\n"
            "• Explain the selected code\n"
            "• Refactor or improve it\n"
            "• Find bugs or issues\n"
            "• Write tests\n"
            "• Add documentation\n\n"
            "Use 'Populate Wiki' to feed the current codebase to the LLM Wiki."
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
            model = settings.get("model", "")

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
            self.status_label.setText(f"LLM Wiki populated with {count} codebase files.")
            if hasattr(self, "_wiki_progress_callback"):
                self._wiki_progress_callback(f"LLM Wiki populated with {count} codebase files.")
            self._append_sys(f"LLM Wiki populated with {count} codebase files.")
        except Exception as e:
            self.status_label.setText(f"Failed to populate wiki: {e}")
            if hasattr(self, "_wiki_progress_callback"):
                self._wiki_progress_callback(f"Failed to populate wiki: {e}")
            _LOG.error(f"Failed to populate wiki: {e}")

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
        self._process_message(text)

    def _process_message(self, text: str) -> None:
        """Process the user's message and generate a response."""
        context = self._get_context() if self._get_context_callback else ""
        wiki_context = self._get_wiki_context(text)

        full_prompt = context + ("\n\n" if context else "")
        if wiki_context:
            full_prompt += f"LLM Wiki Context:\n{wiki_context}\n\n"
        full_prompt += f"User: {text}"

        self.status_label.setText("Thinking...")
        QtWidgets.QApplication.processEvents()

        try:
            response = self._call_agent(full_prompt)
            self._append_assistant(response)
            self._chat_history.append({"role": "user", "content": text})
            self._chat_history.append({"role": "assistant", "content": response})
            _save_history(self._chat_history)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self._append_sys(error_msg)
            _LOG.error(f"Agent error: {e}")
        finally:
            self.status_label.setText("")

    def _call_agent(self, user_text: str) -> str:
        """Call the LLM API directly using settings from AI Settings."""
        import requests

        provider_key = self.provider_combo.currentData()
        if not provider_key:
            return "No provider selected."

        settings = ai_settings.get_api_settings(provider_key)
        api_key = settings.get("api_key", "")
        base_url = settings.get("base_url", "").strip().rstrip("/")
        model = settings.get("model", "")

        if not base_url:
            return "No base URL configured. Please set your AI API settings in Tools > AI Settings."
        if not model:
            return "No model configured. Please set your AI API settings in Tools > AI Settings."
        if not api_key:
            return "No API key configured. Please set your AI API key in Tools > AI Settings."

        messages = [{"role": "system", "content": self._system_prompt}]
        for msg in self._chat_history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": settings.get("temperature", 0.3),
            "max_tokens": settings.get("max_tokens", 4096),
        }

        url = base_url + "/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return f"API error {response.status_code}: {response.text[:200]}"

        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "No response content.")

        return "No response from API."

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
        self._append_sys("History cleared. Current file context will be included with your next question.")

    def _update_provider_label(self, provider_key: str | None = None) -> None:
        """Update the provider status label."""
        if provider_key is None:
            provider_key = ai_settings.get_provider()
        model = ai_settings.get_model()
        name = _PROVIDER_NAMES.get(provider_key, provider_key)
        if model:
            self.status_label.setText(f"Active: {name} | Model: {model}")
        else:
            self.status_label.setText(f"Active: {name}")

    def _get_wiki_context(self, text: str) -> str:
        """Get relevant wiki and source code context for the user's question."""
        repo_root = pathlib.Path(__file__).resolve().parents[4]
        wiki_dir = repo_root / "llm-wiki" / "wiki"
        contexts = []

        # Get wiki context
        if wiki_dir.exists():
            index_path = wiki_dir / "index.md"
            if index_path.exists():
                try:
                    with open(index_path, encoding="utf-8") as f:
                        f.read()
                except Exception:
                    pass

                keywords = [word.lower() for word in text.split() if len(word) > 3]
                if keywords:
                    for section in ["entities", "concepts", "sources", "synthesis"]:
                        section_dir = wiki_dir / section
                        if not section_dir.exists():
                            continue

                        for page_path in section_dir.glob("*.md"):
                            try:
                                with open(page_path, encoding="utf-8") as f:
                                    content = f.read()
                            except Exception:
                                continue

                            content_lower = content.lower()
                            if any(keyword in content_lower for keyword in keywords):
                                title = self._extract_wiki_title(page_path)
                                contexts.append(f"\n\n[[Wiki: {title}]]\n{content[:3000]}")
                                if len(contexts) >= 2:
                                    break
                        if len(contexts) >= 2:
                            break

        # Get source code context
        source_context = self._get_source_code_context(text, repo_root)
        if source_context:
            contexts.append(source_context)

        if contexts:
            return "\n".join(contexts[:3])
        return ""

    def _get_source_code_context(self, text: str, repo_root: pathlib.Path) -> str:
        """Get relevant source code context for the user's question."""
        keywords = [word.lower() for word in text.split() if len(word) > 3]
        if not keywords:
            return ""

        matches = []
        package_dir = repo_root / "chisurf"
        for py_file in package_dir.rglob("*.py"):
            if "__pycache__" in py_file.parts or "test" in py_file.parts:
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            content_lower = content.lower()
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                rel_path = py_file.relative_to(repo_root)
                matches.append((score, rel_path.as_posix(), content[:4000]))

        matches.sort(reverse=True, key=lambda x: x[0])
        if not matches:
            return ""

        contexts = []
        for score, path, content in matches[:2]:
            contexts.append(f"\n\n[[Source: {path}]]\n```python\n{content}\n```")

        return "\n".join(contexts)

    def _extract_wiki_title(self, page_path: pathlib.Path) -> str:
        """Extract title from a wiki page."""
        try:
            with open(page_path, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return page_path.stem

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

    def _call_agent(self, user_text: str) -> str:
        """Call the LLM API directly using settings from AI Settings."""
        import requests

        provider_key = self.provider_combo.currentData()
        if not provider_key:
            return "No provider selected."

        settings = ai_settings.get_api_settings(provider_key)
        api_key = settings.get("api_key", "")
        base_url = settings.get("base_url", "").strip().rstrip("/")
        model = settings.get("model", "")

        if not base_url:
            return "No base URL configured. Please set your AI API settings in Tools > AI Settings."
        if not model:
            return "No model configured. Please set your AI API settings in Tools > AI Settings."
        if not api_key:
            return "No API key configured. Please set your AI API key in Tools > AI Settings."

        messages = [{"role": "system", "content": self._system_prompt}]
        for msg in self._chat_history[-10:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": settings.get("temperature", 0.3),
            "max_tokens": settings.get("max_tokens", 4096),
        }

        url = base_url + "/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return f"API error {response.status_code}: {response.text[:200]}"

        data = response.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "No response content.")

        return "No response from API."

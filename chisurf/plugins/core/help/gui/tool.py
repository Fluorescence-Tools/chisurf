"""HelpWidget — main GUI for the Help plugin with toolbar and emoji buttons."""

from __future__ import annotations

import pathlib
import re
import webbrowser
from typing import Optional

from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QImage, QTextDocument
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTextBrowser,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import chisurf as cs
import chisurf.core.settings
from chisurf.core.info import help_url
from chisurf.plugins.core.help.gui.client import HelpClient

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    def _noop_persist_plugin_state(name):
        def decorator(cls):
            return cls
        return decorator
    persist_plugin_state = _noop_persist_plugin_state


class HelpTextBrowser(QTextBrowser):
    """Custom text browser that handles local and remote resource loading."""

    def loadResource(self, type, name):
        """Load local or remote resources for the help browser."""
        try:
            if name.scheme() in ("http", "https"):
                import urllib.request
                try:
                    with urllib.request.urlopen(name.toString()) as resp:
                        data = resp.read()
                except Exception:
                    return super().loadResource(type, name)
                if type == QTextDocument.ImageResource:
                    img = QImage()
                    try:
                        if img.loadFromData(data):
                            return img
                    except Exception:
                        pass
                    return data
        except Exception:
            pass
        return super().loadResource(type, name)


@persist_plugin_state("help_documentation")
class HelpWidget(QMainWindow):
    """Documentation browser and help resource viewer for ChiSurf.

    Features
    --------
    - Tree navigation of documentation files
    - Full-text search across all docs
    - In-app Markdown editing and saving
    - Quick links to online docs and video tutorials

    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📖 ChiSurf Help")
        self.setMinimumSize(800, 500)
        self.docs_index = {}
        self.current_path = None
        self.client = HelpClient()
        self._setup_central_widget()
        self._setup_toolbar()
        self.populate_docs()

    # ── central widget ─────────────────────────────────────────────

    def _setup_central_widget(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        header = QHBoxLayout()
        title = QLabel("📖 ChiSurf Documentation and Help Resources")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-weight: bold; font-size: 12pt;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Splitter: tree | content
        splitter = QSplitter(Qt.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(False)
        self.tree.setHeaderLabel("Documents")
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(16)
        self.tree.itemClicked.connect(self._on_item_clicked)
        splitter.addWidget(self.tree)

        self.viewer = HelpTextBrowser()
        self.viewer.setOpenExternalLinks(False)
        self.viewer.setOpenLinks(False)
        self.viewer.anchorClicked.connect(self._on_anchor_clicked)
        self.viewer.setStyleSheet("QTextBrowser { padding: 8px; }")

        self.editor = QPlainTextEdit()
        self.editor.setVisible(False)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        self.title_label = QLabel("📄 Select a document")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color: #888888; font-size: 8pt;")

        right_layout.addWidget(self.title_label)
        right_layout.addWidget(self.path_label)
        right_layout.addWidget(self.viewer, 1)
        right_layout.addWidget(self.editor, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

    # ── toolbar ─────────────────────────────────────────────────────

    def _setup_toolbar(self):
        toolbar = QToolBar("Help Tools")
        toolbar.setMovable(False)
        toolbar.setIconSize(toolbar.iconSize())
        toolbar.setStyleSheet(
            "QToolBar { spacing: 4px; }"
            "QToolButton { font-size: 11pt; padding: 4px 8px; }"
        )
        self.addToolBar(toolbar)

        # Edit
        self.edit_btn = toolbar.addAction("✏️  Edit")
        self.edit_btn.setCheckable(True)
        self.edit_btn.setEnabled(False)
        self.edit_btn.toggled.connect(self._on_edit_toggled)

        # Save
        self.save_btn = toolbar.addAction("💾  Save")
        self.save_btn.setEnabled(False)
        self.save_btn.triggered.connect(self._save_current_document)

        toolbar.addSeparator()

        # Open Documentation
        doc_action = toolbar.addAction("📖  Open Docs")
        doc_action.triggered.connect(self._open_documentation)

        # Video Tutorials
        vid_action = toolbar.addAction("🎬  Video Tutorials")
        vid_action.triggered.connect(self._open_video_tutorials)

        toolbar.addSeparator()

        filter_label = QLabel("🔍 Filter:")
        self.filter_line_edit = QLineEdit()
        self.filter_line_edit.setPlaceholderText("Type to filter documents...")
        self.filter_line_edit.setMaximumWidth(260)
        self.filter_line_edit.textChanged.connect(self._on_filter_text_changed)
        toolbar.addWidget(filter_label)
        toolbar.addWidget(self.filter_line_edit)

        toolbar.addSeparator()

        # Close
        close_action = toolbar.addAction("❌  Close")
        close_action.triggered.connect(self.hide)

    # ── document discovery ──────────────────────────────────────────

    def populate_docs(self):
        """Populate the documentation tree."""
        self.tree.clear()
        self.docs_index = {}
        manual_root = QTreeWidgetItem(self.tree, ["📘 User manual"])
        core_root = QTreeWidgetItem(self.tree, ["📗 Core"])
        plugins_root = QTreeWidgetItem(self.tree, ["📙 Plugins"])
        self._build_manual_docs(manual_root)
        self._build_core_docs(core_root)
        self._build_plugin_docs(plugins_root)

    def _build_manual_docs(self, parent_item):
        base = pathlib.Path(cs.__file__).resolve().parent
        root = base.parent
        docs_dir = root / "docs"
        if not docs_dir.exists():
            return
        for path in sorted(docs_dir.rglob("*.md")):
            try:
                rel = path.relative_to(docs_dir)
            except ValueError:
                rel = path.name
            title = self._extract_markdown_title(path)
            label = title if title else str(rel)
            item = QTreeWidgetItem(parent_item, [label])
            item.setData(0, Qt.UserRole, str(path))
            self._index_document_item(item, path)

    def _build_core_docs(self, parent_item):
        base = pathlib.Path(cs.__file__).resolve().parent
        root = base.parent
        for path in sorted(root.rglob("*.md")):
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == "docs":
                continue
            if "plugins" in rel.parts:
                continue
            title = self._extract_markdown_title(path)
            label = title if title else str(rel)
            item = QTreeWidgetItem(parent_item, [label])
            item.setData(0, Qt.UserRole, str(path))
            self._index_document_item(item, path)

    def _build_plugin_docs(self, parent_item):
        plugin_settings = cs.core.settings.cs_settings.get('plugins', {})
        disabled_plugins = plugin_settings.get('disabled_plugins', [])
        hide_disabled_plugins = plugin_settings.get('hide_disabled_plugins', True)
        plugin_order = plugin_settings.get('plugin_order', {})
        experimental_mode = cs.core.settings.cs_settings.get('enable_experimental', False)

        try:
            plugin_infos = list(cs.plugins.iter_plugins())
        except Exception:
            plugin_infos = []

        module_order_pairs = []

        for info in plugin_infos:
            plugin_name = info.get('plugin_name') or info.get('module_name')
            module_name = info.get('module_name') or ''
            if not plugin_name:
                continue

            clean_name = plugin_name.split(":")[-1].strip() if ":" in plugin_name else plugin_name

            is_disabled = (
                plugin_name in disabled_plugins
                or module_name in disabled_plugins
                or clean_name in disabled_plugins
            )

            if is_disabled and hide_disabled_plugins and not experimental_mode:
                continue

            order = plugin_order.get(plugin_name, 0)
            module_order_pairs.append((order, plugin_name, info))

        module_order_pairs.sort(key=lambda x: (x[0], x[1]))

        for _order, plugin_name, info in module_order_pairs:
            plugin_dir = pathlib.Path(info.get('package_dir')).resolve()
            markdown_files = sorted(plugin_dir.rglob("*.md"))
            if not markdown_files:
                continue

            readme_path = None
            for p in markdown_files:
                if p.name.lower() in {"readme.md", "readme"}:
                    readme_path = p
                    break
            if readme_path is not None:
                readme_title = self._extract_markdown_title(readme_path)
            else:
                readme_title = None

            if len(markdown_files) == 1:
                md_path = markdown_files[0]
                doc_title = self._extract_markdown_title(md_path)
                if doc_title:
                    label = doc_title
                elif readme_path is not None and md_path == readme_path and readme_title:
                    label = readme_title
                else:
                    if ":" in plugin_name:
                        clean_name = plugin_name.split(":")[-1].strip()
                    else:
                        clean_name = plugin_name
                    label = clean_name or plugin_dir.name
                    if md_path.name.lower() not in {"readme.md", "readme"}:
                        label = f"{label} ({md_path.name})"
                item = QTreeWidgetItem(parent_item, [label])
                item.setData(0, Qt.UserRole, str(md_path))
                self._index_document_item(item, md_path)
            else:
                if ":" in plugin_name:
                    clean_name = plugin_name.split(":")[-1].strip()
                else:
                    clean_name = plugin_name
                plugin_label = readme_title or clean_name or plugin_dir.name
                plugin_item = QTreeWidgetItem(parent_item, [plugin_label])
                for md_path in markdown_files:
                    rel = md_path.relative_to(plugin_dir)
                    doc_title = self._extract_markdown_title(md_path)
                    file_label = doc_title if doc_title else str(rel)
                    file_item = QTreeWidgetItem(plugin_item, [file_label])
                    file_item.setData(0, Qt.UserRole, str(md_path))
                    self._index_document_item(file_item, md_path)

    # ── indexing & search ───────────────────────────────────────────

    def _extract_markdown_title(self, path):
        if path is None or not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            m = re.match(r"^(#{1,6})\s+(.*)", stripped)
            if not m:
                continue
            heading = m.group(2)
            heading = re.sub(r"\{\s*#[-\w]+\s*\}\s*$", "", heading).strip()
            if heading:
                return heading
        return None

    def _index_document_item(self, item, path):
        key = str(path)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            text = ""
        lower_text = text.lower()
        headings = []
        first_heading = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            m = re.match(r"^(#{1,6})\s+(.*)", stripped)
            if not m:
                continue
            heading = m.group(2)
            heading = re.sub(r"\{\s*#[-\w]+\s*\}\s*$", "", heading).strip()
            if not heading:
                continue
            headings.append(heading)
            if first_heading is None:
                first_heading = heading
        headings_lc = "\n".join(h.lower() for h in headings) if headings else ""
        title = first_heading.lower() if first_heading is not None else None
        label = item.text(0).lower()
        file_name = path.name.lower()
        self.docs_index[key] = {
            "item": item,
            "path": path,
            "label": label,
            "file_name": file_name,
            "title": title,
            "headings": headings_lc,
            "text": lower_text,
        }

    def _find_first_leaf(self, parent):
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.data(0, Qt.UserRole):
                return child
            result = self._find_first_leaf(child)
            if result is not None:
                return result
        return None

    def _on_filter_text_changed(self, text):
        text = text.strip().lower()
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            child = root.child(i)
            self._apply_filter(child, text)
        self.tree.expandAll()

    def _apply_filter(self, item, text):
        if not text:
            item.setHidden(False)
            for i in range(item.childCount()):
                self._apply_filter(item.child(i), text)
            return True
        label_match = text in item.text(0).lower()
        meta_match = False
        path = item.data(0, Qt.UserRole)
        if path:
            info = self.docs_index.get(str(path))
            if info is not None:
                title = info.get("title") or ""
                if text in title:
                    meta_match = True
                elif text in info.get("file_name", ""):
                    meta_match = True
                elif text in info.get("headings", ""):
                    meta_match = True
                elif text in info.get("text", ""):
                    meta_match = True
        matches_self = label_match or meta_match
        child_matches = False
        for i in range(item.childCount()):
            if self._apply_filter(item.child(i), text):
                child_matches = True
        visible = matches_self or child_matches
        item.setHidden(not visible)
        return visible

    # ── document navigation ─────────────────────────────────────────

    def _on_item_clicked(self, item, column):
        path = item.data(0, Qt.UserRole)
        if not path:
            self._find_first_leaf(item)
            return
        file_path = pathlib.Path(path)
        self._open_markdown_path(file_path)

    def _open_markdown_path(self, file_path: pathlib.Path, anchor: Optional[str] = None):
        if not file_path.exists():
            return
        self.current_path = file_path
        self.title_label.setText(file_path.name)
        self.path_label.setText(str(file_path))
        self.edit_btn.setEnabled(True)
        result = self.client.read_doc(str(file_path))
        if result is None:
            self.viewer.setPlainText(f"Could not read {file_path}")
            return
        text = result.get("content", "")
        if self.edit_btn.isChecked():
            self.editor.setPlainText(text)
            self.editor.show()
            self.viewer.hide()
            self.save_btn.setEnabled(True)
        else:
            html = result.get("html")
            if html is None:
                self.viewer.setPlainText(text)
            else:
                try:
                    base_url = QUrl.fromLocalFile(str(file_path))
                    self.viewer.setHtml(html, base_url)
                except Exception:
                    self.viewer.setHtml(html)
            self.viewer.show()
            self.editor.hide()
            self.save_btn.setEnabled(False)
            if anchor:
                try:
                    self.viewer.scrollToAnchor(anchor)
                except Exception:
                    pass

    def _on_anchor_clicked(self, url):
        try:
            if url.scheme() in ("http", "https"):
                webbrowser.open(url.toString())
                return
            if not url.scheme() and not url.path() and url.fragment():
                self.viewer.scrollToAnchor(url.fragment())
                return
            if url.isLocalFile() or url.scheme() == "file":
                local_path = pathlib.Path(url.toLocalFile())
                fragment = url.fragment() or None
                if local_path.suffix.lower() == ".md":
                    self._open_markdown_path(local_path, fragment)
                    return
            self.viewer.setSource(url)
        except Exception:
            pass

    # ── edit / save ──────────────────────────────────────────────────

    def _on_edit_toggled(self, checked):
        if self.current_path is None:
            if checked:
                self.edit_btn.setChecked(False)
            return

        self.edit_btn.setText("👁️  View" if checked else "✏️  Edit")

        if checked:
            result = self.client.read_doc(str(self.current_path))
            if result is None:
                QMessageBox.critical(
                    self, "Error", f"Could not read {self.current_path}"
                )
                self.edit_btn.setChecked(False)
                return
            text = result.get("content", "")
            self.editor.setPlainText(text)
            self.editor.show()
            self.viewer.hide()
            self.save_btn.setEnabled(True)
        else:
            result = self.client.read_doc(str(self.current_path))
            if result is None:
                self.viewer.setPlainText(f"Could not read {self.current_path}")
                self.viewer.show()
                self.editor.hide()
                self.save_btn.setEnabled(False)
                return
            text = result.get("content", "")
            html = result.get("html")
            if html is None:
                self.viewer.setPlainText(text)
            else:
                try:
                    base_url = QUrl.fromLocalFile(str(self.current_path))
                    self.viewer.setHtml(html, base_url)
                except Exception:
                    self.viewer.setHtml(html)
            self.viewer.show()
            self.editor.hide()
            self.save_btn.setEnabled(False)

    def _save_current_document(self):
        if self.current_path is None:
            return
        text = self.editor.toPlainText()
        ok = self.client.save_doc(str(self.current_path), text)
        if not ok:
            QMessageBox.critical(
                self, "Error", f"Could not save {self.current_path}"
            )
            return
        if not self.edit_btn.isChecked():
            result = self.client.read_doc(str(self.current_path))
            if result is None:
                self.viewer.setPlainText(text)
                return
            html = result.get("html")
            if html is None:
                self.viewer.setPlainText(text)
            else:
                try:
                    base_url = QUrl.fromLocalFile(str(self.current_path))
                    self.viewer.setHtml(html, base_url)
                except Exception:
                    self.viewer.setHtml(html)

    # ── external links ──────────────────────────────────────────────

    def _open_documentation(self):
        """Open the ChiSurf documentation in a web browser."""
        webbrowser.open_new(help_url)

    def _open_video_tutorials(self):
        """Open the ChiSurf video tutorials in a web browser."""
        webbrowser.open_new("https://www.peulen.xyz/tutorial/")

"""
ChiSurf Help Plugin

This plugin provides access to the ChiSurf documentation and help resources.

Features:
- Open the ChiSurf documentation in a web browser
- Access to video tutorials
- Access to user guides and tutorials
- Quick reference for common tasks
"""

import webbrowser
import pathlib
import re
import html as _html
import pkgutil
import importlib
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextBrowser, QTreeWidget, QTreeWidgetItem, QSplitter, QLineEdit, QPlainTextEdit, QMessageBox
from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QTextDocument, QImage

import chisurf as cs
import chisurf.plugins
import chisurf.core.settings
from chisurf.core import info

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c


try:
    import markdown
except Exception:
    markdown = None

# Define the plugin name - this will appear in the Plugins menu
name = "Help:Documentation"

class HelpTextBrowser(QTextBrowser):
    def loadResource(self, type, name):  # type: ignore[override]
        try:
            if name.scheme() in ("http", "https"):
                import urllib.request

                try:
                    with urllib.request.urlopen(name.toString()) as resp:  # nosec
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
                    return super().loadResource(type, name)
                return data
        except Exception:
            pass
        return super().loadResource(type, name)


@persist_plugin_state("help_documentation")
class HelpWidget(QWidget):
    """
    A widget that provides access to ChiSurf documentation and help resources.
    """

    def __init__(self, parent=None):
        """Initialize the help widget."""
        super().__init__(parent)
        self.setWindowTitle("ChiSurf Help")
        self.setMinimumSize(800, 500)
        self.docs_index = {}
        self.current_path = None
        self.setup_ui()
        self.populate_docs()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        description = QLabel("ChiSurf Documentation and Help Resources")
        description.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        description.setStyleSheet("font-weight: bold; font-size: 12pt;")
        header_layout.addWidget(description)
        header_layout.addStretch()

        filter_label = QLabel("Filter:")
        self.filter_line_edit = QLineEdit()
        self.filter_line_edit.setPlaceholderText("Type to filter documents...")
        self.filter_line_edit.textChanged.connect(self.on_filter_text_changed)
        header_layout.addWidget(filter_label)
        header_layout.addWidget(self.filter_line_edit)

        layout.addLayout(header_layout)

        splitter = QSplitter(Qt.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(False)
        self.tree.setHeaderLabel("Documents")
        self.tree.setAlternatingRowColors(True)
        self.tree.setIndentation(16)
        self.tree.itemClicked.connect(self.on_item_clicked)
        splitter.addWidget(self.tree)

        self.viewer = HelpTextBrowser()
        self.viewer.setOpenExternalLinks(False)
        self.viewer.setOpenLinks(False)
        self.viewer.anchorClicked.connect(self.on_anchor_clicked)
        self.viewer.setStyleSheet(
            "QTextBrowser {"
            " padding: 8px;"
            " }"
        )

        self.editor = QPlainTextEdit()
        self.editor.setVisible(False)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(4)

        self.title_label = QLabel("Select a document")
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

        buttons_layout = QHBoxLayout()
        buttons_layout.addStretch()

        self.edit_button = QPushButton("Edit")
        self.edit_button.setCheckable(True)
        self.edit_button.setEnabled(False)
        self.edit_button.toggled.connect(self.on_edit_toggled)
        buttons_layout.addWidget(self.edit_button)

        self.save_button = QPushButton("Save")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_current_document)
        buttons_layout.addWidget(self.save_button)

        doc_button = QPushButton("Open Documentation")
        doc_button.clicked.connect(self.open_documentation)
        buttons_layout.addWidget(doc_button)

        tutorials_button = QPushButton("Video Tutorials")
        tutorials_button.clicked.connect(self.open_video_tutorials)
        buttons_layout.addWidget(tutorials_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.hide)
        buttons_layout.addWidget(close_button)

        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def populate_docs(self):
        self.tree.clear()
        self.docs_index = {}
        manual_root = QTreeWidgetItem(self.tree, ["User manual"])
        core_root = QTreeWidgetItem(self.tree, ["Core"])
        plugins_root = QTreeWidgetItem(self.tree, ["Plugins"])
        self.build_manual_docs(manual_root)
        self.build_core_docs(core_root)
        self.build_plugin_docs(plugins_root)

    def build_manual_docs(self, parent_item):
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

    def build_core_docs(self, parent_item):
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
    def build_plugin_docs(self, parent_item):
        """Populate the Plugins section, sorted like the main GUI plugin menu.

        Ordering is based on the same plugin settings (plugin_order, disabled plugins),
        but labels in the tree still prefer README titles where available.
        """
        # Plugin settings and ordering
        plugin_settings = cs.core.settings.cs_settings.get('plugins', {})
        disabled_plugins = plugin_settings.get('disabled_plugins', [])
        hide_disabled_plugins = plugin_settings.get('hide_disabled_plugins', True)
        plugin_order = plugin_settings.get('plugin_order', {})
        experimental_mode = cs.core.settings.cs_settings.get('enable_experimental', False)

        # Discover plugin modules (built-in + user, including nested subpackages)
        try:
            plugin_infos = list(cs.plugins.iter_plugins())
        except Exception:
            plugin_infos = []

        module_order_pairs = []  # (order, plugin_name, info)

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

        # Sort by order (ascending) and then by plugin_name (alphabetically),
        # mirroring populate_plugins in the main GUI.
        module_order_pairs.sort(key=lambda x: (x[0], x[1]))

        for _order, plugin_name, info in module_order_pairs:
            plugin_dir = pathlib.Path(info.get('package_dir')).resolve()
            markdown_files = sorted(plugin_dir.rglob("*.md"))
            if not markdown_files:
                continue

            # Try to determine a human-friendly title from the plugin's README.
            readme_path = None
            for p in markdown_files:
                if p.name.lower() in {"readme.md", "readme"}:
                    readme_path = p
                    break
            readme_title = self._extract_markdown_title(readme_path) if readme_path is not None else None

            # If there is only a single Markdown file for this plugin, show it as a leaf
            # directly under the Plugins node (no nested subtree).
            if len(markdown_files) == 1:
                md_path = markdown_files[0]
                doc_title = self._extract_markdown_title(md_path)
                if doc_title:
                    label = doc_title
                elif readme_path is not None and md_path == readme_path and readme_title:
                    label = readme_title
                else:
                    # Fallback: clean plugin name (without submenu prefix) or module name.
                    clean_name = plugin_name.split(":")[-1].strip() if ":" in plugin_name else plugin_name
                    label = clean_name or plugin_dir.name
                    # If the file name is not the default README, include it in the label.
                    if md_path.name.lower() not in {"readme.md", "readme"}:
                        label = f"{label} ({md_path.name})"
                item = QTreeWidgetItem(parent_item, [label])
                item.setData(0, Qt.UserRole, str(md_path))
                self._index_document_item(item, md_path)
            else:
                # Group node: prefer README title, then clean plugin name, then directory name.
                clean_name = plugin_name.split(":")[-1].strip() if ":" in plugin_name else plugin_name
                plugin_label = readme_title or clean_name or plugin_dir.name
                plugin_item = QTreeWidgetItem(parent_item, [plugin_label])
                for md_path in markdown_files:
                    rel = md_path.relative_to(plugin_dir)
                    doc_title = self._extract_markdown_title(md_path)
                    file_label = doc_title if doc_title else str(rel)
                    file_item = QTreeWidgetItem(plugin_item, [file_label])
                    file_item.setData(0, Qt.UserRole, str(md_path))
                    self._index_document_item(file_item, md_path)

    def _extract_markdown_title(self, path: pathlib.Path):
        """Return the first heading line from a Markdown file, if any.

        We look for the first line that starts with one or more '#' characters
        followed by a space, and use the remainder as the title. Any trailing
        '{#anchor}' marker is stripped.
        """
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

    def _index_document_item(self, item, path: pathlib.Path):
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

    def find_first_leaf(self, parent):
        for i in range(parent.childCount()):
            child = parent.child(i)
            if child.data(0, Qt.UserRole):
                return child
            result = self.find_first_leaf(child)
            if result is not None:
                return result
        return None

    def on_filter_text_changed(self, text):
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
                child = item.child(i)
                self._apply_filter(child, text)
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
            child = item.child(i)
            if self._apply_filter(child, text):
                child_matches = True
        visible = matches_self or child_matches
        item.setHidden(not visible)
        return visible

    def on_item_clicked(self, item, column):
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        file_path = pathlib.Path(path)
        self.open_markdown_path(file_path)

    def open_markdown_path(self, file_path: pathlib.Path, anchor: str | None = None):
        if not file_path.exists():
            return
        self.current_path = file_path
        self.title_label.setText(file_path.name)
        self.path_label.setText(str(file_path))
        self.edit_button.setEnabled(True)
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            self.viewer.setPlainText(f"Could not read {file_path}:\n{exc}")
            return
        if self.edit_button.isChecked():
            self.editor.setPlainText(text)
            self.editor.show()
            self.viewer.hide()
            self.save_button.setEnabled(True)
        else:
            html = self.render_markdown(text)
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
            self.save_button.setEnabled(False)
            if anchor:
                try:
                    self.viewer.scrollToAnchor(anchor)
                except Exception:
                    pass

    def on_anchor_clicked(self, url):
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
                    self.open_markdown_path(local_path, fragment)
                    return
            self.viewer.setSource(url)
        except Exception:
            pass

    def on_edit_toggled(self, checked):
        if self.current_path is None:
            # No document selected; reset toggle.
            if checked:
                self.edit_button.setChecked(False)
            return

        self.edit_button.setText("View" if checked else "Edit")

        if checked:
            try:
                text = self.current_path.read_text(encoding="utf-8")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Could not read {self.current_path}:\n{exc}")
                self.edit_button.setChecked(False)
                return
            self.editor.setPlainText(text)
            self.editor.show()
            self.viewer.hide()
            self.save_button.setEnabled(True)
        else:
            # Switch back to rendered view using the current file contents.
            try:
                text = self.current_path.read_text(encoding="utf-8")
            except Exception as exc:
                self.viewer.setPlainText(f"Could not read {self.current_path}:\n{exc}")
                self.viewer.show()
                self.editor.hide()
                self.save_button.setEnabled(False)
                return
            html = self.render_markdown(text)
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
            self.save_button.setEnabled(False)

    def save_current_document(self):
        if self.current_path is None:
            return
        text = self.editor.toPlainText()
        try:
            self.current_path.write_text(text, encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not save {self.current_path}:\n{exc}")
            return

        # If we're in view mode, refresh the rendered content immediately.
        if not self.edit_button.isChecked():
            html = self.render_markdown(text)
            if html is None:
                self.viewer.setPlainText(text)
            else:
                try:
                    base_url = QUrl.fromLocalFile(str(self.current_path))
                    self.viewer.setHtml(html, base_url)
                except Exception:
                    self.viewer.setHtml(html)

    def render_markdown(self, text):
        prepared = self._prepare_markdown_with_heading_ids(text)
        if markdown is not None:
            try:
                body = markdown.markdown(
                    prepared,
                    output_format="html5",
                    extensions=["attr_list"],
                )
            except Exception:
                body = markdown.markdown(prepared, output_format="html5")
        else:
            body = self._basic_markdown_to_html(prepared)

        css = (
            "<style>"
            "body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt; }"
            "h1, h2, h3 { margin-top: 0.8em; margin-bottom: 0.4em; }"
            "pre, code { font-family: 'Consolas', 'Courier New', monospace; }"
            "pre { padding: 6px; border-radius: 3px; }"
            "</style>"
        )
        return f"<html><head>{css}</head><body>{body}</body></html>"

    def _prepare_markdown_with_heading_ids(self, text):
        lines = text.splitlines()
        out_lines = []
        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if not m:
                out_lines.append(line)
                continue
            prefix, title = m.group(1), m.group(2)
            if re.search(r"\{\s*#[-\w]+\s*\}\s*$", title):
                out_lines.append(line)
                continue
            slug = self._slugify_heading(title)
            out_lines.append(f"{prefix} {title} " + "{" + f"#{slug}" + "}")
        return "\n".join(out_lines)

    def _slugify_heading(self, text):
        slug = text.strip().lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"\s+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug or "section"

    def _basic_markdown_to_html(self, text):
        """Very simple Markdown-to-HTML converter used as a fallback.

        Supports headings (# .. ######), and basic inline **bold**, *italic*, and `code`.
        Also handles simple unordered ("- ", "* ") and ordered ("1. ") lists.
        """
        lines = text.splitlines()
        html_lines = []

        in_ul = False
        in_ol = False

        def _process_inline_with_images(raw: str) -> str:
            placeholders = {}

            def _img_repl(m):
                alt = m.group(1)
                src = m.group(2).strip()
                key = f"__CS_IMG_{len(placeholders)}__"
                alt_esc = _html.escape(alt)
                src_esc = _html.escape(src, quote=True)
                placeholders[key] = f'<img src="{src_esc}" alt="{alt_esc}">'  # nosec
                return key

            tmp = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img_repl, raw)
            escaped = _html.escape(tmp)
            escaped = self._apply_inline_markdown(escaped)
            for key, tag in placeholders.items():
                escaped = escaped.replace(key, tag)
            return escaped

        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                if in_ul:
                    html_lines.append("</ul>")
                    in_ul = False
                if in_ol:
                    html_lines.append("</ol>")
                    in_ol = False
                html_lines.append("")
                continue

            # Unordered list item
            m_ul = re.match(r"^[-*+]\s+(.*)", stripped)
            # Ordered list item (e.g. "1. item")
            m_ol = re.match(r"^(\d+)[.)]\s+(.*)", stripped)

            if m_ul:
                if in_ol:
                    html_lines.append("</ol>")
                    in_ol = False
                if not in_ul:
                    html_lines.append("<ul>")
                    in_ul = True
                content = _process_inline_with_images(m_ul.group(1))
                html_lines.append(f"<li>{content}</li>")
                continue

            if m_ol:
                if in_ul:
                    html_lines.append("</ul>")
                    in_ul = False
                if not in_ol:
                    html_lines.append("<ol>")
                    in_ol = True
                content = _process_inline_with_images(m_ol.group(2))
                html_lines.append(f"<li>{content}</li>")
                continue

            # Close any open list before non-list content
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False

            # Headings
            m = re.match(r"^(#{1,6})\s+(.*)", stripped)
            if m:
                level = len(m.group(1))
                raw_content = m.group(2)
                anchor = None
                m_id = re.search(r"\{\s*#([-\w]+)\s*\}\s*$", raw_content)
                if m_id:
                    anchor = m_id.group(1)
                    raw_content = raw_content[:m_id.start()].rstrip()
                else:
                    anchor = self._slugify_heading(raw_content)
                content = _process_inline_with_images(raw_content)
                if anchor:
                    html_lines.append(f'<h{level} id="{anchor}">{content}</h{level}>' )
                else:
                    html_lines.append(f"<h{level}>{content}</h{level}>")
            else:
                content = _process_inline_with_images(line)
                html_lines.append(f"<p>{content}</p>")
        if in_ul:
            html_lines.append("</ul>")
        if in_ol:
            html_lines.append("</ol>")

        return "\n".join(html_lines)

    def _apply_inline_markdown(self, text):
        # `code`
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        # **bold**
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        # *italic*
        text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
        return text

    def open_documentation(self):
        """Open the ChiSurf documentation in a web browser."""
        webbrowser.open_new(info.help_url)

    def open_video_tutorials(self):
        """Open the ChiSurf video tutorials in a web browser."""
        webbrowser.open_new("https://www.peulen.xyz/tutorial/")

# When the plugin is loaded as a module with __name__ == "plugin",
# this code will be executed
if __name__ == "plugin":
    # Create an instance of the HelpWidget class
    window = HelpWidget()
    # Show the window
    window.show()

from __future__ import annotations

import yaml
from qtpy import QtCore, QtGui, QtWidgets

import chisurf as cs
import chisurf.core.settings
from chisurf import logging

EDITOR_DEFAULT_FONTS = [
    "Courier New",
    "Consolas",
    "SF Mono",
    "Menlo",
    "Monaco",
    "DejaVu Sans Mono",
    "Liberation Mono",
    "Courier",
    "Monospace",
]

EDITOR_LANGUAGE_OPTIONS = ["Python", "JSON", "YAML", "Plain text"]

EDITOR_COLOR_SCHEMES = {
    "ChiSurf": {
        "paper_color": "#cfcfcf",
        "default_color": "#000006",
        "margins_background_color": "#808080",
        "marker_background_color": "#f0f0f0",
        "caret_line_background_color": "#afafaf",
    },
    "Light": {
        "paper_color": "#ffffff",
        "default_color": "#000000",
        "margins_background_color": "#f0f0f0",
        "marker_background_color": "#d8e8ff",
        "caret_line_background_color": "#eef4ff",
    },
    "Dark": {
        "paper_color": "#1e1e1e",
        "default_color": "#d4d4d4",
        "margins_background_color": "#252526",
        "marker_background_color": "#3c3c3c",
        "caret_line_background_color": "#2d2d30",
    },
    "Monokai": {
        "paper_color": "#272822",
        "default_color": "#f8f8f2",
        "margins_background_color": "#1e1f1c",
        "marker_background_color": "#3e3d32",
        "caret_line_background_color": "#3e3d32",
    },
}

EDITOR_SETTINGS_KEYS = [
    "font_family",
    "font_size",
    "language",
    "color_scheme",
    "paper_color",
    "default_color",
    "margins_background_color",
    "marker_background_color",
    "caret_line_background_color",
    "caret_line_visible",
    "line_numbers_visible",
    "enable_lsp",
    "enable_ruff",
    "run_ruff_on_save",
    "ruff_timeout_ms",
    "ruff_extra_args",
    "enable_rpc",
    "rpc_host",
    "rpc_cmd_port",
    "rpc_pub_port",
]


def editor_language_key(language: str | None) -> str:
    """Return the normalized internal language key for *language*."""
    key = str(language or "").strip().lower().replace("_", "-")
    if key in {"plain-text", "plain text"}:
        return "plain"
    return key


def normalize_editor_language(language: str | None) -> str:
    """Return a display language name for *language*."""
    key = editor_language_key(language)
    if key in {"json"}:
        return "JSON"
    if key in {"yaml", "yml"}:
        return "YAML"
    if key in {"plain", "text"}:
        return "Plain text"
    return "Python"


def default_editor_settings() -> dict[str, str | int | bool | list[str]]:
    """Return the default editor settings dictionary."""
    settings: dict[str, str | int | bool | list[str]] = {
        "font_family": "Courier New",
        "font_size": 9,
        "language": "Python",
        "color_scheme": "ChiSurf",
        "caret_line_visible": False,
        "line_numbers_visible": True,
        "enable_lsp": True,
        "enable_ruff": True,
        "run_ruff_on_save": False,
        "ruff_timeout_ms": 5000,
        "ruff_extra_args": [],
        "enable_rpc": False,
        "rpc_host": "127.0.0.1",
        "rpc_cmd_port": 8775,
        "rpc_pub_port": 8776,
    }
    settings.update(EDITOR_COLOR_SCHEMES["ChiSurf"])
    return settings


def get_editor_settings() -> dict[str, str | int | bool | list[str]]:
    """Return editor settings merged with the current ChiSurf GUI settings."""
    settings = default_editor_settings()
    cfg = cs.core.settings.gui.get("editor", {})
    if not isinstance(cfg, dict):
        cfg = {}
    for key, value in cfg.items():
        if key in EDITOR_SETTINGS_KEYS:
            settings[key] = value
    scheme = settings.get("color_scheme")
    if scheme in EDITOR_COLOR_SCHEMES:
        settings.update(EDITOR_COLOR_SCHEMES[scheme])
    for key, value in cfg.items():
        if key in EDITOR_SETTINGS_KEYS:
            settings[key] = value
    settings["language"] = normalize_editor_language(settings.get("language"))
    return settings


def make_editor_font(settings: dict | None = None) -> QtGui.QFont:
    """Create a QFont from editor settings."""
    cfg = settings or get_editor_settings()
    font = QtGui.QFont()
    font.setFamily(str(cfg.get("font_family", "Courier New")))
    try:
        font.setPointSize(int(cfg.get("font_size", 9)))
    except (TypeError, ValueError):
        font.setPointSize(9)
    return font


def save_editor_settings(settings: dict) -> bool:
    """Persist editor settings to the user ChiSurf settings file."""
    editor_settings = {key: settings[key] for key in EDITOR_SETTINGS_KEYS if key in settings}
    if not editor_settings:
        return False

    try:
        settings_file = cs.core.settings.chisurf_settings_file
        data = cs.core.settings.safe_open_file(
            file_path=settings_file,
            processor=yaml.safe_load,
            default_value={},
            error_message=f"Error opening settings file {settings_file}",
        )
        if not isinstance(data, dict):
            data = {}

        gui_cfg = data.setdefault("gui", {})
        if not isinstance(gui_cfg, dict):
            gui_cfg = {}
            data["gui"] = gui_cfg

        editor_cfg = gui_cfg.setdefault("editor", {})
        if not isinstance(editor_cfg, dict):
            editor_cfg = {}
            gui_cfg["editor"] = editor_cfg

        editor_cfg.update(editor_settings)

        cs_settings = cs.core.settings.cs_settings
        if not isinstance(cs_settings, dict):
            cs.core.settings.cs_settings = {}
            cs_settings = cs.core.settings.cs_settings
        cs_gui = cs_settings.setdefault("gui", {})
        if not isinstance(cs_gui, dict):
            cs_gui = {}
            cs_settings["gui"] = cs_gui
        cs_editor = cs_gui.setdefault("editor", {})
        if not isinstance(cs_editor, dict):
            cs_editor = {}
            cs_gui["editor"] = cs_editor
        cs_editor.update(editor_settings)

        gui = cs.core.settings.gui
        if not isinstance(gui, dict):
            cs.core.settings.gui = {}
            gui = cs.core.settings.gui
        gui_editor = gui.setdefault("editor", {})
        if not isinstance(gui_editor, dict):
            gui_editor = {}
            gui["editor"] = gui_editor
        gui_editor.update(editor_settings)

        with open(settings_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        logging.log(1, f"Error saving editor settings: {e}")
        return False


class EditorSettingsDialog(QtWidgets.QDialog):
    """Dialog for persistent code editor settings."""

    settings_applied = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_buttons = {}
        self.color_labels = {
            "paper_color": "Paper color",
            "default_color": "Text color",
            "margins_background_color": "Margin color",
            "marker_background_color": "Marker color",
            "caret_line_background_color": "Current line color",
        }
        self.setWindowTitle("Editor Settings")
        self.resize(440, 560)
        self.setup_ui()
        self._set_controls_from_settings(get_editor_settings())

    def setup_ui(self) -> None:
        """Create the settings dialog widgets."""
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.setObjectName("language_combo")
        self.language_combo.setToolTip("Syntax highlighting language support")
        for language in EDITOR_LANGUAGE_OPTIONS:
            self.language_combo.addItem(language)
        form.addRow("Language support:", self.language_combo)

        self.font_combo = QtWidgets.QComboBox()
        self.font_combo.setObjectName("font_combo")
        self.font_combo.setEditable(True)
        self._populate_fonts()
        form.addRow("Font:", self.font_combo)

        self.font_size_spin = QtWidgets.QSpinBox()
        self.font_size_spin.setObjectName("font_size_spin")
        self.font_size_spin.setRange(4, 72)
        form.addRow("Font size:", self.font_size_spin)

        self.color_scheme_combo = QtWidgets.QComboBox()
        self.color_scheme_combo.setObjectName("color_scheme_combo")
        for scheme in EDITOR_COLOR_SCHEMES:
            self.color_scheme_combo.addItem(scheme)
        self.color_scheme_combo.currentTextChanged.connect(self._on_color_scheme_changed)
        form.addRow("Color scheme:", self.color_scheme_combo)

        layout.addLayout(form)

        group = QtWidgets.QGroupBox("Colors")
        colors_layout = QtWidgets.QFormLayout()
        for key, label in self.color_labels.items():
            button = QtWidgets.QPushButton("#ffffff")
            button.setObjectName(f"{key}_button")
            button.setFixedWidth(120)
            button.clicked.connect(lambda _checked, item=key: self._choose_color(item))
            self.color_buttons[key] = button
            colors_layout.addRow(label, button)
        group.setLayout(colors_layout)
        layout.addWidget(group)

        self.caret_line_check = QtWidgets.QCheckBox("Highlight current line")
        self.caret_line_check.setObjectName("caret_line_check")
        layout.addWidget(self.caret_line_check)

        behavior_group = QtWidgets.QGroupBox("Editor behavior")
        behavior_layout = QtWidgets.QFormLayout()
        self.line_numbers_check = QtWidgets.QCheckBox("Show line numbers")
        self.line_numbers_check.setObjectName("line_numbers_check")
        self.lsp_check = QtWidgets.QCheckBox("Enable Python LSP")
        self.lsp_check.setObjectName("lsp_check")
        self.ruff_check = QtWidgets.QCheckBox("Enable Ruff checks")
        self.ruff_check.setObjectName("ruff_check")
        self.ruff_on_save_check = QtWidgets.QCheckBox("Run Ruff after save")
        self.ruff_on_save_check.setObjectName("ruff_on_save_check")
        self.ruff_timeout_spin = QtWidgets.QSpinBox()
        self.ruff_timeout_spin.setObjectName("ruff_timeout_spin")
        self.ruff_timeout_spin.setRange(1000, 60000)
        self.ruff_timeout_spin.setSingleStep(500)
        self.ruff_timeout_spin.setSuffix(" ms")
        self.ruff_args_edit = QtWidgets.QLineEdit()
        self.ruff_args_edit.setObjectName("ruff_args_edit")
        self.rpc_check = QtWidgets.QCheckBox("Enable editor RPC server")
        self.rpc_check.setObjectName("rpc_check")
        self.rpc_host_edit = QtWidgets.QLineEdit()
        self.rpc_host_edit.setObjectName("rpc_host_edit")
        self.rpc_cmd_spin = QtWidgets.QSpinBox()
        self.rpc_cmd_spin.setObjectName("rpc_cmd_spin")
        self.rpc_cmd_spin.setRange(1024, 65535)
        self.rpc_pub_spin = QtWidgets.QSpinBox()
        self.rpc_pub_spin.setObjectName("rpc_pub_spin")
        self.rpc_pub_spin.setRange(1024, 65535)
        behavior_layout.addRow(self.line_numbers_check)
        behavior_layout.addRow(self.lsp_check)
        behavior_layout.addRow(self.ruff_check)
        behavior_layout.addRow(self.ruff_on_save_check)
        behavior_layout.addRow("Ruff timeout:", self.ruff_timeout_spin)
        behavior_layout.addRow("Ruff extra args:", self.ruff_args_edit)
        behavior_layout.addRow(self.rpc_check)
        behavior_layout.addRow("RPC host:", self.rpc_host_edit)
        behavior_layout.addRow("RPC command port:", self.rpc_cmd_spin)
        behavior_layout.addRow("RPC publish port:", self.rpc_pub_spin)
        behavior_group.setLayout(behavior_layout)
        layout.addWidget(behavior_group)

        button_layout = QtWidgets.QHBoxLayout()
        self.apply_button = QtWidgets.QPushButton("Apply & Save")
        self.apply_button.setObjectName("apply_settings_button")
        self.restore_button = QtWidgets.QPushButton("Restore Defaults")
        self.restore_button.setObjectName("restore_defaults_button")
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.setObjectName("close_settings_button")

        self.apply_button.clicked.connect(self._apply_clicked)
        self.restore_button.clicked.connect(self.restore_defaults)
        self.close_button.clicked.connect(self.reject)

        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.restore_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

    def _populate_fonts(self) -> None:
        """Populate the font combo box with common monospace fonts first."""
        added = set()
        try:
            available = set(QtGui.QFontDatabase.families())
        except Exception:
            available = set()

        for font in EDITOR_DEFAULT_FONTS:
            if font in available or not available:
                self.font_combo.addItem(font)
                added.add(font)

        for font in sorted(available):
            if font not in added:
                self.font_combo.addItem(font)

        if self.font_combo.count() == 0:
            self.font_combo.addItem("Courier New")

    def _set_controls_from_settings(self, settings: dict) -> None:
        """Populate dialog controls from *settings*."""
        language = normalize_editor_language(settings.get("language"))
        index = self.language_combo.findText(language)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)

        font_family = str(settings.get("font_family", "Courier New"))
        index = self.font_combo.findText(font_family, QtCore.Qt.MatchFixedString)
        if index < 0:
            index = self.font_combo.findText(font_family, QtCore.Qt.MatchContains)
        if index >= 0:
            self.font_combo.setCurrentIndex(index)
        else:
            self.font_combo.setCurrentText(font_family)

        try:
            self.font_size_spin.setValue(int(settings.get("font_size", 9)))
        except (TypeError, ValueError):
            self.font_size_spin.setValue(9)

        scheme = str(settings.get("color_scheme", "ChiSurf"))
        index = self.color_scheme_combo.findText(scheme)
        if index >= 0:
            self.color_scheme_combo.setCurrentIndex(index)

        for key, button in self.color_buttons.items():
            button.setText(str(settings.get(key, "#ffffff")))
            self._update_color_button(button)

        self.caret_line_check.setChecked(bool(settings.get("caret_line_visible", False)))
        self.line_numbers_check.setChecked(bool(settings.get("line_numbers_visible", True)))
        self.lsp_check.setChecked(bool(settings.get("enable_lsp", True)))
        self.ruff_check.setChecked(bool(settings.get("enable_ruff", True)))
        self.ruff_on_save_check.setChecked(bool(settings.get("run_ruff_on_save", False)))
        try:
            self.ruff_timeout_spin.setValue(int(settings.get("ruff_timeout_ms", 5000)))
        except (TypeError, ValueError):
            self.ruff_timeout_spin.setValue(5000)
        extra_args = settings.get("ruff_extra_args", [])
        if isinstance(extra_args, list):
            self.ruff_args_edit.setText(" ".join(str(arg) for arg in extra_args))
        else:
            self.ruff_args_edit.setText(str(extra_args))
        self.rpc_check.setChecked(bool(settings.get("enable_rpc", False)))
        self.rpc_host_edit.setText(str(settings.get("rpc_host", "127.0.0.1")))
        try:
            self.rpc_cmd_spin.setValue(int(settings.get("rpc_cmd_port", 8775)))
            self.rpc_pub_spin.setValue(int(settings.get("rpc_pub_port", 8776)))
        except (TypeError, ValueError):
            self.rpc_cmd_spin.setValue(8775)
            self.rpc_pub_spin.setValue(8776)

    def _on_color_scheme_changed(self, scheme: str) -> None:
        """Update color controls when the color scheme changes."""
        if scheme in EDITOR_COLOR_SCHEMES:
            for key, value in EDITOR_COLOR_SCHEMES[scheme].items():
                button = self.color_buttons.get(key)
                if button is not None:
                    button.setText(str(value))
                    self._update_color_button(button)

    def _choose_color(self, key: str) -> None:
        """Open a color picker and update the selected color button."""
        button = self.color_buttons[key]
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(button.text()), self)
        if color.isValid():
            text = f"#{color.red():02x}{color.green():02x}{color.blue():02x}"
            button.setText(text)
            self._update_color_button(button)

    @staticmethod
    def _update_color_button(button: QtWidgets.QPushButton) -> None:
        """Update a color button's text color for readability."""
        color = QtGui.QColor(button.text())
        brightness = sum(color.getRgb()[:3])
        button.setStyleSheet(
            f"background-color: {button.text()}; color: {'black' if brightness > 382 else 'white'};"
        )

    def editor_settings(self) -> dict:
        """Return the settings currently selected in the dialog."""
        settings = {
            "font_family": self.font_combo.currentText(),
            "font_size": self.font_size_spin.value(),
            "language": self.language_combo.currentText(),
            "color_scheme": self.color_scheme_combo.currentText(),
            "caret_line_visible": self.caret_line_check.isChecked(),
            "line_numbers_visible": self.line_numbers_check.isChecked(),
            "enable_lsp": self.lsp_check.isChecked(),
            "enable_ruff": self.ruff_check.isChecked(),
            "run_ruff_on_save": self.ruff_on_save_check.isChecked(),
            "ruff_timeout_ms": self.ruff_timeout_spin.value(),
            "ruff_extra_args": self._split_extra_args(self.ruff_args_edit.text()),
            "enable_rpc": self.rpc_check.isChecked(),
            "rpc_host": self.rpc_host_edit.text().strip() or "127.0.0.1",
            "rpc_cmd_port": self.rpc_cmd_spin.value(),
            "rpc_pub_port": self.rpc_pub_spin.value(),
        }
        settings.update({key: button.text() for key, button in self.color_buttons.items()})
        return settings

    @staticmethod
    def _split_extra_args(text: str) -> list[str]:
        """Return shell-like extra arguments from a single text field."""
        import shlex

        try:
            return shlex.split(text)
        except ValueError:
            return text.split()

    def restore_defaults(self) -> None:
        """Reset dialog controls to default editor settings."""
        self._set_controls_from_settings(default_editor_settings())

    def _apply_clicked(self) -> None:
        """Save and apply the current dialog settings."""
        self.settings_applied.emit(self.editor_settings())
        self.accept()


__all__ = [
    "EDITOR_COLOR_SCHEMES",
    "EDITOR_DEFAULT_FONTS",
    "EDITOR_LANGUAGE_OPTIONS",
    "EDITOR_SETTINGS_KEYS",
    "EditorSettingsDialog",
    "default_editor_settings",
    "editor_language_key",
    "get_editor_settings",
    "make_editor_font",
    "normalize_editor_language",
    "save_editor_settings",
]

import yaml

import chisurf as cs
from chisurf.plugins.core.code_editor.text_editor import (
    CodeEditor,
    EditorSettingsDialog,
    TextEditor,
    save_editor_settings,
)


class TestCodeEditorSettings:
    """Tests for persistent code editor settings."""

    def test_settings_button_uses_gear_and_settings_tooltip(self, qapp):
        """The editor settings button should use a gear icon and Settings tooltip."""
        widget = CodeEditor()
        button = widget.create_settings_button()

        assert "⚙" in button.text()
        assert button.toolTip() == "Settings"

    def test_settings_dialog_has_font_combo_size_spin_and_options(self, qapp):
        """The settings dialog exposes font, size, language, and color options."""
        dialog = EditorSettingsDialog()

        assert dialog.font_combo.count() > 0
        assert dialog.font_size_spin.minimum() == 4
        assert dialog.font_size_spin.maximum() == 72
        assert dialog.language_combo.count() == 4
        assert dialog.color_scheme_combo.count() >= 4

        settings = dialog.editor_settings()
        for key in [
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
        ]:
            assert key in settings

    def test_text_editor_settings_apply_to_widget(self, qapp):
        """Editor settings should update font, language, and highlight state."""
        editor = TextEditor(language="JSON")

        editor.set_editor_settings(
            {
                "font_family": "Courier New",
                "font_size": 12,
                "language": "Python",
                "paper_color": "#ffffff",
                "default_color": "#000000",
                "caret_line_visible": True,
            }
        )

        assert editor.font().family() == "Courier New"
        assert editor.font().pointSize() == 12
        assert editor.language == "Python"
        assert editor.caret_line_visible is True

    def test_save_editor_settings_persists_to_yaml(self, tmp_path, monkeypatch):
        """Editor settings should be written to the ChiSurf settings YAML."""
        settings_file = tmp_path / "settings_chisurf.yaml"
        settings_file.write_text(
            "gui:\n  editor:\n    font_family: Courier New\n    font_size: 9\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(cs.core.settings, "chisurf_settings_file", settings_file)
        monkeypatch.setattr(
            cs.core.settings,
            "cs_settings",
            {"gui": {"editor": {"font_family": "Courier New", "font_size": 9}}},
        )
        monkeypatch.setattr(
            cs.core.settings,
            "gui",
            {"editor": {"font_family": "Courier New", "font_size": 9}},
        )

        assert save_editor_settings({"font_family": "Consolas", "font_size": 11})

        data = yaml.safe_load(settings_file.read_text(encoding="utf-8"))
        assert data["gui"]["editor"]["font_family"] == "Consolas"
        assert data["gui"]["editor"]["font_size"] == 11
        assert cs.core.settings.gui["editor"]["font_family"] == "Consolas"
        assert cs.core.settings.gui["editor"]["font_size"] == 11

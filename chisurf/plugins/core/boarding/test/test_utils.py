"""Tests for the boarding utils module."""

from __future__ import annotations

import pathlib

from chisurf.plugins.core.boarding import utils


class TestImportCheck:
    """Tests for the import_check function."""

    def test_existing_module(self) -> None:
        """Test checking for an existing module."""
        ok, msg = utils.import_check("os")
        assert ok is True
        assert msg == "ok"

    def test_nonexistent_module(self) -> None:
        """Test checking for a nonexistent module."""
        ok, msg = utils.import_check("nonexistent_module_xyz_123")
        assert ok is False
        assert "No module named" in msg

    def test_standard_library_module(self) -> None:
        """Test checking for a standard library module."""
        ok, msg = utils.import_check("pathlib")
        assert ok is True
        assert msg == "ok"


class TestHtmlCode:
    """Tests for the _html_code function."""

    def test_plain_text(self) -> None:
        """Test escaping plain text."""
        result = utils._html_code("hello")
        assert result == "<code>hello</code>"

    def test_html_special_chars(self) -> None:
        """Test escaping HTML special characters."""
        result = utils._html_code("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;" in result

    def test_non_string(self) -> None:
        """Test handling non-string input."""
        result = utils._html_code(123)
        assert "<code>123</code>" in result


class TestSettingsPaths:
    """Tests for the settings_paths function."""

    def test_returns_dict(self) -> None:
        """Test that settings_paths returns a dictionary."""
        result = utils.settings_paths()
        assert isinstance(result, dict)

    def test_contains_expected_keys(self) -> None:
        """Test that settings_paths contains expected keys."""
        result = utils.settings_paths()
        expected_keys = [
            "user_settings_dir",
            "settings_chisurf_yaml",
            "settings_colors_yaml",
            "anisotropy_corrections_json",
            "detector_setups_json",
            "styles_dir",
            "plugins_dir",
            "logs_dir",
        ]
        for key in expected_keys:
            assert key in result

    def test_values_are_paths(self) -> None:
        """Test that all values in settings_paths are pathlib.Path objects."""
        result = utils.settings_paths()
        for key, value in result.items():
            assert isinstance(value, pathlib.Path)


class TestBuildStatusHtml:
    """Tests for the build_status_html function."""

    def test_returns_html_string(self) -> None:
        """Test that build_status_html returns an HTML string."""
        result = utils.build_status_html()
        assert isinstance(result, str)
        assert "<h3>" in result
        assert "</table>" in result

    def test_contains_settings_status(self) -> None:
        """Test that the HTML contains settings status header."""
        result = utils.build_status_html()
        assert "Settings status" in result


class TestBuildDepsHtml:
    """Tests for the build_deps_html function."""

    def test_returns_html_string(self) -> None:
        """Test that build_deps_html returns an HTML string."""
        result = utils.build_deps_html()
        assert isinstance(result, str)
        assert "<h3>" in result
        assert "</table>" in result

    def test_contains_optional_dependencies(self) -> None:
        """Test that the HTML contains optional dependencies header."""
        result = utils.build_deps_html()
        assert "Optional dependencies" in result

"""Tests for the FCS convert CLI plugin."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from chisurf.plugins.fcs.fcs_convert.cli import (
    _format_list,
    _normalize,
    _SUPPORTED_INPUT_TYPES,
    _SUPPORTED_OUTPUT_TYPES,
    convert_fcs,
)


class TestNormalize:
    """Tests for the _normalize helper function."""

    def test_normalize_empty_string(self) -> None:
        """Test normalizing an empty string."""
        assert _normalize("") == ""

    def test_normalize_none(self) -> None:
        """Test normalizing None."""
        assert _normalize(None) == ""

    def test_normalize_whitespace(self) -> None:
        """Test normalizing whitespace."""
        assert _normalize("  ") == ""

    def test_normalize_mixed_case(self) -> None:
        """Test normalizing mixed case string."""
        assert _normalize("AlV") == "alv"

    def test_normalize_with_whitespace(self) -> None:
        """Test normalizing string with leading/trailing whitespace."""
        assert _normalize("  csv  ") == "csv"


class TestFormatList:
    """Tests for the _format_list helper function."""

    def test_format_empty_list(self) -> None:
        """Test formatting an empty list."""
        assert _format_list([]) == ""

    def test_format_single_item(self) -> None:
        """Test formatting a single-item list."""
        assert _format_list(["csv"]) == "csv"

    def test_format_multiple_items(self) -> None:
        """Test formatting a multi-item list."""
        result = _format_list(["csv", "alv", "yaml"])
        assert result == "alv, csv, yaml"

    def test_format_unsorted_input(self) -> None:
        """Test that the output is sorted regardless of input order."""
        result = _format_list(["yaml", "csv", "alv"])
        assert result == "alv, csv, yaml"


class TestConvertFcs:
    """Tests for the convert_fcs function."""

    def test_unsupported_input_type(self) -> None:
        """Test that unsupported input types raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.txt"
            output_file = Path(tmpdir) / "output.txt"
            input_file.write_text("dummy")

            with pytest.raises(ValueError, match="Unsupported input type"):
                convert_fcs(
                    input_filename=input_file,
                    input_type="unsupported",
                    output_filename=output_file,
                    output_type="yaml",
                )

    def test_unsupported_output_type(self) -> None:
        """Test that unsupported output types raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = Path(tmpdir) / "input.txt"
            output_file = Path(tmpdir) / "output.txt"
            input_file.write_text("dummy")

            with pytest.raises(ValueError, match="Unsupported output type"):
                convert_fcs(
                    input_filename=input_file,
                    input_type="csv",
                    output_filename=output_file,
                    output_type="unsupported",
                )

    def test_supported_types_are_lowercase(self) -> None:
        """Test that supported input/output types are lowercase."""
        for t in _SUPPORTED_INPUT_TYPES:
            assert t == t.lower()
        for t in _SUPPORTED_OUTPUT_TYPES:
            assert t == t.lower()

    def test_normalize_input_type(self) -> None:
        """Test that input types are normalized (case-insensitive)."""
        # This should work because _normalize converts to lowercase
        # and _SUPPORTED_INPUT_TYPES are all lowercase
        assert _normalize("CSV") == "csv"
        assert _normalize("CSV") in _SUPPORTED_INPUT_TYPES

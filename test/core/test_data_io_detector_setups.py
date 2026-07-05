from __future__ import annotations

"""Tests for chisurf.core.data_io.detector_setups (Qt-free)."""

import json
import pathlib

import pytest

from chisurf.core.data_io.detector_setups import (
    load_detector_setups,
    save_detector_setups,
)


class TestDetectorSetupsIO:

    def test_load_no_file_returns_empty(self, tmp_path: pathlib.Path):
        result = load_detector_setups(file_path=tmp_path / "nonexistent.json")
        assert result == {"setups": {}}

    def test_save_and_load_roundtrip(self, tmp_path: pathlib.Path):
        f = tmp_path / "setups.json"
        payload = {
            "setups": {
                "my_setup": {"detectors": {"SPAD": {"chs": [0, 1]}}},
            },
            "last_used": "my_setup",
        }
        save_detector_setups(payload, file_path=f)
        assert f.exists()
        result = load_detector_setups(file_path=f)
        assert result["setups"]["my_setup"] == payload["setups"]["my_setup"]
        assert result["last_used"] == "my_setup"

    def test_load_bad_json_returns_empty(self, tmp_path: pathlib.Path):
        f = tmp_path / "bad.json"
        f.write_text("not json")
        result = load_detector_setups(file_path=f)
        assert result == {"setups": {}}

    def test_save_creates_parent_dirs(self, tmp_path: pathlib.Path):
        f = tmp_path / "a" / "b" / "setups.json"
        payload = {"setups": {"test": {}}}
        save_detector_setups(payload, file_path=f)
        assert f.exists()
        with open(f) as fp:
            data = json.load(fp)
        assert data["setups"]["test"] == {}

    def test_no_gui_import(self):
        import ast
        import inspect
        import chisurf.core.data_io.detector_setups as mod
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.name if isinstance(node, ast.Import) else node.module or ""
                    assert "chisurf.gui" not in name, (
                        f"core.data_io.detector_setups must not import chisurf.gui "
                        f"(found: import {name})"
                    )

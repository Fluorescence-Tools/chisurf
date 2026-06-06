from __future__ import annotations

"""Tests for chisurf.server.services.projects."""

from unittest.mock import patch, MagicMock
import tempfile
import pathlib

import pytest

from chisurf.server.services.projects import (
    save_project,
    load_project,
    get_project_info,
)
from chisurf.server.session import SessionState


class TestProjectsService:

    def test_get_project_info_no_project(self):
        state = SessionState()
        result = get_project_info(state)
        assert result["ok"]
        assert result["project_path"] is None
        assert result["fit_count"] == 0
        assert result["dataset_count"] == 0

    @patch("chisurf.core.project.Project")
    def test_save_project(self, mock_project_cls):
        mock_project = MagicMock()
        mock_project.save = MagicMock(return_value=pathlib.Path("/tmp/test/project.json"))
        mock_project_cls.return_value = mock_project

        state = SessionState()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_project(state, target_path=tmpdir)
            assert result["ok"]
            assert "/project.json" in result.get("path", "")

    def test_load_project_not_found(self):
        state = SessionState()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_project(state, project_path=tmpdir)
            assert not result["ok"]
            assert "not found" in result["error"]

"""Integration tests for BurstSelectionClient via InProcessClient."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chisurf.core.plugin.client import InProcessClient
from chisurf.core.plugin.manifest import PluginManifest, load_manifest


class TestBurstSelectionClientConstruction:
    """BurstSelectionClient creates a local InProcessClient."""

    def test_creates_local_client(self):
        from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient

        client = BurstSelectionClient()
        assert client._client is not None
        assert isinstance(client._client, InProcessClient)

    def test_accepts_external_client(self):
        from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient

        mock_client = MagicMock()
        mock_client.call.return_value = {"result": {}}
        bc = BurstSelectionClient(client=mock_client)
        assert bc._client is mock_client

    def test_analyze_files(self):
        from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient

        mock_client = MagicMock()
        mock_client.call.return_value = {
            "result": {
                "files": ["test.spc"],
                "dataframes": {},
                "metadata": {"n_bursts": 0},
            }
        }
        bc = BurstSelectionClient(client=mock_client)
        result = bc.analyze_files([Path("test.spc")])
        assert result["files"] == ["test.spc"]
        mock_client.call.assert_called_once()

    def test_analyze_files_passes_windows_and_detectors(self):
        from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient

        mock_client = MagicMock()
        mock_client.call.return_value = {"result": {}}
        bc = BurstSelectionClient(client=mock_client)

        windows = {"prompt": [0, 2048]}
        detectors = {"green": {"chs": [0]}}
        bc.analyze_files([Path("t.spc")], windows=windows, detectors=detectors)

        call_args = mock_client.call.call_args
        assert call_args is not None
        method, params = call_args[0]
        assert method == "burst_selection.jobs.analyze_files"
        assert params.get("windows") == windows
        assert params.get("detectors") == detectors

    def test_analyze_files_skips_windows_when_none(self):
        from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient

        mock_client = MagicMock()
        mock_client.call.return_value = {"result": {}}
        bc = BurstSelectionClient(client=mock_client)
        bc.analyze_files([Path("t.spc")])

        call_args = mock_client.call.call_args
        params = call_args[0][1]
        assert "windows" not in params
        assert "detectors" not in params

    def test_inspect_bur(self):
        from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient

        mock_client = MagicMock()
        mock_client.call.return_value = {
            "result": {"path": "test.bur", "n_rows": 10, "columns": ["a", "b"]}
        }
        bc = BurstSelectionClient(client=mock_client)
        result = bc.inspect_bur(Path("test.bur"))
        assert result["path"] == "test.bur"

    def test_fit_gmm(self):
        from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient

        mock_client = MagicMock()
        mock_client.call.return_value = {
            "result": {"path": "test.bur", "features": [], "gmm": {}}
        }
        bc = BurstSelectionClient(client=mock_client)
        result = bc.fit_gmm(Path("test.bur"), {})
        assert "features" in result

    def test_load_diagnostics_returns_empty_on_no_file(self):
        from chisurf.plugins.burst.burst_selection.gui.client import BurstSelectionClient

        bc = BurstSelectionClient(client=MagicMock())
        result = bc.load_diagnostics(Path("/nonexistent/file.spc"), {})
        assert result == {}

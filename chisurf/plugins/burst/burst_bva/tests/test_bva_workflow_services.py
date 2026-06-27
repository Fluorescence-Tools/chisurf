"""Tests for BVA workflow RPC preparation."""

from __future__ import annotations


def test_prepare_workflow_uses_context_folder_and_channels(tmp_path) -> None:
    """BVA prepare resolves folder and derives donor/acceptor settings."""
    from chisurf.plugins.burst.burst_bva.backend.services import prepare_workflow_handler

    burst_folder = tmp_path / "burstwise"
    burst_folder.mkdir()
    response = prepare_workflow_handler(
        workflow_context={
            "burst_folder": str(burst_folder),
            "channel_settings": {
                "detectors": {
                    "green": {"chs": [8, 0], "micro_time_ranges": [[0, 100]]},
                    "red": {"chs": [9, 1], "micro_time_ranges": [[100, 200]]},
                },
                "tttr_reading": {"file_type": "SPC-130"},
            },
        }
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["analysis_folder"] == str(burst_folder)
    assert result["settings"]["donor_channels"] == [8, 0]
    assert result["settings"]["acceptor_channels"] == [9, 1]
    assert result["settings"]["donor_micro_time_ranges"] == [[0, 100]]
    assert result["settings"]["acceptor_micro_time_ranges"] == [[100, 200]]


def test_bva_services_register_workflow_prepare() -> None:
    """BVA register_services exposes workflow prepare RPC."""
    from chisurf.plugins.burst.burst_bva.backend.services import (
        METHOD_PREPARE_WORKFLOW,
        register_services,
    )

    calls: dict[str, object] = {}

    class Dispatcher:
        def register(self, name, handler):
            calls[name] = handler

    register_services(Dispatcher())
    assert METHOD_PREPARE_WORKFLOW in calls

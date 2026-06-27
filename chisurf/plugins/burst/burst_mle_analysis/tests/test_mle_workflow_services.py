"""Tests for Burst MLE workflow RPC preparation."""

from __future__ import annotations


def test_mle_prepare_resolves_bur_files_from_context(tmp_path) -> None:
    """MLE prepare resolves .bur files from workflow context burst folder."""
    from chisurf.plugins.burst.burst_mle_analysis.backend.services import prepare_workflow_handler

    bur_folder = tmp_path / "burstwise" / "bi4_bur"
    bur_folder.mkdir(parents=True)
    bur_file = bur_folder / "a.bur"
    bur_file.write_text("First File\tFirst Photon\tLast Photon\t\n\n")

    response = prepare_workflow_handler(
        workflow_context={
            "burst_folder": str(tmp_path / "burstwise"),
            "raw_files": [str(tmp_path / "a.spc")],
            "channel_settings": {"detectors": {"green": {"chs": [8]}}},
            "mfdb_artifacts": {"sidecar_artifacts": {"output_folder": "artifact-1"}},
        }
    )

    assert response["ok"] is True
    result = response["result"]
    assert result["bur_files"] == [str(bur_file)]
    assert result["raw_files"] == [str(tmp_path / "a.spc")]
    assert result["channel_settings"]["detectors"]["green"]["chs"] == [8]
    assert result["mfdb_artifacts"]["sidecar_artifacts"]["output_folder"] == "artifact-1"


def test_mle_services_register_workflow_prepare() -> None:
    """MLE register_services exposes workflow prepare RPC."""
    from chisurf.plugins.burst.burst_mle_analysis.backend.services import (
        METHOD_PREPARE_WORKFLOW,
        register_services,
    )

    calls: dict[str, object] = {}

    class Dispatcher:
        def register(self, name, handler):
            calls[name] = handler

    register_services(Dispatcher())
    assert METHOD_PREPARE_WORKFLOW in calls

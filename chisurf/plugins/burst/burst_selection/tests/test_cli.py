"""Tests for the Burst Selection Click CLI."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from chisurf.core import cli as core_cli
from chisurf.core.cli import _discover_plugin_metadata, _parse_cli_entrypoint
from chisurf.plugins.burst.burst_selection import cli_entrypoint
from chisurf.plugins.burst.burst_selection.api.models import (
    AnalysisSettings,
    BurstDetectionSettings,
    DeltaMacroTimeFilterSettings,
    PhotonFilterSettings,
)
from chisurf.plugins.burst.burst_selection.api.selection import analyze_file
from chisurf.plugins.burst.burst_selection.cli.main import cli as cli_module

DATA_FILE = Path(__file__).resolve().parent / "data" / "bh_spc132_sm_dna" / "m000.spc"
STREAM_CHANNELS = [0, 1, 8, 9]


def real_data_settings() -> AnalysisSettings:
    """Return deterministic settings for the bundled BH SPC example."""
    settings = AnalysisSettings()
    settings.photon_filter = PhotonFilterSettings(
        channels=STREAM_CHANNELS,
        filter_active=False,
        delta_macro_time_filter=DeltaMacroTimeFilterSettings(dT_min=0.0),
    )
    settings.burst_detection = BurstDetectionSettings(
        min_photons=20,
        photon_window=10,
        time_window=1e-3,
    )
    return settings


def test_inspect_command(tmp_path: Path) -> None:
    """The inspect command should summarize a real generated .bur file."""
    result = analyze_file(DATA_FILE, settings=real_data_settings(), output_dir=tmp_path)
    bur_path = Path(result.output_paths["bur"])
    result = CliRunner().invoke(cli_module, ["inspect", str(bur_path)])
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["n_rows"] == 9533


def test_fit_gmm_command(tmp_path: Path) -> None:
    """The fit-gmm command should fit features from a real generated .bur file."""
    result = analyze_file(DATA_FILE, settings=real_data_settings(), output_dir=tmp_path)
    bur_path = Path(result.output_paths["bur"])
    result = CliRunner().invoke(
        cli_module,
        ["fit-gmm", str(bur_path), "--settings", '{"covariance_type":"spherical","n_init":1}'],
    )
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["n_components"] == 1
    assert len(payload["labels"]) == 9533


def test_entrypoint_is_discoverable() -> None:
    """Plugin CLI metadata should be discoverable by the core CLI scanner."""
    metadata = list(_discover_plugin_metadata())
    burst_selection = [item for item in metadata if item["module_path"].endswith("burst_selection")][0]
    assert burst_selection["cli_entrypoint"] == cli_entrypoint


def test_entrypoint_has_cli_alias() -> None:
    """The plugin metadata should register the CLI with a stable alias."""
    alias, module_path, attr = _parse_cli_entrypoint(cli_entrypoint)
    assert (alias, module_path, attr) == (
        "burst-selection",
        "chisurf.plugins.burst.burst_selection.cli",
        "cli",
    )


def test_core_cli_forwards_to_burst_selection_analyze(tmp_path: Path) -> None:
    """The core CLI should forward subcommands to the Burst Selection CLI."""
    result = CliRunner().invoke(
        core_cli.cli,
        [
            "burst-selection",
            "analyze",
            str(DATA_FILE),
            "--output-dir",
            str(tmp_path),
            "--min-photons",
            "20",
        ],
    )
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["metadata"]["n_photons"] == 174438


def test_analyze_command_uses_shared_api(tmp_path: Path) -> None:
    """The analyze command should run the shared API request object on real data."""
    result = CliRunner().invoke(
        cli_module,
        [
            "analyze",
            str(DATA_FILE),
            "--output-dir",
            str(tmp_path),
            "--min-photons",
            "20",
        ],
    )
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["metadata"]["n_photons"] == 174438


def test_analyze_mfdb_registers_raw_sample_and_group(tmp_path: Path) -> None:
    """``analyze --mfdb`` registers raw+sample, burst tables, and a single
    output-folder group whose artifact resolves to the on-disk burst folder —
    the handoff Burst Selection -> ndXplorer relies on."""
    import shutil

    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.core.mfdb.provenance.result_registry import set_global_db

    # Copy the fixture so the co-located burst output folder lands in tmp.
    spc = tmp_path / "m000.spc"
    shutil.copy(DATA_FILE, spc)
    det_json = tmp_path / "det.json"
    det_json.write_text(
        json.dumps(
            {
                "green": {"chs": [0, 8], "micro_time_ranges": [[0, 4095]], "g_factor": 1, "l1": 0, "l2": 0},
                "red": {"chs": [1, 9], "micro_time_ranges": [[0, 4095]], "g_factor": 1, "l1": 0, "l2": 0},
            }
        )
    )
    db_path = tmp_path / "mfdb.sqlite"

    try:
        result = CliRunner().invoke(
            cli_module,
            [
                "analyze",
                str(spc),
                "--filetype", "SPC-130",
                "--detectors-json", str(det_json),
                "--min-photons", "20",
                "--mfdb",
                "--db", str(db_path),
                "--sample-name", "DNA burst sample",
                "--selected-setup", "BS",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["ok"] is True
        artifacts = payload["result"]["mfdb_artifacts"]
        assert artifacts["input_artifacts"], "raw input not registered"
        assert artifacts["burst_table_artifacts"], "burst table not registered"
        group_id = artifacts["sidecar_artifacts"]["output_folder"]

        # The group artifact resolves (as mfdb.datasets.open does) to the on-disk
        # burst folder co-located with the TTTR — what ndXplorer opens.
        db = MFDatabase(db_path)
        try:
            folder = db.open_dataset(group_id)
        finally:
            db.close()
        assert Path(folder).is_dir()
        bur_files = list((Path(folder) / "bi4_bur").glob("*.bur"))
        assert bur_files, f"no .bur files under {folder}"
        assert str(tmp_path) in folder, "burst folder not co-located with the TTTR copy"
    finally:
        set_global_db(None)


def test_contract_command_outputs_workflow_contract() -> None:
    """The CLI should expose the machine-readable workflow contract."""
    result = CliRunner().invoke(cli_module, ["contract"])
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["plugin_id"] == "burst_selection"
    assert "AnalyzeFiles" in payload["inputs"]
    assert "AnalysisResult" in payload["outputs"]
    assert "burst_selection.jobs.analyze_files" in payload["rpc_methods"]

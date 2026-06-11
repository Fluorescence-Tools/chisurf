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


def test_contract_command_outputs_workflow_contract() -> None:
    """The CLI should expose the machine-readable workflow contract."""
    result = CliRunner().invoke(cli_module, ["contract"])
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["plugin_id"] == "burst_selection"
    assert "AnalyzeFiles" in payload["inputs"]
    assert "AnalysisResult" in payload["outputs"]
    assert "burst_selection.jobs.analyze_files" in payload["rpc_methods"]

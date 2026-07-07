"""Command Line Interface for Burst Selection."""

from __future__ import annotations

import json
from typing import Any

import click
import pandas as pd

from ..api.contract import (
    analysis_request_from_payload,
    analysis_result_to_payload,
    contract_descriptor,
)
from ..api.features import extract_features, fit_gmm
from ..api.models import AnalysisSettings
from ..api.selection import analyze_request
from ..api.serialization import settings_from_dict


def load_json_file(path: str | None) -> dict[str, Any]:
    """Load a JSON file if a path is provided."""
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def create_settings(
    settings_file: str | None,
    min_photons: int,
    photon_window: int,
    time_window: float,
    n_ph_max: int,
    count_rate_window: float,
    output_formats: list[str],
) -> AnalysisSettings:
    """Create analysis settings from CLI options and an optional JSON file."""
    settings = settings_from_dict(load_json_file(settings_file) or {})
    settings.burst_detection.min_photons = min_photons
    settings.burst_detection.photon_window = photon_window
    settings.burst_detection.time_window = time_window
    settings.photon_filter.count_rate_filter.n_ph_max = n_ph_max
    settings.photon_filter.count_rate_filter.time_window = count_rate_window
    settings.output_formats = output_formats
    return settings


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show the version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Burst Selection CLI.

    Analyze TTTR files, inspect ``.bur`` files, fit GMMs, or serve the API over ZMQ.
    """
    if version:
        click.echo("Burst Selection CLI v1.0.0")
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--filetype", default=None, help="Explicit TTTR file type for tttrlib.")
@click.option("--settings-file", type=click.Path(exists=True), help="JSON file with analysis settings.")
@click.option("--output-dir", type=click.Path(file_okay=False), help="Directory for generated .bur files.")
@click.option("--format", "output_formats", multiple=True, default=("bur",), show_default=True, help="Output format: bur.")
@click.option("--min-photons", default=60, show_default=True, type=int, help="Minimum photons per burst.")
@click.option("--photon-window", default=10, show_default=True, type=int, help="Photon window size.")
@click.option("--time-window", default=1e-3, show_default=True, type=float, help="Burst time window in seconds.")
@click.option("--n-ph-max", default=60, show_default=True, type=int, help="Count-rate filter maximum photons.")
@click.option("--count-rate-window", default=1e-3, show_default=True, type=float, help="Count-rate filter window in seconds.")
@click.option("--windows-json", type=click.Path(exists=True), help="PIE windows JSON file.")
@click.option("--detectors-json", type=click.Path(exists=True), help="Detector definitions JSON file.")
@click.option(
    "--mfdb/--no-mfdb",
    "use_mfdb",
    default=False,
    show_default=True,
    help="Register the run in MFDB (raw inputs linked to a sample, burst tables, and the "
    "co-located output folder) so downstream tools such as ndXplorer can open it.",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="MFDB SQLite path for --mfdb. Defaults to the configured database.",
)
@click.option("--sample-id", default=None, help="Existing MFDB sample ID to link the run to.")
@click.option(
    "--sample-name",
    default=None,
    help="Sample name to find-or-create (idempotent) when --sample-id is not given.",
)
@click.option("--selected-setup", default=None, help="Detector setup label stored in MFDB metadata.")
@click.option(
    "--legacy-output/--no-legacy-output",
    "legacy_output",
    default=None,
    help="Write the legacy burstwise output folder next to the TTTR files "
    "(default: on when --mfdb so ndXplorer has a folder to open).",
)
def analyze(
    files: list[str],
    filetype: str | None,
    settings_file: str | None,
    output_dir: str | None,
    output_formats: list[str],
    min_photons: int,
    photon_window: int,
    time_window: float,
    n_ph_max: int,
    count_rate_window: float,
    windows_json: str | None,
    detectors_json: str | None,
    use_mfdb: bool,
    db_path: str | None,
    sample_id: str | None,
    sample_name: str | None,
    selected_setup: str | None,
    legacy_output: bool | None,
) -> None:
    """Analyze TTTR FILES with the shared Burst Selection API.

    With ``--mfdb`` the run is archived to MFDB: raw inputs are registered and
    linked to a sample (``raw+sample``), burst tables are stored, and the
    co-located burst output folder is registered as a single group whose
    artifact ID is reported in the output (consumed by ndXplorer).
    """
    if not files:
        click.echo("No files specified. Use --help for usage information.", err=True)
        return

    settings = create_settings(
        settings_file=settings_file,
        min_photons=min_photons,
        photon_window=photon_window,
        time_window=time_window,
        n_ph_max=n_ph_max,
        count_rate_window=count_rate_window,
        output_formats=list(output_formats),
    )

    if use_mfdb:
        _analyze_with_mfdb(
            files=list(files),
            filetype=filetype,
            windows=load_json_file(windows_json),
            detectors=load_json_file(detectors_json),
            settings=settings,
            db_path=db_path,
            sample_id=sample_id,
            sample_name=sample_name,
            selected_setup=selected_setup,
            legacy_output=legacy_output,
        )
        return

    request = analysis_request_from_payload(
        {
            "files": list(files),
            "filetype": filetype,
            "windows": load_json_file(windows_json),
            "detectors": load_json_file(detectors_json),
            "settings": settings,
            "output_dir": output_dir,
        }
    )
    result = analyze_request(request)
    click.echo(json.dumps(analysis_result_to_payload(result), indent=2, default=str))


def _analyze_with_mfdb(
    *,
    files: list[str],
    filetype: str | None,
    windows: dict[str, Any],
    detectors: dict[str, Any],
    settings: AnalysisSettings,
    db_path: str | None,
    sample_id: str | None,
    sample_name: str | None,
    selected_setup: str | None,
    legacy_output: bool | None,
) -> None:
    """Run analysis with MFDB registration (``raw+sample -> BS``).

    Reuses the backend ``analyze_files_handler`` (the same path the GUI and RPC
    use) so input registration, burst tables, and the output-folder group are
    registered identically. Prints the service response, including
    ``mfdb_artifacts`` with the artifact IDs ndXplorer can open.
    """
    from dataclasses import asdict

    from chisurf.core.mfdb.repository import MFDatabase
    from chisurf.core.mfdb.provenance.result_registry import set_global_db
    from chisurf.core.mfdb.samples.sample_manager import SampleDefinition, create_sample

    from ..backend.services import analyze_files_handler

    db = MFDatabase(db_path) if db_path else None
    if db is not None:
        set_global_db(db)

    resolved_sample_id = sample_id
    if not resolved_sample_id:
        if not sample_name:
            raise click.UsageError("--mfdb requires --sample-id or --sample-name.")
        if db is None:
            raise click.UsageError("--mfdb with --sample-name requires --db (or a configured database).")
        # create_sample is idempotent by name (find-or-create).
        resolved_sample_id = create_sample(db, SampleDefinition(name=sample_name))

    legacy = True if legacy_output is None else legacy_output
    response = analyze_files_handler(
        files=files,
        filetype=filetype,
        windows=windows or {},
        detectors=detectors or {},
        settings=asdict(settings),
        legacy_output=legacy,
        selected_setup=selected_setup,
        mfdb={
            "enabled": True,
            "sample_id": resolved_sample_id,
            "register_missing_inputs": True,
        },
    )
    click.echo(json.dumps(response, indent=2, default=str))


@cli.command("contract")
def contract_cmd() -> None:
    """Print the JSON workflow contract for node/RPC integrations."""
    click.echo(json.dumps(contract_descriptor(), indent=2, default=str))


@cli.command("inspect")
@click.argument("bur_file", type=click.Path(exists=True, dir_okay=False))
def inspect_cmd(bur_file: str) -> None:
    """Inspect a saved ``.bur`` file."""
    df = pd.read_csv(bur_file, sep="\t")
    features = extract_features([df])
    summary = {
        "path": bur_file,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "feature_columns": list(features.columns),
        "n_bursts": int(len(df)),
    }
    click.echo(json.dumps(summary, indent=2))


@cli.command("fit-gmm")
@click.argument("bur_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--settings-file", type=click.Path(exists=True), help="JSON file with GMM settings.")
@click.option("--settings", "settings_json", default=None, help="Inline JSON GMM settings.")
def fit_gmm_cmd(bur_file: str, settings_file: str | None, settings_json: str | None) -> None:
    """Fit a Gaussian mixture model to burst features from BUR_FILE."""
    df = pd.read_csv(bur_file, sep="\t")
    settings = settings_from_dict({"gmm": (load_json_file(settings_file) or json.loads(settings_json or "{}"))})
    features = extract_features([df])
    result = fit_gmm(features, settings.gmm)
    click.echo(json.dumps(result, indent=2, default=str))


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option("--cmd-port", default=8765, show_default=True, type=int, help="ZMQ command port.")
@click.option("--pub-port", default=8766, show_default=True, type=int, help="ZMQ PUB port.")
def serve(host: str, cmd_port: int, pub_port: int) -> None:
    """Serve Burst Selection methods over ZMQ."""
    from ..server.methods import serve

    serve(host=host, cmd_port=cmd_port, pub_port=pub_port)


if __name__ == "__main__":
    cli()

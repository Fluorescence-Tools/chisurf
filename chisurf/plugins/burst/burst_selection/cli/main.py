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
) -> None:
    """Analyze TTTR FILES with the shared Burst Selection API."""
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

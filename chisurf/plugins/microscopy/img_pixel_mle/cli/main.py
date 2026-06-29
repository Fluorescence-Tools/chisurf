"""CLI for Pixel-wise MLE lifetime analysis.

Usage:
    img-pixel-mle analyze --irf-file IRF.ptu FILE [FILE …]
    img-pixel-mle contract
"""

from __future__ import annotations

import json

import click


@click.group()
def cli() -> None:
    """Pixel-wise MLE lifetime analysis for TTTR imaging data."""


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--irf-file", "-i", required=True, type=click.Path(exists=True), help="Path to IRF PTU file.")
@click.option("--output-dir", "-o", default="", help="Output directory.")
@click.option("--channels-p", default="0", help="Parallel detector channels (comma-separated).")
@click.option("--channels-s", default="2", help="Perpendicular detector channels (comma-separated).")
@click.option("--mt-start", default=0, type=int, help="Micro-time start bin.")
@click.option("--mt-stop", default=256, type=int, help="Micro-time stop bin.")
@click.option("--mt-binning", default=1, type=int, help="Micro-time binning factor.")
@click.option("--irf-threshold", default=0.02, type=float, help="IRF threshold fraction.")
@click.option("--shift-sp", default=0.0, type=float, help="Parallel IRF sub-bin shift.")
@click.option("--shift-ss", default=0.0, type=float, help="Perpendicular IRF sub-bin shift.")
@click.option("--tau", default=1.0, type=float, help="Initial lifetime (ns).")
@click.option("--rho", default=1.0, type=float, help="Initial rotational correlation time.")
@click.option("--min-photons", default=10, type=int, help="Minimum photon count per pixel.")
@click.option("--twoi-star/--no-twoi-star", default=True, help="Enable twoI* flag.")
@click.option("--bifl-scatter/--no-bifl-scatter", default=False, help="Enable BIFL scatter flag.")
@click.option("--json", "json_output", is_flag=True, help="Print result as JSON.")
def analyze(
    files: tuple[str, ...],
    irf_file: str,
    output_dir: str,
    channels_p: str,
    channels_s: str,
    mt_start: int,
    mt_stop: int,
    mt_binning: int,
    irf_threshold: float,
    shift_sp: float,
    shift_ss: float,
    tau: float,
    rho: float,
    min_photons: int,
    twoi_star: bool,
    bifl_scatter: bool,
    json_output: bool,
) -> None:
    """Analyze TTTR FILES with pixel-wise MLE (headless)."""
    from ..api.models import PixelMleRequest, PixelMleSettings
    from ..backend.services import _handle_analyze

    det_p = [int(c.strip()) for c in channels_p.split(",")]
    det_s = [int(c.strip()) for c in channels_s.split(",")]

    settings = PixelMleSettings(
        detector_chs_p=det_p,
        detector_chs_s=det_s,
        micro_time_start=mt_start,
        micro_time_stop=mt_stop,
        micro_time_binning=mt_binning,
        irf_threshold=irf_threshold,
        shift_sp=shift_sp,
        shift_ss=shift_ss,
        tau=tau,
        rho=rho,
        min_photons=min_photons,
        twoi_star=twoi_star,
        bifl_scatter=bifl_scatter,
    )

    import dataclasses

    result = _handle_analyze({
        "files": list(files),
        "irf_file": irf_file,
        "output_dir": output_dir,
        "settings": dataclasses.asdict(settings),
    })

    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        if result.get("ok"):
            r = result["result"]
            click.echo(f"Processed: {r.get('processed_files', [])}")
            click.echo(f"Outputs:   {r.get('output_paths', [])}")
            for w in r.get("warnings", []):
                click.echo(f"WARNING: {w}", err=True)
        else:
            click.echo(f"ERROR: {result.get('error')}", err=True)


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Print as JSON.")
def contract(json_output: bool) -> None:
    """Print the RPC contract descriptor."""
    from ..api.contract import contract_descriptor

    desc = contract_descriptor()
    if json_output:
        click.echo(json.dumps(desc, indent=2))
    else:
        click.echo(f"Plugin: {desc['plugin_id']} v{desc['version']}")
        click.echo(f"Methods: {', '.join(desc['methods'])}")

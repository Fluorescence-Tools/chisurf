"""CLI for Molecule-wise MLE analysis.

Usage:
    sm-image-mle analyze --irf-file IRF.ptu FILE [FILE …]
    sm-image-mle contract
    sm-image-mle serve
"""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def cli() -> None:
    """Molecule-wise MLE lifetime analysis for TTTR imaging data."""


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--irf-file", "-i", required=True, type=click.Path(exists=True), help="Path to IRF PTU file.")
@click.option("--output-dir", "-o", default="", help="Override output directory.")
@click.option("--detector-chs", default="2,0", help="Detector channels (comma-separated, e.g. '2,0').")
@click.option("--micro-time-range", default="0,256", help="Micro-time range as 'start,stop'.")
@click.option("--micro-time-binning", default=32, type=int, help="Micro-time binning factor.")
@click.option("--normalize-counts", default=0, type=int, help="Normalization mode (0-3).")
@click.option("--threshold", default=-1.0, type=float, help="Threshold fraction (-1 = disabled).")
@click.option("--minlength", default=-1, type=int, help="Minimum histogram length (-1 = auto).")
@click.option("--shift-sp", default=0.0, type=float, help="Parallel IRF shift (bins).")
@click.option("--shift-ss", default=0.0, type=float, help="Perpendicular IRF shift (bins).")
@click.option("--irf-threshold-fraction", default=0.08, type=float, help="IRF threshold fraction.")
@click.option("--l1", default=0.04, type=float, help="Fit23 l1 parameter.")
@click.option("--l2", default=0.04, type=float, help="Fit23 l2 parameter.")
@click.option("--twoi-star/--no-twoi-star", default=True, help="Enable twoI* flag.")
@click.option("--bifl-scatter/--no-bifl-scatter", default=False, help="Enable BIFL scatter flag.")
@click.option("--seg-sigma", default=1.0, type=float, help="Gaussian smoothing sigma for segmentation.")
@click.option("--seg-threshold", default=-1.0, type=float, help="Segmentation threshold (-1 = Otsu).")
@click.option("--peak-footprint-size", default=6, type=int, help="Peak detection footprint size.")
@click.option("--json", "json_output", is_flag=True, help="Print result as JSON.")
def analyze(
    files: tuple[str, ...],
    irf_file: str,
    output_dir: str,
    detector_chs: str,
    micro_time_range: str,
    micro_time_binning: int,
    normalize_counts: int,
    threshold: float,
    minlength: int,
    shift_sp: float,
    shift_ss: float,
    irf_threshold_fraction: float,
    l1: float,
    l2: float,
    twoi_star: bool,
    bifl_scatter: bool,
    seg_sigma: float,
    seg_threshold: float,
    peak_footprint_size: int,
    json_output: bool,
) -> None:
    """Analyze PTU FILES with molecule-wise MLE."""
    from ..api.models import MoleculeMleRequest, MoleculeMleSettings
    from ..api.molecule_mle import analyze_request

    det_chs = [int(c.strip()) for c in detector_chs.split(",")]
    mtr = tuple(int(x.strip()) for x in micro_time_range.split(","))

    settings = MoleculeMleSettings(
        detector_chs=det_chs,
        micro_time_range=(mtr[0], mtr[1]),
        micro_time_binning=micro_time_binning,
        normalize_counts=normalize_counts,
        threshold=threshold,
        minlength=minlength,
        shift_sp=shift_sp,
        shift_ss=shift_ss,
        irf_threshold_fraction=irf_threshold_fraction,
        l1=l1,
        l2=l2,
        twoi_star=twoi_star,
        bifl_scatter=bifl_scatter,
        seg_sigma=seg_sigma,
        seg_threshold=seg_threshold,
        peak_footprint_size=peak_footprint_size,
    )
    request = MoleculeMleRequest(
        files=list(files),
        irf_file=irf_file,
        output_dir=output_dir,
        settings=settings,
    )

    click.echo(f"Analyzing {len(files)} file(s)…")
    result = analyze_request(request)

    if json_output:
        import dataclasses
        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        click.echo(f"Processed: {result.processed_files}")
        click.echo(f"Output TSVs: {result.output_paths}")
        if result.joint_tsv:
            click.echo(f"Joint TSV: {result.joint_tsv}")
        for w in result.warnings:
            click.echo(f"WARNING: {w}", err=True)


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


@cli.command()
@click.option("--host", default="127.0.0.1", help="ZMQ bind host.")
@click.option("--port", default=5555, type=int, help="ZMQ bind port.")
def serve(host: str, port: int) -> None:
    """Start the sm_image_mle RPC service (ZMQ/JSON-RPC)."""
    click.echo(f"Starting sm_image_mle service on {host}:{port} …")
    try:
        from chisurf.server.dispatcher import ServiceDispatcher
        from chisurf.server.session import SessionState
        from ..backend.services import register_services

        state = SessionState()
        dispatcher = ServiceDispatcher(state)
        register_services(dispatcher)
        click.echo("Services registered. Listening…")
        dispatcher.serve(host=host, port=port)
    except Exception as exc:
        click.echo(f"Failed to start service: {exc}", err=True)
        raise SystemExit(1) from exc

"""CLI entrypoint for H2MM burst analysis."""

from __future__ import annotations

import pathlib

import click

from chisurf.plugins.burst.burst_h2mm.api.models import H2mmSettings, StreamSettings
from chisurf.plugins.burst.burst_h2mm.backend.services import (
    run_analysis,
    write_result_tables,
)
from chisurf.plugins.burst.burst_h2mm.core.engines import ENGINES


@click.group(name="h2mm")
def cli():
    """Photon-by-photon Hidden Markov Model (H2MM) burst analysis."""


@cli.command()
@click.argument("analysis_folder", type=click.Path(exists=True))
@click.option("--file-type", default="SPC-130", help="tttrlib container name (or 'auto')")
@click.option("--pattern", default="*.bur", help="Glob for .bur files")
@click.option("--donor-channels", default="0,8", help="Comma-separated donor routing channels")
@click.option("--acceptor-channels", default="1,9", help="Comma-separated acceptor routing channels")
@click.option("--min-states", default=1, type=int, help="Smallest state count to scan")
@click.option("--max-states", default=3, type=int, help="Largest state count to scan")
@click.option("--criterion", default="bic", type=click.Choice(["bic", "icl"]))
@click.option("--engine", default="em", type=click.Choice(list(ENGINES)),
              help="Compute engine (exact EM vs fast/approximate variants)")
@click.option("--surrogate", "surrogate_path", type=click.Path(exists=True),
              help="Trained surrogate .pkl (for the surrogate engines)")
@click.option("--refine-iters", default=20, type=int, help="EM polish maps for surrogate-refine")
@click.option("--patience", default=None, type=int,
              help="Early-stop the state scan after the criterion rises (safe speed-up)")
@click.option("--restarts", default=2, type=int, help="Random restarts per state count")
@click.option("--max-iter", default=500, type=int, help="Max EM iterations per fit")
@click.option("--time-scale", default=1, type=int, help="Macro-time down-scaling")
@click.option("--min-photons", default=5, type=int, help="Min stream photons per burst")
@click.option("--no-photons", is_flag=True, help="Skip the per-photon (ndX) table output")
@click.option("--output", "-o", type=click.Path(), help="Output directory (default <folder>/h2mm)")
def compute(
    analysis_folder,
    file_type,
    pattern,
    donor_channels,
    acceptor_channels,
    min_states,
    max_states,
    criterion,
    engine,
    surrogate_path,
    refine_iters,
    patience,
    restarts,
    max_iter,
    time_scale,
    min_photons,
    no_photons,
    output,
):
    """Fit H2MM models to burst data in ANALYSIS_FOLDER."""
    try:
        donor = [int(x) for x in donor_channels.split(",") if x.strip()]
        acceptor = [int(x) for x in acceptor_channels.split(",") if x.strip()]
    except ValueError as exc:
        raise click.ClickException(f"Error parsing channels: {exc}")

    settings = H2mmSettings(
        streams=[
            StreamSettings("green", donor, []),
            StreamSettings("red", acceptor, []),
        ],
        min_states=min_states,
        max_states=max_states,
        criterion=criterion,
        n_restarts=restarts,
        max_iter=max_iter,
        time_scale=time_scale,
        min_photons=min_photons,
        file_type=file_type,
        engine=engine,
        refine_iters=refine_iters,
        patience=patience,
        surrogate_path=surrogate_path or "",
        write_photons=not no_photons,
    )

    click.echo(f"Reading bursts from {analysis_folder} (engine={engine}) ...")
    result, bundle = run_analysis(settings, analysis_folder=analysis_folder, pattern=pattern)

    click.echo(
        f"Analysed {result.n_bursts} bursts / {result.n_photons} photons; "
        f"selected {result.n_states} states by {criterion.upper()}."
    )
    for f in result.scan:
        click.echo(
            f"  states={f.n_states}  logL={f.loglik:.1f}  BIC={f.bic:.1f}  ICL={f.icl:.1f}"
        )
    click.echo(f"  FRET per state: {[round(x, 3) for x in result.fret]}")
    click.echo(f"  populations:    {[round(x, 3) for x in result.populations]}")

    out_dir = pathlib.Path(output) if output else pathlib.Path(analysis_folder) / "h2mm"
    write_result_tables(result, bundle, out_dir)
    for label, path in result.output_paths.items():
        click.echo(f"  wrote {label}: {path}")

"""CLI entrypoint for BVA analysis."""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import click

from chisurf.plugins.burst.burst_bva.api.models import BvaSettings
from chisurf.plugins.burst.burst_bva.core.computation import (
    read_burst_analysis,
    compute_bva,
    write_bv4_analysis,
)


@click.group(name="bva")
def cli():
    """Burst Variance Analysis."""


@cli.command()
@click.argument("analysis_folder", type=click.Path(exists=True, file_okay=False))
@click.option("--file-type", default="SPC-130", help="tttrlib container name")
@click.option("--pattern", default="bi4_bur", help="Glob pattern for burst data dirs")
@click.option("--donor-channels", default="0,8", help="Comma-separated donor routing channels")
@click.option("--acceptor-channels", default="1,9", help="Comma-separated acceptor routing channels")
@click.option("--donor-mtr-start", default=0, type=int)
@click.option("--donor-mtr-end", default=32768, type=int)
@click.option("--acceptor-mtr-start", default=0, type=int)
@click.option("--acceptor-mtr-end", default=32768, type=int)
@click.option("--window-length", default=0.01, type=float, help="Minimum window length (s)")
@click.option("--photons-per-slice", default=10, type=int, help="Photons per slice")
@click.option("--output", "-o", type=click.Path(), help="Output directory for bv4 files")
@click.option("--save-settings", is_flag=True, help="Save settings to bv4/bva_settings.json")
def compute(
    analysis_folder,
    file_type,
    pattern,
    donor_channels,
    acceptor_channels,
    donor_mtr_start,
    donor_mtr_end,
    acceptor_mtr_start,
    acceptor_mtr_end,
    window_length,
    photons_per_slice,
    output,
    save_settings,
):
    """Run BVA analysis on burst data in ANALYSIS_FOLDER."""
    try:
        donor_chs = [int(x.strip()) for x in donor_channels.split(",")]
        acceptor_chs = [int(x.strip()) for x in acceptor_channels.split(",")]
    except ValueError as e:
        click.echo(f"Error parsing channels: {e}", err=True)
        raise click.Abort()

    af = pathlib.Path(analysis_folder)
    out_dir = pathlib.Path(output) if output else af / "bv4"

    settings = BvaSettings(
        donor_channels=donor_chs,
        donor_micro_time_ranges=[(donor_mtr_start, donor_mtr_end)],
        acceptor_channels=acceptor_chs,
        acceptor_micro_time_ranges=[(acceptor_mtr_start, acceptor_mtr_end)],
        minimum_window_length=window_length,
        number_of_photons_per_slice=photons_per_slice,
        file_type=file_type,
    )

    click.echo(f"Reading burst data from {af} ...")
    df, tttrs = read_burst_analysis(af, file_type, pattern=pattern)
    click.echo(f"Found {len(df)} bursts across {len(tttrs)} TTTR file(s)")

    click.echo("Computing BVA ...")
    df_v = compute_bva(
        df, tttrs,
        donor_channels=settings.donor_channels,
        donor_micro_time_ranges=settings.donor_micro_time_ranges,
        acceptor_channels=settings.acceptor_channels,
        acceptor_micro_time_ranges=settings.acceptor_micro_time_ranges,
        minimum_window_length=settings.minimum_window_length,
        number_of_photons_per_slice=settings.number_of_photons_per_slice,
    )

    df_selected = df_v[df_v["Proximity Ratio Std"] > 0.0]
    click.echo(f"Valid bursts (Std > 0): {len(df_selected)} / {len(df_v)} total")

    click.echo(f"Writing BV4 files to {out_dir} ...")
    write_bv4_analysis(df_v, str(out_dir.parent))

    if save_settings:
        settings_path = out_dir / "bva_settings.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        from chisurf.plugins.burst.burst_bva.api.serialization import to_jsonable
        with open(settings_path, "w") as f:
            json.dump(to_jsonable(settings), f, indent=4)
        click.echo(f"Settings saved to {settings_path}")

    click.echo("Done")

"""Headless CLI for per-pixel mean micro-time."""

from __future__ import annotations

import click

from .. import core as _core


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--channel", "-c", multiple=True, type=int, default=(0,), help="Detector channel(s).")
@click.option("--min-photons", "-n", type=int, default=2, help="Minimum photons per pixel.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output HDF5 path.")
def cli(filename, channel, min_photons, output):
    """Compute a per-pixel mean-micro-time map from a TTTR imaging FILENAME."""
    result = _core.compute_mean_micro_time(
        filename, channels=tuple(channel), n_ph_min=min_photons
    )
    maps = result["maps"]
    ny, nx = result["shape"]
    import numpy as np

    mt = maps["mean_micro_time"]
    click.echo(f"{nx}x{ny} px | mean micro-time (ns) mean={np.nanmean(mt[mt > 0]):.3f}")
    if output:
        _core.add_mean_micro_time_to_hdf5(maps, output)
        click.echo(f"wrote {output}")


if __name__ == "__main__":
    cli()

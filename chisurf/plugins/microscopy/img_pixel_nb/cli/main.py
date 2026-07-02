"""Headless CLI for per-pixel Number & Brightness."""

from __future__ import annotations

import click

from .. import core as _core


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--channel", "-c", multiple=True, type=int, default=(0,), help="Detector channel(s).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output HDF5 path.")
def cli(filename, channel, output):
    """Compute per-pixel N&B maps from a TTTR imaging FILENAME."""
    result = _core.compute_nb(filename, channels=tuple(channel))
    maps = result["maps"]
    ny, nx = result["shape"]
    import numpy as np

    click.echo(f"{nx}x{ny} px | B mean={np.nanmean(maps['B']):.3f} | N mean={np.nanmean(maps['N']):.3f}")
    if output:
        _core.add_nb_to_hdf5(maps, output)
        click.echo(f"wrote {output}")


if __name__ == "__main__":
    cli()

"""Headless CLI for the per-pixel intensity imaging tool."""

from __future__ import annotations

import click

from .. import core as _core


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--channel", "-c", multiple=True, type=int, default=(0,), help="Detector channel(s).")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output imaging HDF5 path.")
def cli(filename, channel, output):
    """Compute the per-pixel intensity map and create a standard imaging HDF5."""
    import numpy as np

    from chisurf.core.fluorescence.imaging import maps_to_dataframe, write_imaging_hdf5

    result = _core.compute_intensity(filename, channels=tuple(channel))
    maps = result["maps"]
    ny, nx = result["shape"]
    click.echo(f"{nx}x{ny} px | total intensity={np.nansum(maps['intensity']):.0f}")
    if output:
        write_imaging_hdf5(maps_to_dataframe(maps), output, source=filename)
        click.echo(f"wrote {output}")


if __name__ == "__main__":
    cli()

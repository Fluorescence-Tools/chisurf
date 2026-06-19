from __future__ import annotations

import click
import numpy as np

from chisurf.core.fio import read_jordi, write_jordi

from ..api.models import IRFEstimationSettings
from ..core.estimation import estimate_irf


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--file", "file_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to a Jordi decay file.")
@click.option("--dt", default=1.0, show_default=True, type=float, help="Time per channel in ns.")
@click.option("--window-length", default=11, show_default=True, type=int, help="Savitzky-Golay filter window length.")
@click.option("--polyorder", default=3, show_default=True, type=int, help="Savitzky-Golay polynomial order.")
@click.option("--rl-iterations", default=500, show_default=True, type=int, help="Richardson-Lucy deconvolution iterations.")
@click.option("--regularization", default=3, show_default=True, type=int, help="Median filter regularization size.")
@click.option("--output", type=click.Path(dir_okay=False), help="Output path for estimated IRF.")
def cli(
    file_path: str,
    dt: float,
    window_length: int,
    polyorder: int,
    rl_iterations: int,
    regularization: int,
    output: str | None,
) -> None:
    """Estimate an IRF from a Jordi-format fluorescence decay file."""
    data = read_jordi(file_path)
    data = np.asarray(data, dtype=np.float32)
    intensity = data[: len(data) // 2] if len(data) % 2 == 0 else data

    settings = IRFEstimationSettings(
        window_length=window_length,
        polyorder=polyorder,
        rl_iterations=rl_iterations,
        regularization=regularization,
    )

    result = estimate_irf(intensity, dt, settings)

    click.echo(f"Estimated lifetime: {result.lifetime_ns:.4f} ns")
    click.echo(f"Decay rate: {result.decay_rate_ns:.6f} ns⁻¹")
    click.echo(f"Amplitude: {result.amplitude:.2f}")
    click.echo(f"Offset: {result.offset:.2f}")

    if output:
        write_jordi(output, np.array(result.irf), np.array(result.irf))
        click.echo(f"IRF saved to: {output}")


if __name__ == "__main__":
    cli()

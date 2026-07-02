"""CLI for the CLSM plugin.

Examples
--------
    csc clsm setups
    csc clsm info data.ptu --setup "Leica SP5" --channels 0,1
    csc clsm representation data.ptu --setup "Leica SP5" -c 0,1 -o img.npy
    csc clsm decay data.ptu --setup "Leica SP5" -c 0,1 --threshold 0.5 -o decay.txt
    csc clsm frc data.ptu --setup "Leica SP5" -c 0,1 -o frc.txt
    csc clsm contract --json
"""

from __future__ import annotations

import functools
import json
from typing import Any

import click


def _parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def setup_options(func):
    """Attach the shared acquisition-setup options to a command."""

    @click.option(
        "--setup", "setup_name", default=None, help="Setup preset name (see `clsm setups`)."
    )
    @click.option("--channels", "-c", default=None, help="Routing channels, e.g. '0,1'.")
    @click.option("--tttr-type", default=None, help="Override TTTR record type (PTU/HT3/SPC).")
    @click.option("--routine", default=None, help="Override CLSM reading routine.")
    @click.option("--frame-marker", default=None, help="Override frame markers, e.g. '4,6'.")
    @click.option("--line-start", type=int, default=None, help="Override line-start marker.")
    @click.option("--line-stop", type=int, default=None, help="Override line-stop marker.")
    @click.option("--event-marker", type=int, default=None, help="Override event-type marker.")
    @click.option("--pixel-per-line", type=int, default=None, help="Override pixels per line.")
    @click.option(
        "--no-auto-detect", is_flag=True, help="Do not read markers from the file header."
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _setup_kwargs(
    setup_name,
    channels,
    tttr_type,
    routine,
    frame_marker,
    line_start,
    line_stop,
    event_marker,
    pixel_per_line,
    no_auto_detect,
) -> dict[str, Any]:
    """Collect setup-related CLI options into API keyword arguments."""
    kwargs: dict[str, Any] = {
        "setup_name": setup_name,
        "channels": _parse_int_list(channels),
        "tttr_type": tttr_type,
        "routine": routine,
        "frame_marker": _parse_int_list(frame_marker),
        "line_start_marker": line_start,
        "line_stop_marker": line_stop,
        "event_type_marker": event_marker,
        "pixel_per_line": pixel_per_line,
        "auto_detect": not no_auto_detect,
    }
    return {k: v for k, v in kwargs.items() if v is not None}


def _emit(data: Any, json_output: bool, human) -> None:
    if json_output:
        click.echo(json.dumps(data, indent=2))
    else:
        human(data)


@click.group()
def cli() -> None:
    """CLSM-TTTR imaging: representations, pixel selections and decays."""


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Print as JSON.")
def setups(json_output: bool) -> None:
    """List the built-in acquisition-setup presets."""
    from ..client import ClsmClient

    data = ClsmClient().setups()

    def _human(d):
        for name, preset in d.items():
            click.echo(
                f"{name}: {preset.get('tttr_type')} / {preset.get('routine')} "
                f"frames={preset.get('frame_marker')}"
            )

    _emit(data, json_output, _human)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@setup_options
@click.option("--json", "json_output", is_flag=True, help="Print as JSON.")
def info(file, json_output, **setup_opts) -> None:
    """Report image dimensions and the resolved setup for FILE."""
    from ..client import ClsmClient

    data = ClsmClient().info(file, **_setup_kwargs(**setup_opts))

    def _human(d):
        click.echo(f"frames={d['n_frames']} lines={d['n_lines']} pixel={d['n_pixel']}")
        click.echo(
            f"photons={d['n_photons']} micro_time_res={d['micro_time_resolution_ns']:.4f} ns"
        )

    _emit(data, json_output, _human)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--image-type",
    default="Intensity",
    help="'Intensity', 'Mean micro time' or 'Intensity, Mean micro time'.",
)
@click.option("--n-ph-min", default=1, type=int, help="Min photons/pixel for mean micro time.")
@click.option("--frame-mode", default="sum", type=click.Choice(["sum", "mean", "frame"]))
@click.option("--frame-idx", default=0, type=int, help="Frame index for --frame-mode frame.")
@click.option("--output", "-o", default=None, help="Write .npy (stack) or .png/.tif (frame).")
@setup_options
@click.option("--json", "json_output", is_flag=True, help="Print as JSON.")
def representation(
    file, image_type, n_ph_min, frame_mode, frame_idx, output, json_output, **setup_opts
) -> None:
    """Compute an image representation for FILE."""
    from ..client import ClsmClient

    data = ClsmClient().representation(
        file,
        image_type=image_type,
        n_ph_min=n_ph_min,
        frame_mode=frame_mode,
        frame_idx=frame_idx,
        output_path=output,
        **_setup_kwargs(**setup_opts),
    )

    def _human(d):
        click.echo(
            f"{d['image_type']}: {d['n_frames']}x{d['n_lines']}x{d['n_pixel']} "
            f"total_intensity={d['total_intensity']:.0f}"
        )
        if d["output_path"]:
            click.echo(f"saved: {d['output_path']}")

    _emit(data, json_output, _human)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--mask",
    "mask_path",
    default=None,
    type=click.Path(exists=True),
    help="Selection mask image (tif/png). Overrides --threshold.",
)
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Select pixels above this fraction of the image max.",
)
@click.option("--image-type", default="Intensity", help="Representation used for --threshold.")
@click.option("--n-ph-min", default=1, type=int)
@click.option("--frame-mode", default="sum", type=click.Choice(["sum", "mean", "frame"]))
@click.option("--frame-idx", default=0, type=int)
@click.option("--tac-coarsening", default=1, type=int, help="Micro-time binning factor.")
@click.option("--stack/--no-stack", "stack_frames", default=True, help="Sum the decay over frames.")
@click.option("--output", "-o", default=None, help="Write a t/y/ey text file.")
@setup_options
@click.option("--json", "json_output", is_flag=True, help="Print as JSON.")
def decay(
    file,
    mask_path,
    threshold,
    image_type,
    n_ph_min,
    frame_mode,
    frame_idx,
    tac_coarsening,
    stack_frames,
    output,
    json_output,
    **setup_opts,
) -> None:
    """Extract a fluorescence-decay histogram from a pixel selection in FILE."""
    from ..client import ClsmClient

    data = ClsmClient().decay(
        file,
        mask_path=mask_path,
        threshold=threshold,
        image_type=image_type,
        n_ph_min=n_ph_min,
        frame_mode=frame_mode,
        frame_idx=frame_idx,
        tac_coarsening=tac_coarsening,
        stack_frames=stack_frames,
        output_path=output,
        **_setup_kwargs(**setup_opts),
    )

    def _human(d):
        click.echo(f"decay: {len(d['counts'])} bins, {d['n_photons']} photons")
        if d["output_path"]:
            click.echo(f"saved: {d['output_path']}")

    _emit(data, json_output, _human)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--image-type", default="Intensity")
@click.option("--n-ph-min", default=1, type=int)
@click.option("--frame-mode", default="sum", type=click.Choice(["sum", "mean", "frame"]))
@click.option("--frame-idx", default=0, type=int)
@click.option("--bin-width", default=2.0, type=float, help="FRC ring width (Fourier pixels).")
@click.option("--output", "-o", default=None, help="Write a frequency/correlation text file.")
@setup_options
@click.option("--json", "json_output", is_flag=True, help="Print as JSON.")
def frc(
    file, image_type, n_ph_min, frame_mode, frame_idx, bin_width, output, json_output, **setup_opts
) -> None:
    """Compute the Fourier Ring Correlation for FILE's image."""
    from ..client import ClsmClient

    data = ClsmClient().frc(
        file,
        image_type=image_type,
        n_ph_min=n_ph_min,
        frame_mode=frame_mode,
        frame_idx=frame_idx,
        bin_width=bin_width,
        output_path=output,
        **_setup_kwargs(**setup_opts),
    )

    def _human(d):
        click.echo(f"frc: {len(d['correlation'])} rings")
        if d["output_path"]:
            click.echo(f"saved: {d['output_path']}")

    _emit(data, json_output, _human)


@cli.command()
@click.option("--json", "json_output", is_flag=True, help="Print as JSON.")
def contract(json_output: bool) -> None:
    """Print the RPC contract descriptor."""
    from ..client import ClsmClient

    data = ClsmClient().contract()

    def _human(d):
        click.echo(f"Plugin: {d['plugin_id']} v{d['version']}")
        click.echo(f"Methods: {', '.join(d['methods'])}")

    _emit(data, json_output, _human)

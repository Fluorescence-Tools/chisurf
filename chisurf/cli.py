#!/usr/bin/env python

from __future__ import annotations

import os
import sys
from typing import Optional

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="chisurf", prog_name="csc")
def cli() -> None:
    """Unified command-line interface for chisurf tools.

    This entry point groups together various small utilities that were
    previously exposed as separate console scripts (e.g. csc_fcs_convert).

    Example usage::

        csc fcs-convert --help
        csc fcs-convert -if in.fcs -it pq.dat -of out.fcs -ot pycorrfit
    """


@cli.command(name="fcs-convert")
@click.option("-if", "--input-filename", required=True, type=click.Path(exists=True, dir_okay=False), help="Input FCS filename.")
@click.option("-it", "--input-type", required=True, type=str, help="Input file type (kristine, alv, mat, confocor3, pycorrfit, csv, pq.dat).")
@click.option("-of", "--output-filename", required=True, type=str, help="Output FCS filename.")
@click.option("-ot", "--output-type", required=True, type=str, help="Output file type (kristine, alv, china-mat, confocor3, pycorrfit, pq.dat).")
@click.option("-s", "--skiprows", default=0, type=int, show_default=True, help="Number of rows to skip in CSV input.")
@click.option("-e", "--use-header", is_flag=True, default=False, show_default=True, help="Use CSV header row for column names.")
def fcs_convert(
    input_filename: str,
    input_type: str,
    output_filename: str,
    output_type: str,
    skiprows: int,
    use_header: bool,
) -> None:
    """Convert FCS files using chisurf.cmd_tools.fcs_convert."""

    # Import lazily so that importing chisurf.cli is cheap and has minimal
    # side-effects when used from other tools.
    try:
        from chisurf.cmd_tools import fcs_convert as _fcs_convert
        import argparse
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(f"Failed to import chisurf.cmd_tools.fcs_convert: {exc}")

    ns = argparse.Namespace(
        input_filename=input_filename,
        input_type=input_type,
        output_filename=output_filename,
        output_type=output_type,
        skiprows=skiprows,
        use_header=use_header,
    )
    _fcs_convert.main(args=ns)


@cli.command(name="tttr-decay-hist")
@click.option("-c", "--channel", default=1, show_default=True, type=int, help="Detection channel number.")
@click.option("--coarse", default=2, show_default=True, type=int, help="Divider for the micro time (binning factor).")
@click.option(
    "-t",
    "--file-type",
    "file_type",
    required=True,
    type=str,
    help=(
        "TTTR file type (e.g. HT3, PTU, SPC-130, SPC-600_256, "
        "SPC-600_4096, PHOTON-HDF5)."
    ),
)
@click.option(
    "-i",
    "--input",
    "filename",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="TTTR input filename.",
)
@click.option(
    "-o",
    "--output",
    "output",
    type=str,
    default=None,
    help="Output CSV filename (default: <input>.csv).",
)
def tttr_decay_hist(
    channel: int,
    coarse: int,
    file_type: str,
    filename: str,
    output: str | None,
) -> None:
    """Compute a decay histogram from TTTR data.

    This is a click-based version of chisurf.cmd_tools.tttr_decay_histogram.
    """

    try:
        import numpy as np  # type: ignore[import]
        import tttrlib  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(f"Failed to import tttrlib/numpy: {exc}")

    if output:
        output_path = output
    else:
        root, _ext = os.path.splitext(os.path.abspath(filename))
        output_path = root + ".csv"

    click.echo("Make decay histogram from TTTR data")
    click.echo("===================================")
    click.echo(f"\tFilename: {filename}")
    click.echo(f"\tFile type: {file_type}")
    click.echo(f"\tCoarse :\t{coarse}")
    click.echo(f"\tOutput file: {output_path}")

    data = tttrlib.TTTR(filename, file_type)

    channel_selection = data.get_selection_by_channel(np.array([channel]))
    micro_time = data.get_micro_time()
    mt_sel = micro_time[channel_selection]
    counts = np.bincount(mt_sel // coarse)
    header = data.get_header()
    dt = header.micro_time_resolution
    x_axis = np.arange(counts.shape[0]) * dt * coarse

    np.savetxt(fname=output_path, X=np.vstack([x_axis, counts]).T)


@cli.command(
    name="count-rate",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def count_rate(ctx: click.Context) -> None:
    """Forward to the Count Rate Analysis CLI.

    All arguments after ``count-rate`` are passed through to
    :mod:`chisurf.plugins.count_rate_analysis.cli`.
    """

    try:
        from chisurf.plugins.count_rate_analysis import cli as _cr_cli
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(f"Failed to import count rate CLI: {exc}")

    argv = list(ctx.args)
    if not argv:
        argv = ["--help"]

    try:
        _cr_cli.cli.main(args=argv, standalone_mode=True)
    except SystemExit as exc:  # normal click exit path from nested CLI
        code = int(exc.code or 0)
        if code != 0:
            raise click.ClickException(f"count-rate exited with status {code}")

@cli.command(
    name="tttr-correlate",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def tttr_correlate(ctx: click.Context) -> None:
    """Forward to ``chisurf.cmd_tools.tttr_correlate``."""

    try:
        from chisurf.cmd_tools import tttr_correlate as _tttr_correlate
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Failed to import chisurf.cmd_tools.tttr_correlate: {exc}"
        )

    argv = ["tttr_correlate", *ctx.args]
    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        _tttr_correlate.main()
    finally:
        sys.argv = old_argv


@cli.command(
    name="mrc2bvox",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def mrc2bvox(ctx: click.Context) -> None:
    """Forward to ``chisurf.cmd_tools.mrc2bvox``."""

    try:
        from chisurf.cmd_tools import mrc2bvox as _mrc2bvox
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Failed to import chisurf.cmd_tools.mrc2bvox: {exc}"
        )

    argv = ["mrc2bvox", *ctx.args]
    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        _mrc2bvox.main()
    finally:
        sys.argv = old_argv


@cli.command(
    name="protein-mc-fret",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def protein_mc_fret(ctx: click.Context) -> None:
    """Forward to ``chisurf.cmd_tools.protein_mc_fret``."""

    try:
        from chisurf.cmd_tools import protein_mc_fret as _protein_mc_fret
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Failed to import chisurf.cmd_tools.protein_mc_fret: {exc}"
        )

    argv = ["protein_mc_fret", *ctx.args]
    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        _protein_mc_fret.main()
    finally:
        sys.argv = old_argv


@cli.command(
    name="sm-image-mle",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def sm_image_mle(ctx: click.Context) -> None:
    """Forward to the sm_image_mle CLI."""

    try:
        from chisurf.plugins.sm_image_mle import sm_image_mle as _sm_image_mle
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Failed to import sm_image_mle CLI: {exc}"
        )

    argv = list(ctx.args)
    if not argv:
        argv = ["--help"]

    try:
        _sm_image_mle.cli.main(args=argv, standalone_mode=True)
    except SystemExit as exc:  # normal click exit path from nested CLI
        code = int(exc.code or 0)
        if code != 0:
            raise click.ClickException(f"sm-image-mle exited with status {code}")


@cli.command(
    name="microtime-histogram",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def microtime_histogram(ctx: click.Context) -> None:
    """Launch the microtime_histogram helper."""

    try:
        from chisurf.plugins.microtime_histogram.__main__ import (  # type: ignore[import]
            main as _microtime_main,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Failed to import microtime_histogram entry point: {exc}"
        )

    argv = ["microtime_histogram", *ctx.args]
    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        _microtime_main()
    finally:
        sys.argv = old_argv


@cli.command(
    name="burst-background",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def burst_background(ctx: click.Context) -> None:
    """Forward to the Burst Background Estimation CLI."""

    try:
        from chisurf.plugins.burst_background import cli as _bb_cli
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(
            f"Failed to import burst background CLI: {exc}"
        )

    argv = list(ctx.args)
    if not argv:
        argv = ["--help"]

    try:
        _bb_cli.cli.main(args=argv, standalone_mode=True)
    except SystemExit as exc:  # normal click exit path from nested CLI
        code = int(exc.code or 0)
        if code != 0:
            raise click.ClickException(f"burst-background exited with status {code}")


def main(argv: Optional[list[str]] = None) -> int:
    """Entry-point compatible wrapper.

    This allows calling chisurf.cli:main as well as chisurf.cli:cli.
    """

    if argv is None:
        argv = sys.argv[1:]
    try:
        cli.main(args=argv, standalone_mode=True)
    except SystemExit as exc:  # normal click exit path
        return exc.code
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

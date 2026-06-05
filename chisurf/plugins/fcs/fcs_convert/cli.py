"""Click-based CLI for converting FCS files between supported formats."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import click

from chisurf.core.fio.fluorescence import fcs as fcs_io

# Readers exposed by chisurf.core.fio.fluorescence.fcs.read_fcs plus CSV helper
_SUPPORTED_INPUT_TYPES: tuple[str, ...] = (
    "alv",
    "china-mat",
    "confocor3",
    "csv",
    "kristine",
    "pq.dat",
    "pqres",
    "pycorrfit",
    "ries-mat",
    "sin",
    "yaml",
)

# Writers supported by chisurf.core.fio.fluorescence.fcs.write_fcs
_SUPPORTED_OUTPUT_TYPES: tuple[str, ...] = (
    "kristine",
    "yaml",
)


def _normalize(value: str) -> str:
    return str(value or "").strip().lower()


def _format_list(values: Iterable[str]) -> str:
    return ", ".join(sorted(values))


def convert_fcs(
    *,
    input_filename: str | os.PathLike[str],
    input_type: str,
    output_filename: str | os.PathLike[str],
    output_type: str,
    skiprows: int = 0,
    use_header: bool = False,
    verbose: bool = False,
) -> str:
    """Convert an FCS file from one format into another.

    Parameters
    ----------
    input_filename:
        Path to the source file.
    input_type:
        Reader key understood by :func:`chisurf.core.fio.fluorescence.fcs.read_fcs`.
    output_filename:
        Destination path for the converted data.
    output_type:
        Writer key supported by :func:`chisurf.core.fio.fluorescence.fcs.write_fcs`.
    skiprows:
        Number of rows to skip when reading CSV input.
    use_header:
        Whether to treat the first CSV row as header labels.
    verbose:
        Forwarded to ``read_fcs`` and ``write_fcs`` for optional logging.

    Returns
    -------
    str
        The path to the written file (stringified).
    """

    reader = _normalize(input_type)
    writer = _normalize(output_type)

    if reader not in _SUPPORTED_INPUT_TYPES:
        raise ValueError(
            f"Unsupported input type '{input_type}'. Supported values: {_format_list(_SUPPORTED_INPUT_TYPES)}"
        )
    if writer not in _SUPPORTED_OUTPUT_TYPES:
        raise ValueError(
            f"Unsupported output type '{output_type}'. Supported values: {_format_list(_SUPPORTED_OUTPUT_TYPES)}"
        )

    src_path = os.fspath(input_filename)
    dst_path = os.fspath(output_filename)

    data = fcs_io.read_fcs(
        filename=src_path,
        reader_name=reader,
        skiprows=skiprows,
        use_header=use_header,
        verbose=verbose,
    )
    fcs_io.write_fcs(
        data=data,
        filename=dst_path,
        file_type=writer,
        verbose=verbose,
    )

    return dst_path


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i",
    "--input-filename",
    type=click.Path(path_type=Path, dir_okay=False, exists=True),
    required=True,
    help="Input FCS filename.",
)
@click.option(
    "-it",
    "--input-type",
    required=True,
    type=click.Choice(sorted(_SUPPORTED_INPUT_TYPES)),
    help="Input file type / reader.",
)
@click.option(
    "-o",
    "--output-filename",
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    required=True,
    help="Destination filename for the converted data.",
)
@click.option(
    "-ot",
    "--output-type",
    required=True,
    type=click.Choice(sorted(_SUPPORTED_OUTPUT_TYPES)),
    help="Output file type / writer.",
)
@click.option(
    "-s",
    "--skiprows",
    type=int,
    default=0,
    show_default=True,
    help="Rows to skip when reading CSV input.",
)
@click.option(
    "--use-header/--no-header",
    default=False,
    show_default=True,
    help="Interpret the first CSV row as column headers.",
)
@click.option(
    "-v/--verbose",
    "verbose",
    default=False,
    show_default=True,
    help="Emit verbose logging from the reader/writer.",
)
def cli(
    input_filename: Path,
    input_type: str,
    output_filename: Path,
    output_type: str,
    skiprows: int,
    use_header: bool,
    verbose: bool,
) -> None:
    """Convert FCS files between supported formats."""

    click.echo("Converting FCS data...")
    click.echo(f"  Input : {input_filename} ({input_type})")
    click.echo(f"  Output: {output_filename} ({output_type})")

    try:
        convert_fcs(
            input_filename=input_filename,
            input_type=input_type,
            output_filename=output_filename,
            output_type=output_type,
            skiprows=skiprows,
            use_header=use_header,
            verbose=verbose,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - unexpected error paths
        raise click.ClickException(f"Failed to convert FCS data: {exc}") from exc

    click.echo("Conversion completed successfully.")


__all__ = ["cli", "convert_fcs"]

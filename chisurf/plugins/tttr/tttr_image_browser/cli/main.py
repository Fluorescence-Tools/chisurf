"""CLI commands for TTTR Image Browser."""

from __future__ import annotations

import json
from pathlib import Path

import click

from chisurf.plugins.tttr.tttr_image_browser.api.contract import contract_descriptor
from chisurf.plugins.tttr.tttr_image_browser.api.io import list_files
from chisurf.plugins.tttr.tttr_image_browser.core.image import load_image, save_tiff_stacks


@click.group(name="tttr-image-browser")
def cli():
    """Browse and export TTTR images."""


@cli.command("list")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--recursive", is_flag=True, help="Include subfolders.")
def list_cmd(folder: Path, recursive: bool):
    """List TTTR image files in a folder."""
    rows = list_files(str(folder), recursive=recursive)
    for row in rows:
        click.echo(f"{row['name']}\t{row['size_text']}\t★{row['rating']}")


@cli.command("load")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--max-side", default=512, show_default=True, type=int, help="Max dimensions of the mosaic.")
def load_cmd(path: Path, max_side: int):
    """Load and print image mosaic metadata."""
    data = load_image(str(path), max_side=max_side)
    if data is None:
        click.echo(f"Failed to load image from {path}", err=True)
        return
    mosaic = data["mosaic"]
    shape = [len(mosaic), len(mosaic[0]) if mosaic else 0]
    click.echo(
        json.dumps(
            {
                "path": str(path),
                "shape": shape,
                "labels": data["labels"],
                "cols": data["cols"],
                "rows": data["rows"],
            },
            indent=2,
        )
    )


@cli.command("export-tiff")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
def export_tiff_cmd(paths: tuple[Path, ...], output_dir: Path):
    """Export TTTR image stacks to TIFF files."""
    result = save_tiff_stacks([str(path) for path in paths], str(output_dir))
    for path in result["paths"]:
        click.echo(path)


@cli.command("contract")
def contract():
    """Print the RPC contract."""
    click.echo(json.dumps(contract_descriptor(), indent=2))

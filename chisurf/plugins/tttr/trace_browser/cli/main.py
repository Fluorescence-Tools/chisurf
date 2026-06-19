"""CLI commands for Trace Browser."""

from __future__ import annotations

import json
from pathlib import Path

import click

from chisurf.plugins.tttr.trace_browser.api.contract import contract_descriptor
from chisurf.plugins.tttr.trace_browser.api.io import export_csv, list_files
from chisurf.plugins.tttr.trace_browser.core.trace import load_trace


@click.group(name="trace-browser")
def cli():
    """Browse and export TTTR traces."""


@cli.command("list")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--recursive", is_flag=True, help="Include subfolders.")
def list(folder: Path, recursive: bool):
    """List trace files in a folder."""
    rows = list_files(str(folder), recursive=recursive)
    for row in rows:
        click.echo(f"{row['name']}\t{row['size_text']}\t★{row['rating']}")


@cli.command("load")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--window-ms", default=10.0, show_default=True, type=float, help="Trace bin width in ms.")
def load(path: Path, window_ms: float):
    """Load and print binned trace metadata."""
    data = load_trace(str(path), time_window_ms=window_ms)
    counts = data["counts"]
    shape = [len(counts), len(counts[0]) if counts else 0]
    click.echo(json.dumps({"path": str(path), "shape": shape, "labels": data["labels"]}, indent=2))


@cli.command("export-csv")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", "output_dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--window-ms", default=10.0, show_default=True, type=float)
def export(paths, output_dir: Path, window_ms: float):
    """Export traces to CSV files."""
    result = export_csv([str(path) for path in paths], str(output_dir), time_window_ms=window_ms)
    for path in result["paths"]:
        click.echo(path)


@cli.command("contract")
def contract():
    """Print the RPC contract."""
    click.echo(json.dumps(contract_descriptor(), indent=2))

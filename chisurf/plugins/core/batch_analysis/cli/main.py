"""CLI for Batch-Analysis.

Two commands, both backed by the Qt-free
:mod:`chisurf.plugins.core.batch_analysis.core.runner`:

* ``run`` — apply a template fit (already present in the ChiSurf session) to a
  list of files and write the consolidated CSV (plus DOCX/ZIP). Requires an
  in-process fitting client, so run it from ``csc`` with a project loaded.
* ``report`` — regenerate a DOCX report from an existing results CSV. Fully
  headless (no fitting session needed).
"""

from __future__ import annotations

import csv as _csv
import os

import click

from ..core import runner


@click.group()
def cli():
    """Batch analysis: apply one template fit to many files."""


@cli.command("run")
@click.option(
    "--file",
    "files",
    multiple=True,
    type=click.Path(exists=True),
    help="A file to fit (repeatable).",
)
@click.option("--fit-index", default=0, type=int, help="Index of the template fit.")
@click.option("--output", "-o", default=None, help="Output CSV path.")
@click.option("--list-fits", is_flag=True, help="List available fits and exit.")
def run(files, fit_index, output, list_fits):
    """Run the template fit over FILES using the in-process fitting client."""
    from chisurf.gui.widgets.fitting.fitting_client import get_fitting_client

    fit_client = get_fitting_client()
    if fit_client is None:
        raise click.ClickException("No fitting client — run this from a ChiSurf session (csc).")

    fit_objects = fit_client.get_fit_objects()
    if list_fits:
        for i, f in enumerate(fit_objects):
            click.echo(f"[{i}] {f.name}")
        return
    if not files:
        raise click.ClickException("No --file given.")
    if not output:
        raise click.ClickException("No --output given.")

    items = runner.build_queue([], list(files))
    click.echo(f"Running fit [{fit_index}] over {len(items)} file(s)…")
    results = runner.run_batch(
        fit_index,
        items,
        fit_client=fit_client,
        on_progress=lambda i, n, name: click.echo(f"  {i}/{n} {os.path.basename(name)}"),
    )
    results.write_csv(output)
    click.echo(f"Wrote {output} ({len(results.rows)} rows).")


@cli.command("report")
@click.argument("results_csv", type=click.Path(exists=True))
@click.option(
    "--screenshots",
    default=None,
    type=click.Path(exists=True),
    help="Directory of per-run screenshots (matched by group key).",
)
@click.option("--docx", "docx_out", default=None, help="Output DOCX path.")
def report(results_csv, screenshots, docx_out):
    """Regenerate a DOCX report from an existing results CSV (headless)."""
    with open(results_csv, newline="") as fh:
        rows = list(_csv.DictReader(fh))
    # rebuild GroupKey + processing order from the CSV
    for r in rows:
        r.setdefault("GroupKey", runner.norm_key(r.get("Filename", "")))
    file_order = list(dict.fromkeys(r.get("Filename", "") for r in rows))

    screenshot_map = {}
    if screenshots:
        for name in file_order:
            key = runner.norm_key(name)
            png = os.path.join(screenshots, f"{runner.sanitize_filename(name)}.png")
            if os.path.exists(png):
                screenshot_map[key] = png

    docx_out = docx_out or os.path.splitext(results_csv)[0] + ".docx"
    ok, msg = runner.write_docx(
        rows, docx_out, file_order, screenshot_map, csv_name=os.path.basename(results_csv)
    )
    if not ok:
        raise click.ClickException(msg)
    click.echo(f"Wrote {docx_out}.")

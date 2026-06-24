"""CLI commands for the Help plugin."""

from __future__ import annotations

import click


@click.group(name="help")
def cli():
    """Browse ChiSurf documentation from the command line."""


@cli.command("list")
def list_docs():
    """List all available documentation files."""
    from chisurf.plugins.core.help.api.io import discover_docs

    info = discover_docs()
    click.echo(f"Found {len(info.entries)} documentation files:")
    click.echo("")
    for entry in info.entries:
        click.echo(f"  [{entry.category}] {entry.title}")
        click.echo(f"         {entry.path}")


@cli.command("read")
@click.argument("path", type=str)
def read_doc(path: str):
    """Read a documentation file and print its Markdown content."""
    from chisurf.plugins.core.help.api.io import read_doc

    content = read_doc(path)
    if content is None:
        click.echo(f"Error: cannot read {path}", err=True)
        raise SystemExit(1)
    click.echo(content)


@cli.command("render")
@click.argument("path", type=str)
def render_doc(path: str):
    """Render a documentation file to HTML and print."""
    from chisurf.plugins.core.help.api.io import read_doc
    from chisurf.plugins.core.help.api.markdown import render_markdown

    content = read_doc(path)
    if content is None:
        click.echo(f"Error: cannot read {path}", err=True)
        raise SystemExit(1)
    html = render_markdown(content)
    click.echo(html)


@cli.command("search")
@click.argument("query", type=str)
def search(query: str):
    """Search documentation files for a query string."""
    from chisurf.plugins.core.help.api.io import search_docs

    results = search_docs(query)
    if not results:
        click.echo("No results found.")
        return
    click.echo(f"Found {len(results)} result(s):")
    click.echo("")
    for r in results:
        click.echo(f"  [{r['match_type']}] {r['title']}")
        click.echo(f"         {r['path']}")


if __name__ == "__main__":
    cli()


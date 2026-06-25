"""Command line interface for the light-path simulator plugin."""

from __future__ import annotations

import json
from typing import Any

import click

from ..api.contract import contract_descriptor
from ..core.workflow import (
    get_lightpath,
    get_probes_info,
    list_lightpaths,
    save_lightpath,
    simulate_lightpath,
)


def load_json_file(path: str) -> dict[str, Any]:
    """Load a JSON object from a file."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise click.UsageError(f"Expected JSON object in {path}")
    return data


def emit_json(payload: Any) -> None:
    """Print a JSON payload."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show the version and exit.")
@click.pass_context
def cli(ctx: click.Context, version: bool) -> None:
    """Light-path simulator CLI."""
    if version:
        click.echo("Light Path Simulator CLI v1.0.0")
        return
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
def contract() -> None:
    """Print the JSON workflow contract."""
    emit_json(contract_descriptor())


@cli.command()
@click.argument("graph_json", type=click.Path(exists=True, dir_okay=False))
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None, help="MFDB SQLite path.")
def simulate(graph_json: str, db_path: str | None) -> None:
    """Run a simulation from GRAPH_JSON."""
    emit_json(simulate_lightpath(load_json_file(graph_json), db_path=db_path))


@cli.command(name="save")
@click.argument("graph_json", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", default=None, help="Saved simulation name.")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None, help="MFDB SQLite path.")
def save_cmd(graph_json: str, name: str | None, db_path: str | None) -> None:
    """Save a simulation from GRAPH_JSON to MFDB."""
    emit_json(save_lightpath(load_json_file(graph_json), name=name, db_path=db_path))


@cli.command(name="list")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None, help="MFDB SQLite path.")
def list_cmd(db_path: str | None) -> None:
    """List saved light-path simulations."""
    emit_json(list_lightpaths(db_path=db_path))


@cli.command(name="get")
@click.argument("operation_id")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None, help="MFDB SQLite path.")
def get_cmd(operation_id: str, db_path: str | None) -> None:
    """Load a saved light-path simulation by operation id."""
    emit_json(get_lightpath(operation_id, db_path=db_path))


@cli.command(name="probes")
@click.option("--db", "db_path", type=click.Path(dir_okay=False), default=None, help="MFDB SQLite path.")
def probes_cmd(db_path: str | None) -> None:
    """List probe metadata used by the simulator."""
    emit_json(get_probes_info(db_path=db_path))


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option("--cmd-port", default=8765, show_default=True, type=int, help="ZMQ command port.")
@click.option("--pub-port", default=8766, show_default=True, type=int, help="ZMQ PUB port.")
def serve(host: str, cmd_port: int, pub_port: int) -> None:
    """Serve light-path simulator methods over a dedicated ZMQ server."""
    from ..rpc.methods import serve as serve_methods

    serve_methods(host=host, cmd_port=cmd_port, pub_port=pub_port)

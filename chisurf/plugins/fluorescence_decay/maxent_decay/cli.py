"""Click-based CLI for the MaxEnt TCSPC lifetime / FRET plugin.

This is a minimal stub that provides a ``cli`` entry point so that the
:mod:`fmem` compatibility wrapper can import it. The ChiSurf umbrella
CLI exposes this group under the ``maxent-decay`` subcommand.
"""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """MaxEnt TCSPC lifetime / FRET tools.

    Currently this CLI only provides a placeholder group; subcommands may
    be added in the future.
    """


def main() -> None:
    """Entry point used by legacy scripts (calls :func:`cli`)."""

    cli(prog_name="maxent-decay")


__all__ = ["cli", "main"]

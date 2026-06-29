"""Compat shim: exposes the CLI group at package root for the console_scripts entry point."""

from __future__ import annotations

from .cli.main import cli

__all__ = ["cli"]

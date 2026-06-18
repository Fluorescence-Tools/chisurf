"""MaxEnt TCSPC lifetime/FRET MEM plugin."""

from __future__ import annotations

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest else "Spectroscopy:Fluorescence decay:MaxEnt MEM"
cli_entrypoint = _manifest.entrypoints.cli if _manifest else "maxent-decay=chisurf.plugins.fluorescence_decay.maxent_decay.cli.cli:cli"


def load():
    """Return the standalone MaxEnt MEM widget."""
    from .gui.gui import MaxentDecayWidget

    return MaxentDecayWidget()


if __name__ == "plugin":
    widget = load()
    widget.show()

__all__ = ["name", "cli_entrypoint", "load"]

"""Calculators hub plugin.

A two-panel launcher: pick a calculator on the left, use it embedded on the
right. It groups the independent FRET-line, FRET/homoFRET and FCS calculators
under a single ``Main:Tools:Calculators`` entry. The catalogue of calculators is
Qt-free data in :mod:`.core.registry`; the GUI host (:mod:`.gui.tool`) resolves
and embeds each calculator widget lazily on first selection.
"""

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
if _manifest is not None:
    name = _manifest.display_name
else:
    name = "Main:Tools:Calculators"


if __name__ == "plugin":
    from .gui.tool import CalculatorHub

    _hub = CalculatorHub()
    _hub.show()

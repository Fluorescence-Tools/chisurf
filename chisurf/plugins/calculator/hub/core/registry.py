"""Registry of embeddable calculators for the Calculators hub (Qt-free).

The hub is a two-panel tool: a list of calculators on the left, the selected
calculator embedded on the right. Each entry names an *embeddable* ``QWidget``
(a plain panel or window, no required constructor arguments) by dotted import
path; the GUI resolves and instantiates it lazily on first selection. Keeping the
catalogue here — as plain data — lets it be unit-tested and extended without
importing Qt.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class CalculatorEntry:
    """One selectable calculator in the hub.

    Parameters
    ----------
    id : str
        Stable identifier (used for selection persistence).
    label : str
        Text shown in the selector list.
    description : str
        One-line summary shown above the embedded calculator.
    widget : str
        Dotted import path ``"pkg.module:ClassName"`` of an embeddable ``QWidget``
        that constructs with no required arguments.
    icon : str
        Optional emoji/icon shown next to the label.
    """

    id: str
    label: str
    description: str
    widget: str
    icon: str = ""


def default_calculators() -> list[CalculatorEntry]:
    """Return the built-in calculators embedded by the hub."""
    return [
        CalculatorEntry(
            id="fret_calculator",
            label="FRET / homoFRET",
            description=(
                "Combined heteroFRET and homoFRET parameter calculator — convert "
                "between distance, efficiency, lifetime and rate."
            ),
            widget="chisurf.plugins.calculator.fret_calculator.gui.tool:FretCalculatorTool",
            icon="🧮",
        ),
        CalculatorEntry(
            id="fret_line",
            label="FRET line",
            description=(
                "Compute static, dynamic, WLC and mixture FRET lines for parameter "
                "ranges, ready to overlay on smFRET 2D histograms."
            ),
            widget="chisurf.plugins.fret_line.gui.tool:FRETLineTool",
            icon="📈",
        ),
        CalculatorEntry(
            id="fcs_calculator",
            label="FCS diffusion",
            description=(
                "Confocal-FCS diffusion/volume calculator — solve τ, D, rₕ, V_eff "
                "and concentration from one constraint."
            ),
            widget="chisurf.plugins.fcs.fcs_calculator.wizard:ConfocalCalcWidget",
            icon="🌀",
        ),
        CalculatorEntry(
            id="phasor",
            label="Phasor plot",
            description=(
                "Interactive phasor plot — universal semicircle with a reference-"
                "lifetime grid/ticks, a FRET trajectory and a two-component mixing line."
            ),
            widget="chisurf.plugins.calculator.phasor_calculator.gui.tool:PhasorCalculatorTool",
            icon="◐",
        ),
    ]


__all__ = ["CalculatorEntry", "default_calculators"]

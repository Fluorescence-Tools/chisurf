"""Registry of embeddable wizards for the Wizards hub (Qt-free).

The hub is a two-panel tool: a list of wizards on the left, the selected wizard
embedded on the right. Each entry names an *embeddable* ``QWidget`` (a plain
panel, no window chrome) by dotted import path; the GUI resolves and instantiates
it lazily on first selection. Keeping the catalogue here — as plain data — lets it
be unit-tested and extended without importing Qt.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class WizardEntry:
    """One selectable wizard in the hub.

    Parameters
    ----------
    id : str
        Stable identifier (used for selection persistence).
    label : str
        Text shown in the selector list.
    description : str
        One-line summary shown above the embedded wizard.
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


def default_wizards() -> list[WizardEntry]:
    """Return the built-in wizards embedded by the hub."""
    return [
        WizardEntry(
            id="anisotropy",
            label="Anisotropy",
            description="Build a linked VV/VH global time-resolved anisotropy fit.",
            widget="chisurf.plugins.fluorescence_decay.tr_anisotropy.gui.tool:AnisotropyAssistantWidget",
            icon="🔬",
        ),
        WizardEntry(
            id="batch_analysis",
            label="Batch analysis",
            description="Apply one template fit to many datasets or files.",
            widget="chisurf.plugins.core.batch_analysis.gui.tool:BatchAnalysisWidget",
            icon="📋",
        ),
    ]


__all__ = ["WizardEntry", "default_wizards"]

"""Interactive phasor-plot calculator (PRD-56).

A data-free phasor plot for the Calculators hub: place reference lifetimes, a FRET
trajectory and a two-component mixing line on the universal semicircle. Built from a
declarative ``phasor.view.json`` (AutoForm) over a plain model that assembles the
overlay geometry through the shared phasor toolkit
(:mod:`chisurf.plugins.microscopy.img_pixel_phasor.analysis`).
"""

from pathlib import Path

from chisurf.core.plugin import load_manifest

_manifest = load_manifest(Path(__file__).with_name("manifest.json"))
name = _manifest.display_name if _manifest is not None else "Main:Tools:Phasor-Calculator"

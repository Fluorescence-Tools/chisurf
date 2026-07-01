# Calculators hub

`Main:Tools:Calculators` — a two-panel launcher that groups ChiSurf's standalone
calculators under one window. Pick a calculator on the left; it is embedded on
the right and built lazily on first selection.

Bundled calculators:

- **FRET / homoFRET** — `chisurf.plugins.calculator.fret_calculator` — convert
  between donor–acceptor distance, transfer efficiency, lifetime and rate
  (hetero- and homoFRET).
- **FRET line** — `chisurf.plugins.fret_line` — static/dynamic/WLC/mixture FRET
  lines for overlaying on smFRET 2D histograms.
- **FCS diffusion / volume** — `chisurf.plugins.fcs.fcs_calculator` — solve τ, D,
  rₕ, V_eff and concentration for confocal FCS from one constraint.

The catalogue is Qt-free data in `core/registry.py` (unit-tested headless); the
GUI host in `gui/tool.py` resolves each widget from its dotted import path. Adding
a calculator is a one-line `CalculatorEntry` — no GUI changes needed.

Each bundled calculator remains independently launchable from its own menu entry;
the hub simply composes them. Launch standalone with `csg_calculators`.

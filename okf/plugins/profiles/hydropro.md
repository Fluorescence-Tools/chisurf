---
type: Plugin Profile
title: HydroPro plugin
description: OKF profile for the HYDROPRO/HYDRO++ front-end.
resource: chisurf/plugins/modelling/hydropro/
tags: [plugins, modelling, cli, rpc, external-tool]
timestamp: '2026-07-05T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `hydropro` |
| Display name | `Structure:Computation:HydroPro` |
| Categories | `Structure` |
| Version | `1.0.0` |
| State namespace | `hydropro` |
| Local README | Missing |

The manifest describes a GUI front-end to HYDROPRO/HYDRO++ for hydrodynamic
properties such as translational diffusion coefficients from atomic or bead-model
structures.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| Core | `core/runner.py`, `core/settings.py`. |
| RPC | `rpc/services.py` with `hydropro.run` and `hydropro.parse_res`. |
| CLI | `cli/main.py`, manifest command `hydropro=...`. |
| GUI | `gui/tool.py`, `gui/dialogs.py`, `gui/hydropro.view.json`. |
| Legacy | `hydrogui.py` remains present. |
| Tests | `test/test_widgets.py`. |

# Data And Provenance Impact

The main risk is external executable behavior and generated output files. Docs should
state how the HYDROPRO/HYDRO++ binary is discovered, where working files are written,
which outputs are parsed, and how failures are surfaced.

# Verification Surface

Focused smoke test:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/modelling/hydropro/test
```

External-binary integration needs a separate manual or optional test path.

# Documentation Work

- Add `chisurf/plugins/modelling/hydropro/README.md`.
- Include external binary setup and platform assumptions.
- Add CLI examples for run and result parsing.
- Document generated files and cleanup behavior.

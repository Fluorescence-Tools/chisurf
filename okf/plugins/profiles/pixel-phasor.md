---
type: Plugin Profile
title: Pixel Phasor plugin
description: OKF profile for the Phasor-FLIM imaging plugin.
resource: chisurf/plugins/microscopy/img_pixel_phasor/
tags: [plugins, imaging, phasor, flim, rpc]
timestamp: '2026-07-05T00:00:00Z'
---

# Identity

| Field | Value |
| --- | --- |
| Plugin id | `img_pixel_phasor` |
| Display name | `Imaging:Phasor-FLIM` |
| Categories | `Imaging`, `Phasor-FLIM` |
| Version | `1.0.0` |
| State namespace | `img_pixel_phasor` |
| Local README | Missing |

The manifest describes per-pixel phasor `(g, s)` maps and a phasor plot from TTTR
imaging data.

# Architecture Evidence

| Layer | Evidence |
| --- | --- |
| Core/analysis | `core.py`, `analysis.py`. |
| Backend services | `backend/services.py`. |
| CLI | `cli/main.py`, manifest command `img-pixel-phasor=...`. |
| GUI | `gui/tool.py`, `gui/view_model.py`, `gui/phasor.view.json`. |
| Tests | `test/test_analysis.py`, `test/test_services.py`. |

Manifest RPC methods cover description, apparent lifetime, filtering, component
fractions, unmixing, cursor masks, pseudo-color maps, and overlays.

# Data And Provenance Impact

This plugin is part of the imaging analysis pipeline. It should document whether it
reads the standard imaging HDF5 produced by intensity reconstruction, what arrays it
writes back or exports, and how calibration or modulation frequency settings are
provided.

# Verification Surface

Focused test command:

```bash
PYTHONPATH="modules/chinet:modules/imp-tricks/src:." python3 -m pytest chisurf/plugins/microscopy/img_pixel_phasor/test
```

# Documentation Work

- Add `chisurf/plugins/microscopy/img_pixel_phasor/README.md`.
- Document input HDF5/image assumptions and output arrays.
- Add CLI examples for headless phasor processing.
- Link GUI controls in `phasor.view.json` to service methods.

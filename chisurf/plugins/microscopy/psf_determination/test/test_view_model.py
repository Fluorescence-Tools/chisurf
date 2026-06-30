"""Headless tests for the Qt-free :class:`PsfViewModel` (no Qt/pyqtgraph)."""

from __future__ import annotations

import numpy as np

from chisurf.plugins.microscopy.psf_determination.gui.view_model import PsfViewModel


def _synthetic_stack(nz=21, ny=40, nx=40, sigma_xy=2.0, sigma_z=4.0):
    """Return a single 3D Gaussian bead centred in the stack on a small offset."""
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    zc, yc, xc = nz / 2, ny / 2, nx / 2
    bead = 1000.0 * np.exp(
        -0.5 * (((x - xc) / sigma_xy) ** 2 + ((y - yc) / sigma_xy) ** 2 + ((z - zc) / sigma_z) ** 2)
    )
    return (bead + 5.0).astype(np.float32)


def test_load_detect_fit_roundtrip():
    model = PsfViewModel()
    events: list[str] = []
    model.add_observer(events.append)

    stack = _synthetic_stack()
    model.set_stack(stack)
    assert model.stack is not None
    assert "stack" in events

    n = model.detect_beads()
    assert n >= 1
    assert "beads" in events
    assert model.selected_bead is not None

    fit = model.fit_selected()
    assert fit is not None and fit["success"]
    # Fitted lateral sigma should be close to the ground-truth 2.0 px.
    _, _, _, sigma_z, sigma_y, sigma_x, _, _ = fit["params"]
    assert abs(sigma_x - 2.0) < 1.0
    assert abs(sigma_y - 2.0) < 1.0
    assert sigma_z > sigma_x  # axial worse than lateral

    # Profiles and overlay accessors are populated.
    assert len(model.x_profile_series()) == 2
    assert len(model.z_profile_series()) == 2
    assert model.fit_circle() is not None
    assert "PSF Fit Results" in model.results_text


def test_fit_all_and_export(tmp_path):
    model = PsfViewModel()
    model.set_stack(_synthetic_stack())
    model.detect_beads()
    results = model.fit_all()
    assert results
    assert "Batch PSF fits" in model.results_text

    out = tmp_path / "psf.csv"
    model.export_csv(str(out))
    assert out.exists()
    assert out.read_text().splitlines()[0].startswith("index,")

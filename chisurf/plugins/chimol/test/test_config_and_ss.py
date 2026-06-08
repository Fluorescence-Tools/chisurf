"""Regression tests for Chimol config loading and SS assignment."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from chisurf.plugins.chimol.chimol import config
from chisurf.plugins.chimol.chimol.analysis.ss import assign_ss_c3_from_atoms


def _build_minimal_atoms(n_res: int) -> np.ndarray:
    """Construct a tiny ChiSurf-style atoms array with N/CA/C/O per residue."""

    # Simple straight backbone along x-axis; spacing 1.5 Å.
    coords = []
    for i in range(n_res):
        base = float(i) * 1.5
        coords.extend(
            [
                ("N", np.array([base, 0.0, 0.0]), i),
                ("CA", np.array([base + 0.5, 0.0, 0.0]), i),
                ("C", np.array([base + 1.0, 0.0, 0.0]), i),
                ("O", np.array([base + 1.2, 0.2, 0.0]), i),
            ]
        )
    dtype = [("atom_name", "U4"), ("xyz", float, (3,)), ("res_id", int)]
    return np.array(coords, dtype=dtype)


def test_display_config_prefers_settings_dir(tmp_path, monkeypatch):
    """Ensure config reads chimol_display.json from ChiSurf settings dir."""

    cfg_path = tmp_path / "chimol_display.json"
    cfg_path.write_text('{"background": "w", "camera": {"near_clip": 0.5}}', encoding="utf-8")

    fake_settings = SimpleNamespace(get_path=lambda name: tmp_path)
    monkeypatch.setattr(config, "_cs_settings", fake_settings)
    monkeypatch.setenv("CHIMOL_DISPLAY_CONFIG", str(cfg_path))

    importlib.reload(config)

    assert config._DISPLAY_CONFIG["background"] == "w"
    # camera keys should be merged with defaults
    assert config._DISPLAY_CONFIG["camera"]["near_clip"] == 0.5
    assert "far_clip" in config._DISPLAY_CONFIG["camera"]


def test_display_config_falls_back_to_defaults(tmp_path, monkeypatch):
    """When no config files exist, defaults should be loaded."""

    fake_settings = SimpleNamespace(get_path=lambda name: tmp_path)
    monkeypatch.setattr(config, "_cs_settings", fake_settings)
    monkeypatch.delenv("CHIMOL_DISPLAY_CONFIG", raising=False)

    importlib.reload(config)

    assert config._DISPLAY_CONFIG["background"] == "k"
    assert config._DISPLAY_CONFIG["cartoon"]["style"] == "ribbon"


def test_assign_ss_c3_from_atoms_returns_codes():
    """Basic sanity: SS assignment returns H/E/C codes with requested length."""

    atoms = _build_minimal_atoms(6)
    codes = assign_ss_c3_from_atoms(atoms, n_res=6, verbose=False)
    assert codes is not None
    assert len(codes) == 6
    assert set(codes).issubset({"H", "E", "C"})

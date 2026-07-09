"""Example-data generator + end-to-end file-based analysis test."""

from __future__ import annotations

import pathlib

import pytest

from chisurf.plugins.burst.burst_h2mm.api.models import H2mmSettings, StreamSettings
from chisurf.plugins.burst.burst_h2mm.backend.services import (
    run_analysis,
    write_result_tables,
)
from chisurf.plugins.burst.burst_h2mm.examples.generate_example_data import (
    generate_example_data,
)

pytest.importorskip("tttrlib")
pytest.importorskip("tables")  # pandas HDF5 backend for the ndX output


def test_generate_and_analyze_example_dataset(tmp_path):
    """The generated .bur + Photon-HDF5 loads and fits through the real pipeline."""
    bur_path, tttr_path = generate_example_data(tmp_path, n_bursts=120, burst_len=80, seed=1)
    assert bur_path.exists() and tttr_path.exists()
    assert tttr_path.name.endswith(".photon.h5")

    settings = H2mmSettings(
        streams=[StreamSettings("donor", [0]), StreamSettings("acceptor", [1])],
        min_states=1,
        max_states=2,
        n_restarts=1,
        max_iter=200,
        min_photons=5,
        file_type="auto",   # Photon-HDF5 is auto-detected
    )
    result, bundle = run_analysis(settings, analysis_folder=str(tmp_path))

    # Two clearly-separated states were simulated → recovered.
    assert result.n_bursts > 0
    assert result.n_states == 2
    e = sorted(result.fret)
    assert e[0] < 0.4 < e[1]

    # The defined output set is written and openable.
    out_dir = tmp_path / "h2mm"
    write_result_tables(result, bundle, out_dir)
    assert (out_dir / "h2mm_result.json").exists()
    assert (out_dir / "h2mm_photons.h5").exists() or (out_dir / "h2mm_photons.csv").exists()
    assert (out_dir / "h2mm_bursts.csv").exists()
    assert pathlib.Path(result.output_paths["result_json"]).exists()

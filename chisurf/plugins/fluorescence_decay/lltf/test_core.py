from __future__ import annotations

from pathlib import Path
import json

import pytest
import yaml

from chisurf.plugins.lltf.core.fitter import fit_lifetime


HERE = Path(__file__).parent
EXAMPLE_DIR = HERE / "example"
DECAY_FILE = EXAMPLE_DIR / "5-44_D0.dat"
IRF_FILE = EXAMPLE_DIR / "IRF_D0.dat"
CONFIG_FILE = EXAMPLE_DIR / "config.yml"


if not (DECAY_FILE.exists() and IRF_FILE.exists() and CONFIG_FILE.exists()):
    pytest.skip("LLTF example data not available", allow_module_level=True)


@pytest.mark.parametrize("n_lifetimes", [1, 2])
def test_fit_lifetime_example_data(tmp_path, n_lifetimes: int) -> None:
    """Basic smoke test for the fit_lifetime API using bundled example data."""
    output_file = tmp_path / f"lltf_n{n_lifetimes}.json"
    plot_file = tmp_path / f"lltf_n{n_lifetimes}.png"

    with CONFIG_FILE.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    result = fit_lifetime(
        decay_file=str(DECAY_FILE),
        irf_file=str(IRF_FILE),
        n_lifetimes=n_lifetimes,
        skiprows=0,
        delimiter=None,
        time_column=0,
        counts_column=1,
        output_file=str(output_file),
        plot_file=str(plot_file),
        verbose=False,
        config=config,
        save_intermediate_results=False,
        intermediate_results_base_filename=None,
    )

    # Basic invariants on the returned result
    assert result["n_lifetimes"] == n_lifetimes
    assert isinstance(result["lifetime_spectrum"], list)
    assert len(result["lifetime_spectrum"]) == 2 * n_lifetimes

    # Files should have been written and be parseable
    assert output_file.exists()
    assert plot_file.exists()

    with output_file.open("r", encoding="utf-8") as fh:
        json.load(fh)

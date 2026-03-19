from __future__ import annotations

from pathlib import Path
import json

import pytest
from click.testing import CliRunner

from chisurf.plugins.fluorescence_decay.lltf.core.cli import cli


HERE = Path(__file__).parent
EXAMPLE_DIR = HERE / "example"
DECAY_FILE = EXAMPLE_DIR / "5-44_D0.dat"
IRF_FILE = EXAMPLE_DIR / "IRF_D0.dat"
CONFIG_FILE = EXAMPLE_DIR / "config.yml"


if not (DECAY_FILE.exists() and IRF_FILE.exists() and CONFIG_FILE.exists()):
    pytest.skip("LLTF example data not available", allow_module_level=True)


def test_lltf_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    # For debugging in CI
    print(result.output)

    assert result.exit_code == 0
    # CLI help text should at least mention lifetime fitting
    assert "lifetime" in result.output.lower()


def test_lltf_cli_fit_example(tmp_path) -> None:
    """Run the LLTF CLI against the bundled example data.

    This exercises the Click command wiring and the fitter end-to-end, while
    keeping all outputs in a temporary directory.
    """
    out_dir = tmp_path

    args = [
        "fit",
        str(DECAY_FILE),
        str(IRF_FILE),
        "-c",
        str(CONFIG_FILE),
        "-sp",
        str(out_dir),
        "-n",
        "2",
        "-v",
    ]

    runner = CliRunner()
    result = runner.invoke(cli, args)

    # For debugging in CI
    print(result.output)

    assert result.exit_code == 0

    base = DECAY_FILE.stem
    json_path = out_dir / f"{base}_fit.json"
    png_path = out_dir / f"{base}_fit.png"

    assert json_path.exists()
    assert png_path.exists()

    # Ensure the JSON result has the structure we expect
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data.get("n_lifetimes") >= 1
    assert isinstance(data.get("lifetime_spectrum"), list)

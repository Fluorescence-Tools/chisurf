"""Real-data tests for the Burst Selection API and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from chisurf.core.fio.fluorescence.burst import generate_burst_dataframe
from chisurf.plugins.burst.burst_selection.api.io import load_tttr
from chisurf.plugins.burst.burst_selection.api.models import (
    AnalysisSettings,
    BurstDetectionSettings,
    DeltaMacroTimeFilterSettings,
    GMMSettings,
    PhotonFilterSettings,
)
from chisurf.plugins.burst.burst_selection.api.selection import (
    analyze_file,
    apply_photon_filters,
    find_bursts,
)
from chisurf.plugins.burst.burst_selection.cli import cli

DATA_DIR = Path(__file__).resolve().parent / "data" / "bh_spc132_sm_dna"
BH_SPC_FILE = DATA_DIR / "m000.spc"
STREAM_CHANNELS = [0, 1, 8, 9]


def real_data_settings() -> AnalysisSettings:
    """Return deterministic settings for the bundled BH SPC example."""
    settings = AnalysisSettings()
    settings.photon_filter = PhotonFilterSettings(
        channels=STREAM_CHANNELS,
        filter_active=False,
        delta_macro_time_filter=DeltaMacroTimeFilterSettings(dT_min=0.0),
    )
    settings.burst_detection = BurstDetectionSettings(
        min_photons=20,
        photon_window=10,
        time_window=1e-3,
    )
    settings.gmm = GMMSettings(
        covariance_type="spherical",
        random_state=42,
        max_iter=50,
        n_init=1,
    )
    return settings


@pytest.fixture()
def bur_file(tmp_path: Path) -> Path:
    """Analyze the real BH SPC file and return the generated .bur path."""
    result = analyze_file(BH_SPC_FILE, settings=real_data_settings(), output_dir=tmp_path)
    return Path(result.output_paths["bur"])


def test_real_bh_spc_channel_mask_uses_stream_channels() -> None:
    """The bundled BH SPC example uses stream channels 0/8 green and 1/9 red."""
    tttr = load_tttr(BH_SPC_FILE)
    selected = apply_photon_filters(tttr, real_data_settings().photon_filter)
    selected_mask = selected.astype(bool)
    assert set(np.unique(tttr.routing_channel[selected_mask]).tolist()) == set(STREAM_CHANNELS)
    assert int(selected_mask.sum()) == 126887


def test_real_bh_spc_analyze_file_matches_core_burst_dataframe(tmp_path: Path) -> None:
    """Real BH SPC API analysis should match the existing core burst helper."""
    tttr = load_tttr(BH_SPC_FILE)
    settings = real_data_settings()
    selected = apply_photon_filters(tttr, settings.photon_filter)
    start_stop = find_bursts(selected)
    api_result = analyze_file(BH_SPC_FILE, settings=settings, output_dir=tmp_path)
    core_df = generate_burst_dataframe(
        start_stop=start_stop,
        filename=BH_SPC_FILE,
        tttr=tttr,
        windows={},
        detectors={},
        include_interleaved_zeros=True,
    )
    api_df = pd.DataFrame(api_result.dataframes[str(BH_SPC_FILE)])

    pd.testing.assert_frame_equal(api_df, core_df)
    assert api_result.metadata == {
        "n_photons": 174438,
        "n_selected": 126887,
        "n_bursts": int(len(start_stop)),
    }
    assert api_result.metadata["n_bursts"] == 5577
    assert Path(api_result.output_paths["bur"]).exists()


def test_real_bh_spc_cli_inspect_uses_generated_bur(bur_file: Path) -> None:
    """The CLI inspect command should summarize a real generated .bur file."""
    result = CliRunner().invoke(cli, ["inspect", str(bur_file)])
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["n_rows"] == 9533
    assert "nphotons" in payload["feature_columns"]


def test_real_bh_spc_cli_fit_gmm_uses_generated_bur(bur_file: Path) -> None:
    """The CLI fit-gmm command should fit features from a real generated .bur file."""
    result = CliRunner().invoke(
        cli,
        ["fit-gmm", str(bur_file), "--settings", '{"covariance_type":"spherical","n_init":1}'],
    )
    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["n_components"] == 1
    assert len(payload["labels"]) == 9533

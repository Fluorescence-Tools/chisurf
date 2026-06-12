"""Tests for the Burst Selection API adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chisurf.core.fio.fluorescence.burst import generate_burst_dataframe
from chisurf.plugins.burst.burst_selection.api import selection as selection_module
from chisurf.plugins.burst.burst_selection.api.contract import (
    METHOD_ANALYZE_FILES,
    analysis_request_from_payload,
    analysis_request_to_payload,
    analysis_result_to_payload,
    contract_descriptor,
)
from chisurf.plugins.burst.burst_selection.api.features import extract_features, fit_gmm
from chisurf.plugins.burst.burst_selection.api.io import load_tttr
from chisurf.plugins.burst.burst_selection.api.models import (
    AnalysisRequest,
    AnalysisResult,
    AnalysisSettings,
    BurstDetectionSettings,
    BurstFilterMode,
    DeltaMacroTimeFilterSettings,
    GMMSettings,
    PhotonFilterSettings,
)
from chisurf.plugins.burst.burst_selection.api.selection import (
    analyze_file,
    analyze_request,
    apply_photon_filters,
    find_bursts,
    legacy_output_folder_name,
    summarize_bursts,
)
from chisurf.plugins.burst.burst_selection.api.serialization import settings_from_dict, to_jsonable
from chisurf.plugins.burst.burst_selection.gui.adapter import (
    burst_rows_for_display,
    make_ui_dataframe,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "bh_spc132_sm_dna"
BH_SPC_FILE = DATA_DIR / "m000.spc"
STREAM_CHANNELS = [0, 1, 8, 9]


def real_data_settings() -> AnalysisSettings:
    """Return deterministic settings for the bundled BH SPC example."""
    settings = AnalysisSettings()
    settings.photon_filter = PhotonFilterSettings(
        channels=STREAM_CHANNELS,
        filter_active=False,
        delta_macro_time_filter=DeltaMacroTimeFilterSettings(
            dT_min=0.0,
            dT_min_active=False,
            dT_max_active=False,
        ),
    )
    settings.burst_detection = BurstDetectionSettings(
        min_photons=20,
        photon_window=10,
        time_window=1e-3,
    )
    return settings


def test_contract_descriptor_defines_workflow_io() -> None:
    """The public contract should define canonical analysis input and output."""
    contract = contract_descriptor()
    assert contract["plugin_id"] == "burst_selection"
    assert contract["contract_version"]
    assert contract["inputs"]["AnalyzeFiles"]["required"] == ["files"]
    assert contract["outputs"]["AnalysisResult"]["required"] == [
        "files",
        "dataframes",
        "output_paths",
        "metadata",
    ]
    assert contract["rpc_methods"][METHOD_ANALYZE_FILES]["input"] == "AnalyzeFiles"


def test_analysis_request_payload_roundtrip_normalizes_json_inputs() -> None:
    """Workflow payloads should normalize to dataclasses and back to JSON."""
    request = analysis_request_from_payload(
        {
            "files": ["m000.spc"],
            "windows": {"prompt": [0, 2048]},
            "settings": {
                "photon_filter": {
                    "channels": [0, 1],
                    "microtime_ranges": None,
                    "used_filter": "burst",
                },
                "output_formats": ["bur", "hdf5"],
            },
            "legacy_output": True,
            "selected_setup": "Test",
        }
    )

    assert request.files == ["m000.spc"]
    assert request.windows == {"prompt": (0, 2048)}
    assert request.settings.photon_filter.microtime_ranges == []
    assert request.settings.photon_filter.used_filter == BurstFilterMode.BURST
    assert request.settings.output_formats == ["bur", "hdf5"]
    assert request.legacy_output is True
    assert request.selected_setup == "Test"

    payload = analysis_request_to_payload(request)
    assert payload["windows"] == {"prompt": [0, 2048]}
    assert payload["settings"]["photon_filter"]["used_filter"] == "burst"


def test_analysis_result_payload_is_json_safe() -> None:
    """Analysis results should serialize through the workflow output helper."""
    payload = analysis_result_to_payload(
        AnalysisResult(
            files=[str(BH_SPC_FILE)],
            dataframes={str(BH_SPC_FILE): [{"First Photon": 0}]},
            output_paths={"bur": str(BH_SPC_FILE.with_suffix(".bur"))},
            metadata={"n_photons": 1},
        )
    )

    assert payload["files"] == [str(BH_SPC_FILE)]
    assert payload["dataframes"][str(BH_SPC_FILE)][0]["First Photon"] == 0
    assert payload["output_paths"]["bur"].endswith(".bur")


def test_find_bursts_bridges_configured_gap() -> None:
    """Burst finding should bridge small gaps in the selection mask."""
    mask = np.array([1, 1, 0, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
    bursts = find_bursts(mask, max_gap=2)
    assert bursts.tolist() == [[0, 8]]


def test_apply_photon_filters_without_filter_uses_tttr_length() -> None:
    """Disabled filtering should return a selection mask with the TTTR length."""
    tttr = load_tttr(BH_SPC_FILE)
    settings = real_data_settings()
    selected = apply_photon_filters(tttr, settings.photon_filter)
    assert selected.shape == (len(tttr),)
    assert np.all(selected == 1)


def test_analyze_file_writes_bur(tmp_path: Path) -> None:
    """API analysis should produce a ChiSurf-compatible .bur file from real data."""
    settings = real_data_settings()
    result = analyze_file(BH_SPC_FILE, settings=settings, output_dir=tmp_path)
    bur_path = tmp_path / "m000.bur"
    assert bur_path.exists()
    assert set(result.output_paths) == {"bur"}
    df = pd.read_csv(bur_path, sep="\t")
    assert len(df) == len(result.dataframes[str(BH_SPC_FILE)])
    assert "First Photon" in df.columns
    assert (pd.to_numeric(df["Number of Photons"], errors="coerce").fillna(0) == 0).any()
    assert len(make_ui_dataframe(df)) < len(df)


def test_analyze_request_writes_legacy_burstwise_output(tmp_path: Path) -> None:
    """API legacy-output mode should own the old burstwise folder layout."""
    source = tmp_path / BH_SPC_FILE.name
    source.write_bytes(BH_SPC_FILE.read_bytes())
    settings = real_data_settings()
    settings.photon_filter.channels = []
    settings.output_formats = ["bur"]
    settings.photon_filter.delta_macro_time_filter.dT_max = 0.2
    settings.burst_detection.min_photons = 60

    result = analyze_request(
        AnalysisRequest(
            files=[str(source)],
            settings=settings,
            legacy_output=True,
            selected_setup="Test",
        )
    )

    output_folder = tmp_path / legacy_output_folder_name(settings)
    bur_path = output_folder / "bi4_bur" / f"{source.stem}.bur"
    info_dir = output_folder / "Info"
    mti_files = list(info_dir.glob("*.mti"))

    assert output_folder.is_dir()
    assert bur_path.exists()
    assert (info_dir / "photon_selection_parameters.json").exists()
    assert (info_dir / "datetime.txt").exists()
    assert len(mti_files) == 1
    assert str(source) in mti_files[0].read_text()
    assert result.output_paths["output_folder"] == str(output_folder)
    assert result.metadata["output_folder"] == str(output_folder)


def test_analyze_request_reuses_first_macro_time_resolution(monkeypatch) -> None:
    """Batch output summaries should reuse the first file's macro-time resolution."""
    calls: list[float | None] = []

    def fake_analyze_file(path: str, **kwargs: object):
        calls.append(kwargs.get("macro_time_resolution"))
        return selection_module.AnalysisResult(
            files=[path],
            dataframes={path: []},
            metadata={
                "n_photons": 1,
                "n_selected": 1,
                "n_bursts": 0,
                "macro_time_resolution": 0.001,
            },
        )

    monkeypatch.setattr(selection_module, "analyze_file", fake_analyze_file)

    selection_module.analyze_request(
        AnalysisRequest(
            files=["first.spc", "second.spc"],
            settings=AnalysisSettings(output_formats=[]),
        )
    )

    assert calls == [None, 0.001]


def test_analyze_file_duration_uses_macro_time_resolution_override() -> None:
    """Output burst durations should use the supplied macro-time resolution."""
    settings = real_data_settings()
    tttr = load_tttr(BH_SPC_FILE)
    native_resolution = float(tttr.header.macro_time_resolution)

    native = analyze_file(BH_SPC_FILE, settings=settings)
    overridden = analyze_file(
        BH_SPC_FILE,
        settings=settings,
        macro_time_resolution=native_resolution * 2.0,
    )

    native_df = make_ui_dataframe(pd.DataFrame(native.dataframes[str(BH_SPC_FILE)]))
    overridden_df = make_ui_dataframe(pd.DataFrame(overridden.dataframes[str(BH_SPC_FILE)]))
    assert overridden_df["Duration (ms)"].iloc[0] == native_df["Duration (ms)"].iloc[0] * 2.0


def test_summarize_bursts_matches_core_helper() -> None:
    """API burst summary output should match the existing core helper on real data."""
    tttr = load_tttr(BH_SPC_FILE)
    settings = real_data_settings()
    selected = apply_photon_filters(tttr, settings.photon_filter)
    start_stop = find_bursts(selected)
    api_df = summarize_bursts(start_stop, BH_SPC_FILE, tttr)
    core_df = generate_burst_dataframe(
        start_stop=start_stop,
        filename=BH_SPC_FILE,
        tttr=tttr,
        windows={},
        detectors={},
        include_interleaved_zeros=True,
    )
    pd.testing.assert_frame_equal(api_df, core_df)


def test_make_ui_dataframe_adds_proximity_ratio() -> None:
    """GUI adapter should create the columns used by the histogram UI."""
    df = pd.DataFrame(
        {
            "First Photon": [0],
            "Last Photon": [2],
            "Duration (ms)": [1.0],
            "Number of Photons (red)": [2],
            "Number of Photons (green)": [2],
        }
    )
    ui_df = make_ui_dataframe(df)
    assert ui_df["Proximity Ratio"].iloc[0] == 0.5


def test_make_ui_dataframe_ignores_margarita_zero_rows() -> None:
    """GUI table data should hide interleaved zero separator rows."""
    df = pd.DataFrame(
        {
            "First Photon": [0, 10, 0, 30],
            "Last Photon": [0, 19, 0, 39],
            "Duration (ms)": [0.0, 1.0, 0.0, 1.5],
            "Mean Macro Time (ms)": [0.0, 12.0, 0.0, 35.0],
            "Number of Photons": [0, 10, 0, 20],
            "Count Rate (KHz)": [0.0, 10.0, 0.0, 13.3],
            "Number of Photons (red)": [0, 4, 0, 8],
            "Number of Photons (green)": [0, 6, 0, 12],
        }
    )

    visible = burst_rows_for_display(df)
    ui_df = make_ui_dataframe(df)

    assert len(df) == 4
    assert visible["Number of Photons"].tolist() == [10, 20]
    assert ui_df["Number of Photons"].tolist() == [10, 20]
    assert ui_df["Proximity Ratio"].tolist() == [0.4, 0.4]


def test_extract_features_supports_chisurf_bur_columns() -> None:
    """Feature extraction should support core .bur display column names."""
    df = pd.DataFrame(
        {
            "Number of Photons": [10, 20],
            "Duration (ms)": [1.0, 2.0],
            "Proximity Ratio": [0.25, 0.75],
        }
    )
    features = extract_features([df])
    assert features.loc[0, "nphotons"] == 10
    assert features.loc[1, "fret"] == 0.75


def test_extract_features_and_fit_gmm() -> None:
    """Feature extraction and GMM fitting should work on burst tables."""
    df = pd.DataFrame(
        {
            "nphotons": [10, 20, 30, 40],
            "duration": [1.0, 2.0, 3.0, 4.0],
            "fret": [0.1, 0.2, 0.8, 0.9],
        }
    )
    features = extract_features([df])
    assert list(features.columns) == ["nphotons", "duration", "brightness", "interphoton", "fret"]
    fit = fit_gmm(features, GMMSettings(covariance_type="spherical"))
    assert fit["n_components"] == 1
    assert fit["labels"] == [0, 0, 0, 0]


def test_settings_roundtrip() -> None:
    """Analysis settings should serialize to JSON-compatible data and deserialize."""
    settings = AnalysisSettings()
    settings.burst_detection.min_photons = 42
    payload = to_jsonable(settings)
    restored = settings_from_dict(payload)
    assert restored.burst_detection.min_photons == 42


def test_settings_from_dict_accepts_unset_microtime_ranges() -> None:
    """Unset microtime ranges from GUI controls should deserialize as no range filter."""
    payload = {
        "photon_filter": {
            "channels": [],
            "microtime_ranges": None,
            "filter_active": True,
            "used_filter": BurstFilterMode.BURST,
            "count_rate_filter": {
                "n_ph_max": 60,
                "time_window": 0.06,
                "invert": False,
            },
            "delta_macro_time_filter": {
                "dT_min": 0.003617366409160019,
                "dT_max": 0.7234732818320043,
                "dT_min_active": False,
                "dT_max_active": True,
            },
            "invert_filter": False,
            "max_gap": 0,
            "use_gap_fill": False,
        },
        "burst_detection": {
            "min_photons": 60,
            "photon_window": 5,
            "time_window": 0.06,
        },
    }

    restored = settings_from_dict(payload)

    assert restored.photon_filter.microtime_ranges == []
    assert restored.photon_filter.used_filter == BurstFilterMode.BURST
    assert restored.burst_detection.photon_window == 5


def test_photon_filter_settings_normalizes_unset_microtime_ranges() -> None:
    """Direct GUI settings construction should treat unset microtime ranges as all photons."""
    settings = PhotonFilterSettings(
        channels=None,
        microtime_ranges=None,
    )

    assert settings.channels == []
    assert settings.microtime_ranges == []

"""GUI adapters for the Burst Selection plugin."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..api.models import (
    AnalysisSettings,
    BurstDetectionSettings,
    BurstFilterMode,
    CountRateFilterSettings,
    DeltaMacroTimeFilterSettings,
    GMMSettings,
    PhotonFilterSettings,
)
from ..api.selection import analyze_file

UI_COLUMNS = [
    "File Idx",
    "First Photon",
    "Last Photon",
    "Duration (ms)",
    "Mean Macro Time (ms)",
    "Number of Photons",
    "Count Rate (KHz)",
    "Number of Photons (red)",
    "Number of Photons (green)",
]

PROXIMITY_RATIO_COLUMN = "Proximity Ratio"


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series | None:
    """Return a numeric series for ``column`` when present."""
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def _ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio series with invalid denominators masked."""
    return (numerator / denominator.where(denominator > 0)).replace([np.inf, -np.inf], np.nan)


def proximity_ratio_from_frame(frame: pd.DataFrame) -> pd.Series | None:
    """Compute or return a proximity-ratio series when source columns exist."""
    if PROXIMITY_RATIO_COLUMN in frame.columns:
        return pd.to_numeric(frame[PROXIMITY_RATIO_COLUMN], errors="coerce")
    red = _numeric_series(frame, "Number of Photons (red)")
    green = _numeric_series(frame, "Number of Photons (green)")
    if red is not None and green is not None:
        return _ratio_series(red, red + green)
    red_rate = _numeric_series(frame, "Red Count Rate (KHz)")
    green_rate = _numeric_series(frame, "Green Count Rate (KHz)")
    if red_rate is not None and green_rate is not None:
        return _ratio_series(red_rate, red_rate + green_rate)
    return None


def analysis_settings_from_wizard(wizard: Any) -> AnalysisSettings:
    """Create API analysis settings from a BurstSelectionTool instance."""
    output_formats = ["bur"]
    if wizard.checkBox_FileMFDHDF.isChecked():
        output_formats.append("hdf5")
    settings = AnalysisSettings(
        output_formats=output_formats,
        zip_output=bool(wizard.checkBox_ZipOutput.isChecked()),
        remove_folder=bool(wizard.checkBox_RemoveFolder.isChecked()),
    )
    settings.photon_filter = photon_filter_settings_from_wizard(wizard.burst_finder)
    settings.burst_detection = BurstDetectionSettings(
        min_photons=int(wizard.burst_finder.min_ph),
        photon_window=int(wizard.burst_finder.ph_window),
        time_window=float(wizard.burst_finder.dT_max) / 1000.0,
    )
    settings.gmm = GMMSettings(
        covariance_type=str(wizard.gmm_settings["covariance_type"]),
        random_state=int(wizard.gmm_settings["random_state"]),
        max_iter=int(wizard.gmm_settings["max_iter"]),
        n_init=int(wizard.gmm_settings["n_init"]),
        tol=float(wizard.gmm_settings["tol"]),
        max_components=int(wizard.gmm_settings["max_components"]),
        reg_covar=float(wizard.gmm_settings["reg_covar"]),
        auto_components=bool(wizard.checkBox_auto_components.isChecked()),
    )
    return settings


def photon_filter_settings_from_wizard(wizard_filter: Any) -> PhotonFilterSettings:
    """Create photon filter settings from a WizardTTTRPhotonFilter instance."""
    return PhotonFilterSettings(
        channels=list(wizard_filter.channels),
        microtime_ranges=list(wizard_filter.microtime_ranges),
        filter_active=bool(wizard_filter.settings.get("filter_active", True)),
        used_filter=BurstFilterMode(str(wizard_filter.used_filter)),
        count_rate_filter=CountRateFilterSettings(
            n_ph_max=int(wizard_filter.settings["count_rate_filter"]["n_ph_max"]),
            time_window=float(wizard_filter.settings["count_rate_filter"]["time_window"]),
            invert=bool(wizard_filter.settings.get("invert_filter", False)),
        ),
        delta_macro_time_filter=DeltaMacroTimeFilterSettings(
            dT_min=float(wizard_filter.dT_min),
            dT_max=float(wizard_filter.dT_max),
            dT_min_active=bool(wizard_filter.use_lower),
            dT_max_active=bool(wizard_filter.use_upper),
        ),
        invert_filter=bool(wizard_filter.settings.get("invert_filter", False)),
        max_gap=int(wizard_filter.max_gap),
        use_gap_fill=bool(wizard_filter.use_gap_fill),
    )


def save_current_selection(
    wizard: Any,
    output_types: set[str],
    zip_output: bool,
    remove_folder: bool,
) -> None:
    """Save the current wizard photon/burst selection using the existing GUI path."""
    wizard.burst_finder.save_selection(
        output_types=output_types,
        zip_output=zip_output,
        remove_folder=remove_folder,
    )


def bur_file_path(file_path: str | Path, target_path: str) -> Path:
    """Return the direct ``.bur`` path used by the current GUI."""
    file_path = Path(file_path)
    return file_path.parent / target_path / "bi4_bur" / f"{file_path.stem}.bur"


def load_burst_dataframe(file_path: str | Path, target_path: str) -> pd.DataFrame | None:
    """Load a saved ``.bur`` file, including zip fallback paths used by the GUI."""
    file_path = Path(file_path)
    direct = bur_file_path(file_path, target_path)
    if direct.exists():
        return pd.read_csv(direct, sep="\t")

    zip_file_path = file_path.parent / target_path / f"{target_path}.zip"
    alt_zip_paths = [
        file_path.parent / f"{target_path}.zip",
        file_path.parent / target_path / "output.zip",
        file_path.parent / "output.zip",
    ]
    for alt_path in alt_zip_paths:
        if alt_path.exists():
            zip_file_path = alt_path
            break
    else:
        if not zip_file_path.exists():
            return None

    bur_filenames = [
        f"bi4_bur/{file_path.stem}.bur",
        f"bur/{file_path.stem}.bur",
        f"{file_path.stem}.bur",
        f"{target_path}/bi4_bur/{file_path.stem}.bur",
        f"{target_path}/bur/{file_path.stem}.bur",
    ]
    with zipfile.ZipFile(str(zip_file_path), "r") as zip_file:
        all_files = zip_file.namelist()
        for bur_filename in bur_filenames:
            try:
                with zip_file.open(bur_filename) as bur_file:
                    return pd.read_csv(io.TextIOWrapper(bur_file), sep="\t")
            except KeyError:
                continue
        matching = [name for name in all_files if name.endswith(f"{file_path.stem}.bur")]
        if matching:
            with zip_file.open(matching[0]) as bur_file:
                return pd.read_csv(io.TextIOWrapper(bur_file), sep="\t")
    return None


def make_ui_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create the limited DataFrame currently shown by the GUI."""
    ui_df = burst_rows_for_display(df)
    for column in UI_COLUMNS:
        if column not in ui_df.columns:
            ui_df[column] = 0
    ui_df = ui_df[UI_COLUMNS].copy()
    proximity_ratio = proximity_ratio_from_frame(ui_df)
    if proximity_ratio is not None:
        ui_df[PROXIMITY_RATIO_COLUMN] = proximity_ratio.fillna(0).round(6)
    return ui_df


def burst_rows_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return burst rows for GUI display, excluding Margarita zero separators.

    The ChiSurf/Margarita ``.bur`` format writes interleaved all-zero rows for
    compatibility. Those rows must remain in files, but they should not appear
    in tables, histograms, or GMM inputs.
    """
    if df.empty:
        return df.copy()
    if "Number of Photons" in df.columns:
        n_photons = pd.to_numeric(df["Number of Photons"], errors="coerce").fillna(0)
        return df.loc[n_photons > 0].copy()
    if {"Number of Photons (red)", "Number of Photons (green)"} <= set(df.columns):
        red = pd.to_numeric(df["Number of Photons (red)"], errors="coerce").fillna(0)
        green = pd.to_numeric(df["Number of Photons (green)"], errors="coerce").fillna(0)
        return df.loc[(red + green) > 0].copy()
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return df.copy()
    return df.loc[~numeric.fillna(0).eq(0).all(axis=1)].copy()


def combine_ui_dataframes(frames: Iterable[pd.DataFrame]) -> pd.DataFrame | None:
    """Concatenate GUI DataFrames when any frames contain rows."""
    frames = list(frames)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def analyze_file_for_wizard(
    path: str | Path,
    wizard: Any,
    output_dir: str | Path | None = None,
) -> Any:
    """Run the shared Burst Selection API for one file using wizard settings."""
    settings = analysis_settings_from_wizard(wizard)
    return analyze_file(
        path,
        settings=settings,
        filetype=None,
        windows=wizard.burst_finder.windows,
        detectors=wizard.burst_finder.detectors,
        output_dir=output_dir,
    )


def gmm_settings_from_wizard(wizard: Any) -> dict[str, Any]:
    """Return the GUI GMM settings dictionary."""
    return dict(wizard.gmm_settings)


def selected_histogram_data(current_df: pd.DataFrame, selected_feature: str) -> np.ndarray:
    """Return numeric histogram data for the selected GUI feature."""
    if selected_feature == PROXIMITY_RATIO_COLUMN:
        data = proximity_ratio_from_frame(current_df)
        if data is not None:
            return data.dropna().to_numpy(dtype=float)
    data = pd.to_numeric(burst_rows_for_display(current_df)[selected_feature], errors="coerce").dropna()
    return data.to_numpy(dtype=float)

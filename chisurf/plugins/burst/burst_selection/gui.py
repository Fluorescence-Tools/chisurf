"""Compatibility GUI adapter module for Burst Selection."""

from .gui.adapter import (
    UI_COLUMNS,
    analyze_file_for_wizard,
    bur_file_path,
    combine_ui_dataframes,
    gmm_settings_from_wizard,
    load_burst_dataframe,
    make_ui_dataframe,
    photon_filter_settings_from_wizard,
    save_current_selection,
    selected_histogram_data,
)

__all__ = [
    "UI_COLUMNS",
    "analyze_file_for_wizard",
    "bur_file_path",
    "combine_ui_dataframes",
    "gmm_settings_from_wizard",
    "load_burst_dataframe",
    "make_ui_dataframe",
    "photon_filter_settings_from_wizard",
    "save_current_selection",
    "selected_histogram_data",
]

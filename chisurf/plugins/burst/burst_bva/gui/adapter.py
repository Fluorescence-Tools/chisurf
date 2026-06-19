"""GUI adapters for the BVA plugin."""

from __future__ import annotations

from typing import Any

from ..api.models import BvaSettings


def bva_settings_from_tool(tool: Any) -> dict[str, Any]:
    """Extract JSON-compatible BVA settings from a BVATool instance."""
    return {
        "donor_channels": tool.bva_settings.get("donor_channels", [0, 8]),
        "donor_micro_time_ranges": tool.bva_settings.get("donor_micro_time_ranges", [(0, 32768)]),
        "acceptor_channels": tool.bva_settings.get("acceptor_channels", [1, 9]),
        "acceptor_micro_time_ranges": tool.bva_settings.get("acceptor_micro_time_ranges", [(0, 32768)]),
        "minimum_window_length": tool.bva_settings.get("minimum_window_length", 0.01),
        "number_of_photons_per_slice": tool.bva_settings.get("number_of_photons_per_slice", 10),
        "file_type": getattr(tool, "file_type", "SPC-130"),
    }

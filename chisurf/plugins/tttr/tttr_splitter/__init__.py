"""TTTR Split / Convert plugin.

This plugin provides functionality for splitting large TTTR (Time-Tagged Time-Resolved)
files, particularly those in the PicoQuant PTU format, into smaller segments. This is
useful for:

1. Breaking down large datasets into manageable chunks
2. Extracting specific time segments from long measurements
3. Creating subsets of data for parallel processing
4. Reducing memory requirements for analysis

The plugin features:
- Support for various TTTR file formats
- Configurable splitting parameters (photons per file, time segments)
- Options for micro-time binning to reduce file size
- Ability to reset macro-times in the output files
- Selection of output container formats (file format conversion)

This tool is particularly valuable for handling large datasets from long-duration
single-molecule or imaging experiments, making them more manageable for subsequent analysis.
"""

name = "TTTR:Editor:Split/Convert"

# Aggregated into the TTTR Tools toolbox (tttr_toolbox); hidden as a top-level
# menu entry but still importable and standalone-launchable.
menu_hidden = True

__all__ = ["PTUSplitter"]


def __getattr__(attr_name: str):
    """Lazy Qt import gate (no Qt import as a package side effect)."""
    if attr_name == "PTUSplitter":
        from .gui.tool import PTUSplitter as _cls

        globals()["PTUSplitter"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


if __name__ == "plugin":
    from .gui.tool import PTUSplitter

    window = PTUSplitter()
    window.show()

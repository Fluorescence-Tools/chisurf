"""PTU Header Editor plugin.

This plugin provides a comprehensive tool for viewing and editing header tags in
PicoQuant Time-Tagged Time-Resolved (PTU) files. It allows users to:
1. Open and inspect PTU files
2. View all header tags and their values in a tabular format
3. Edit existing tag values
4. Add new custom tags to the header
5. Remove unwanted tags
6. Save the modified PTU file with updated header information

The plugin features an intuitive table-based interface that displays tag names,
types, values, and indices. Users can modify any field and see the changes in
real-time. A JSON view is also available to see the complete header structure.

This tool is particularly useful for:
- Correcting metadata in experimental PTU files
- Adding missing information to PTU headers
- Preparing files for specialized analysis
- Troubleshooting issues with file metadata
- Educational purposes to understand PTU file structure

The plugin preserves all event data from the original file while allowing
complete customization of the header information.
"""

name = "TTTR:Editor:PTU Header editor"

# Aggregated into the TTTR Tools toolbox (tttr_toolbox); hidden as a top-level
# menu entry but still importable and standalone-launchable.
menu_hidden = True

__all__ = ["TagsEditor"]


def __getattr__(attr_name: str):
    """Lazy Qt import gate (no Qt import as a package side effect)."""
    if attr_name == "TagsEditor":
        from .gui.tool import TagsEditor as _cls

        globals()["TagsEditor"] = _cls
        return _cls
    raise AttributeError(f"module {__name__!r} has no attribute {attr_name!r}")


if __name__ == "plugin":
    from .gui.tool import TagsEditor

    window = TagsEditor()
    window.show()

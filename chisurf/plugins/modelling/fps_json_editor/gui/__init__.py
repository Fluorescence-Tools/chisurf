"""GUI package for the FPS JSON Editor plugin."""

__all__ = ["FpsJsonEditor", "FpsJsonEditorTool"]


def __getattr__(name: str):
    """Lazily expose GUI entrypoint classes."""
    if name == "FpsJsonEditor":
        from .editor import FpsJsonEditor

        return FpsJsonEditor
    if name == "FpsJsonEditorTool":
        from .tool import FpsJsonEditorTool

        return FpsJsonEditorTool
    raise AttributeError(name)

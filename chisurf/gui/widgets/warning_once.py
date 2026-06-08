"""Utility to show a QMessageBox warning at most once per session.

Use `show_warning_once` to avoid showing the same warning dialog
multiple times when it is triggered from multiple call sites during
initialization (e.g. several widgets that independently check for a
missing file).
"""

from qtpy.QtWidgets import QMessageBox

_shown_warnings: set[str] = set()


def show_warning_once(
    key: str,
    title: str,
    text: str,
    informative_text: str = "",
    details: str = "",
    icon: QMessageBox.Icon = QMessageBox.Warning,
    **kwargs,
) -> None:
    """Show a QMessageBox warning only once per session for a given *key*.

    Parameters
    ----------
    key : str
        Unique identifier for this warning (e.g. ``"missing_detector_setups"``).
        Subsequent calls with the same key are silently ignored.
    title : str
        Window title of the message box.
    text : str
        Main message text.
    informative_text : str, optional
        Additional descriptive text.
    details : str, optional
        Expandable details text.
    icon : QMessageBox.Icon, optional
        Icon to display (default ``QMessageBox.Warning``).
    **kwargs
        Additional keyword arguments forwarded to ``QMessageBox``.
    """
    if key in _shown_warnings:
        return
    _shown_warnings.add(key)

    msg = QMessageBox(**kwargs)
    msg.setWindowTitle(title)
    msg.setIcon(icon)
    msg.setText(text)
    if informative_text:
        msg.setInformativeText(informative_text)
    if details:
        msg.setDetailedText(details)
    msg.exec_()


def mark_warning_shown(key: str) -> None:
    """Manually mark a warning key as already shown.

    Use this when you need to build a custom ``QMessageBox`` (with extra
    buttons, a checkbox, etc.) but still want the once-per-session
    guarantee.
    """
    _shown_warnings.add(key)


def was_warning_shown(key: str) -> bool:
    """Return True if *key* has already been shown this session."""
    return key in _shown_warnings


def reset_warnings() -> None:
    """Clear all previously-shown warning keys (useful for testing)."""
    _shown_warnings.clear()

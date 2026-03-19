from __future__ import annotations

import functools
from typing import Any, Callable, Optional

from qtpy import QtCore


class SignalBlocker:
    """Context manager to temporarily block signals on a Qt widget.
    
    Usage:
        with SignalBlocker(widget):
            widget.setValue(123)
    """

    def __init__(self, widget: Any):
        self.widget = widget
        self._was_enabled = None

    def __enter__(self):
        self._was_enabled = self.widget.signalsBlocked()
        self.widget.blockSignals(True)
        return self.widget

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.widget.blockSignals(self._was_enabled)
        return False


def block_signals(*widgets: Any) -> SignalBlocker:
    """Create a SignalBlocker for multiple widgets."""
    if len(widgets) == 1:
        return SignalBlocker(widgets[0])
    else:
        return _MultiSignalBlocker(widgets)


class _MultiSignalBlocker:
    """Context manager to temporarily block signals on multiple Qt widgets."""

    def __init__(self, widgets: tuple):
        self.widgets = widgets
        self._was_enabled = [w.signalsBlocked() for w in widgets]

    def __enter__(self):
        for w in self.widgets:
            w.blockSignals(True)
        return self.widgets

    def __exit__(self, exc_type, exc_val, exc_tb):
        for w, was_enabled in zip(self.widgets, self._was_enabled):
            w.blockSignals(was_enabled)
        return False


class ReentrancyGuard:
    """Decorator/context manager to prevent reentrant execution.
    
    Usage as decorator:
        @ReentrancyGuard()
        def my_method(self):
            ...
    
    Usage as context manager:
        guard = ReentrancyGuard()
        with guard:
            ...
    """

    def __init__(self, key: Optional[str] = None):
        self.key = key or "default"
        self._lock_attr = f"_reentrancy_guard_{self.key}"

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            guard = getattr(args[0], self._lock_attr, None)
            if guard is None:
                guard = ReentrancyGuard(self.key)
                setattr(args[0], self._lock_attr, guard)
            with guard:
                return func(*args, **kwargs)
        return wrapper

    def __enter__(self):
        obj = getattr(self, "_obj", None)
        if obj is None:
            return self
        lock = getattr(obj, self._lock_attr, False)
        if lock:
            raise RuntimeError(f"Reentrancy detected for {self.key}")
        setattr(obj, self._lock_attr, True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        obj = getattr(self, "_obj", None)
        if obj is not None:
            setattr(obj, self._lock_attr, False)
        return False

    def _set_obj(self, obj: Any):
        self._obj = obj


def connected_signals_blocked(widget: Any, signal_name: str) -> bool:
    """Check if a signal is connected and blocked."""
    try:
        signal = getattr(widget, signal_name, None)
        if signal is None:
            return False
        return widget.signalsBlocked()
    except AttributeError:
        return False


class UiSyncMixin:
    """Mixin class to provide common UI synchronization patterns.
    
    Subclasses should implement:
        - updateUI(): reads from reader, updates widgets
        - onParametersChanged(): reads from widgets, updates reader
    
    The mixin provides:
        - _sync_ui_from_reader(): calls updateUI with signal blocking
        - _sync_reader_from_ui(): calls onParametersChanged with guard
    """

    _ui_sync_in_progress: bool = False

    def _sync_ui_from_reader(self, *widget_names: str) -> None:
        """Call updateUI with signals blocked for specified widgets.
        
        Args:
            *widget_names: Names of widget attributes to block signals on.
                          If empty, blocks signals on self.
        """
        if getattr(self, "_ui_sync_in_progress", False):
            return
        
        self._ui_sync_in_progress = True
        try:
            widgets_to_block = []
            if widget_names:
                for name in widget_names:
                    w = getattr(self, name, None)
                    if w is not None:
                        widgets_to_block.append(w)
            else:
                widgets_to_block = [self]
            
            with block_signals(*widgets_to_block):
                self.updateUI()
        finally:
            self._ui_sync_in_progress = False

    def _sync_reader_from_ui(self) -> None:
        """Call onParametersChanged with reentrancy guard."""
        if getattr(self, "_sync_in_progress", False):
            return
        
        self._sync_in_progress = True
        try:
            self.onParametersChanged()
        finally:
            self._sync_in_progress = False


__all__ = [
    "SignalBlocker",
    "block_signals",
    "ReentrancyGuard",
    "UiSyncMixin",
]

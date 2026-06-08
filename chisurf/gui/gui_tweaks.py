from __future__ import annotations
import chisurf as cs

import os


def apply_platform_window_tweaks(window) -> None:
    """Apply small, non-invasive tweaks to a top-level window frame.

    Currently this enables a dark titlebar on supported Windows
    versions using the DWM "immersive dark mode" attribute, so the
    outer chrome looks less like a stock bright Windows app while
    retaining native move/resize/snap behavior.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return

    # Obtain the native HWND for this top-level window.
    try:
        hwnd = int(window.winId())
    except Exception:
        return

    try:
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20  # Windows 10 1809+
        value = ctypes.c_int(1)
        dwmapi = ctypes.windll.dwmapi

        def _set_attr(attr_id: int) -> bool:
            try:
                res = dwmapi.DwmSetWindowAttribute(
                    wintypes.HWND(hwnd),
                    ctypes.c_uint(attr_id),
                    ctypes.byref(value),
                    ctypes.sizeof(value),
                )
                return res == 0
            except Exception:
                return False

        if not _set_attr(DWMWA_USE_IMMERSIVE_DARK_MODE):
            # Older builds used 19 for the same attribute; try as a
            # best-effort fallback.
            _set_attr(19)
    except Exception:
        # If anything goes wrong, silently fall back to the default
        # system chrome rather than risking a broken window frame.
        return


def apply_dock_tab_colors(window) -> None:
    try:
        from qtpy import QtWidgets, QtGui
    except Exception:
        return
    try:
        pass
    except Exception:
        return
    try:
        gui_cfg = getattr(cs.core.settings, "gui", {})
    except Exception:
        gui_cfg = {}
    if not isinstance(gui_cfg, dict):
        return
    try:
        hex_by_title = gui_cfg.get("dock_tab_colors")
    except Exception:
        hex_by_title = None
    if not isinstance(hex_by_title, dict) or not hex_by_title:
        return
    color_by_title = {}
    for title, value in hex_by_title.items():
        color = None
        if isinstance(value, str):
            c = QtGui.QColor(value)
            if c.isValid():
                color = c
        elif isinstance(value, (tuple, list)) and len(value) >= 3:
            try:
                r, g, b = (int(value[0]), int(value[1]), int(value[2]))
                c = QtGui.QColor(r, g, b)
                if c.isValid():
                    color = c
            except Exception:
                color = None
        if color is not None:
            color_by_title[str(title)] = color
    if not color_by_title:
        return
    try:
        tab_bars = window.findChildren(QtWidgets.QTabBar)
    except Exception:
        tab_bars = []
    for tabbar in tab_bars:
        try:
            count = tabbar.count()
        except Exception:
            continue
        for i in range(count):
            try:
                title = tabbar.tabText(i)
            except Exception:
                continue
            color = color_by_title.get(title)
            if color is not None:
                try:
                    tabbar.setTabTextColor(i, color)
                except Exception:
                    pass


def apply_pyqtgraph_autorange_compat(pg) -> None:
    try:
        from pyqtgraph.widgets.PlotWidget import PlotWidget as _CsPlotWidget
    except Exception:
        return
    try:
        if hasattr(_CsPlotWidget, "autoRangeEnabled"):
            return

        def _chisurf_pg_autorange_enabled_compat(self):
            vb = None
            try:
                vb = self.getViewBox()
            except Exception:
                vb = None
            if vb is not None and hasattr(vb, "autoRangeEnabled"):
                try:
                    return vb.autoRangeEnabled()
                except Exception:
                    pass
            return (True, True)

        _CsPlotWidget.autoRangeEnabled = _chisurf_pg_autorange_enabled_compat
    except Exception:
        return

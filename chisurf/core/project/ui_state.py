from __future__ import annotations

import typing


def get_ui_state(main_window: typing.Any) -> typing.Dict[str, typing.Any]:
    """Capture UI state from main window and all sub-windows.

    Returns a dict with:
    - main_window: geometry and dock state
    - mdi_area: MDI subwindow layout
    - dataset_selector: current selection and expanded state
    - fit_selector: current selection
    - active_tabs: which tabs are active in each panel
    """
    state: typing.Dict[str, typing.Any] = {}

    if main_window is None:
        return state

    try:
        save_geom = getattr(main_window, "saveGeometry", None)
        if callable(save_geom):
            try:
                ba = save_geom()
                state["geometry"] = bytes(ba).hex()
            except Exception:
                pass
    except Exception:
        pass

    try:
        save_state = getattr(main_window, "saveState", None)
        if callable(save_state):
            try:
                ba = save_state()
                state["dock_state"] = bytes(ba).hex()
            except Exception:
                pass
    except Exception:
        pass

    try:
        mdi = getattr(main_window, "mdiarea", None)
        if mdi is not None:
            save_mdi = getattr(mdi, "saveState", None)
            if callable(save_mdi):
                try:
                    ba = save_mdi()
                    state["mdi_area"] = {"state": bytes(ba).hex()}
                except Exception:
                    pass
    except Exception:
        pass

    try:
        history_browser = getattr(main_window, "historyBrowser", None)
        get_hist_state = getattr(history_browser, "get_ui_state", None)
        if callable(get_hist_state):
            try:
                state["history_browser"] = get_hist_state()
            except Exception:
                pass
    except Exception:
        pass

    return state


def set_ui_state(main_window: typing.Any, state: typing.Dict[str, typing.Any]) -> bool:
    """Apply UI state to main window.

    Returns True if successful, False otherwise.
    """
    if main_window is None:
        return False

    success = False

    try:
        geom_hex = state.get("geometry")
        if geom_hex:
            restore_geom = getattr(main_window, "restoreGeometry", None)
            if callable(restore_geom):
                try:
                    geom_bytes = bytes.fromhex(geom_hex)
                    restore_geom(geom_bytes)
                    success = True
                except Exception:
                    pass
    except Exception:
        pass

    try:
        dock_hex = state.get("dock_state")
        if dock_hex:
            restore_state = getattr(main_window, "restoreState", None)
            if callable(restore_state):
                try:
                    state_bytes = bytes.fromhex(dock_hex)
                    restore_state(state_bytes)
                    success = True
                except Exception:
                    pass
    except Exception:
        pass

    try:
        mdi_state = state.get("mdi_area", {})
        mdi_hex = mdi_state.get("state")
        if mdi_hex:
            mdi = getattr(main_window, "mdiarea", None)
            if mdi is not None:
                restore_mdi = getattr(mdi, "restoreState", None)
                if callable(restore_mdi):
                    try:
                        mdi_bytes = bytes.fromhex(mdi_hex)
                        restore_mdi(mdi_bytes)
                        success = True
                    except Exception:
                        pass
    except Exception:
        pass

    return success


def get_dataset_selector_state(main_window: typing.Any) -> typing.Dict[str, typing.Any]:
    """Get dataset selector state (selection, expanded groups)."""
    state: typing.Dict[str, typing.Any] = {}

    if main_window is None:
        return state

    try:
        ds_widget = getattr(main_window, "datasetWidget", None)
        if ds_widget is not None:
            get_state = getattr(ds_widget, "get_selection_state", None)
            if callable(get_state):
                try:
                    state = get_state()
                except Exception:
                    pass
    except Exception:
        pass

    return state


def set_dataset_selector_state(main_window: typing.Any, state: typing.Dict[str, typing.Any]) -> bool:
    """Apply dataset selector state."""
    if main_window is None:
        return False

    try:
        ds_widget = getattr(main_window, "datasetWidget", None)
        if ds_widget is not None:
            set_state = getattr(ds_widget, "set_selection_state", None)
            if callable(set_state):
                try:
                    set_state(state)
                    return True
                except Exception:
                    pass
    except Exception:
        pass

    return False


def get_fit_selector_state(main_window: typing.Any) -> typing.Dict[str, typing.Any]:
    """Get fit selector state (selected fit group, selected local fit)."""
    state: typing.Dict[str, typing.Any] = {}

    if main_window is None:
        return state

    try:
        fit_widget = getattr(main_window, "fitWidget", None)
        if fit_widget is not None:
            get_state = getattr(fit_widget, "get_selection_state", None)
            if callable(get_state):
                try:
                    state = get_state()
                except Exception:
                    pass
    except Exception:
        pass

    return state


def set_fit_selector_state(main_window: typing.Any, state: typing.Dict[str, typing.Any]) -> bool:
    """Apply fit selector state."""
    if main_window is None:
        return False

    try:
        fit_widget = getattr(main_window, "fitWidget", None)
        if fit_widget is not None:
            set_state = getattr(fit_widget, "set_selection_state", None)
            if callable(set_state):
                try:
                    set_state(state)
                    return True
                except Exception:
                    pass
    except Exception:
        pass

    return False


def get_active_tabs(main_window: typing.Any) -> typing.Dict[str, int]:
    """Get active tab indices for main panels."""
    tabs: typing.Dict[str, int] = {}

    if main_window is None:
        return tabs

    panel_names = [
        "datasetPanel",
        "experimentPanel", 
        "analysisPanel",
        "plotPanel",
    ]

    for name in panel_names:
        try:
            panel = getattr(main_window, name, None)
            if panel is not None:
                current_idx = getattr(panel, "currentIndex", None)
                if callable(current_idx):
                    tabs[name] = current_idx()
        except Exception:
            pass

    return tabs


def set_active_tabs(main_window: typing.Any, tabs: typing.Dict[str, int]) -> None:
    """Set active tab indices for main panels."""
    if main_window is None:
        return

    for name, idx in tabs.items():
        try:
            panel = getattr(main_window, name, None)
            if panel is not None:
                set_idx = getattr(panel, "setCurrentIndex", None)
                if callable(set_idx):
                    set_idx(idx)
        except Exception:
            pass

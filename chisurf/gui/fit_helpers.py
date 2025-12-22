from __future__ import annotations

import pathlib
from typing import Optional, Any

import chisurf
from chisurf.gui import QtCore


def close_all_fits(main_window) -> None:
    """Close all fit windows and clear related UI layouts."""
    old_confirm = chisurf.settings.gui.get('confirm_close_fit', True)
    try:
        chisurf.settings.gui['confirm_close_fit'] = False
    except Exception:
        old_confirm = None

    try:
        for sub_window in list(chisurf.gui.fit_windows):
            try:
                setattr(sub_window, 'close_confirm', False)
            except Exception:
                pass
            try:
                widget = sub_window.widget()
                if widget is not None:
                    setattr(widget, 'close_confirm', False)
            except Exception:
                pass
            try:
                sub_window.close()
            except Exception:
                pass

        chisurf.fits.clear()
        chisurf.gui.fit_windows.clear()
    finally:
        if old_confirm is not None:
            try:
                chisurf.settings.gui['confirm_close_fit'] = old_confirm
            except Exception:
                pass

    chisurf.gui.widgets.clear_layout(main_window.modelLayout)
    header_layout = getattr(main_window, "analysisHeaderLayout", None)
    if header_layout is not None:
        chisurf.gui.widgets.clear_layout(header_layout)
    chisurf.gui.widgets.clear_layout(main_window.plotOptionsLayout)


def save_fits(main_window, event: Optional[QtCore.QEvent] = None) -> None:
    """Prompt for a directory and save all fits via the macro."""
    path, _ = chisurf.gui.widgets.get_directory()
    if not path:
        return
    chisurf.working_path = path
    chisurf.run(f'chisurf.macros.save_fits(target_path=r"{path.as_posix()}")')


def save_fit(main_window, event: Optional[QtCore.QEvent] = None, **kwargs: Any) -> None:
    """Prompt for a directory and save the current fit via the macro."""
    try:
        default_dir = None
        fit = getattr(main_window, 'current_fit', None)
        data_obj = getattr(fit, 'data', None) if fit is not None else None
        filename = getattr(data_obj, 'filename', None) if data_obj is not None else None
        if isinstance(filename, str):
            fn = filename.strip()
            if fn and fn.lower() != 'none':
                p = pathlib.Path(fn)
                if p.is_absolute():
                    default_dir = p.parent
        if ('directory' not in kwargs or kwargs.get('directory') is None) and default_dir is not None:
            kwargs['directory'] = default_dir
    except Exception as e:
        chisurf.logging.warning(f"save_fit: could not infer data folder from fit.data.filename: {e}")

    path, _ = chisurf.gui.widgets.get_directory(**kwargs)
    if not path:
        return
    chisurf.working_path = path
    chisurf.run(f'chisurf.macros.save_fit(target_path=r"{path.as_posix()}")')

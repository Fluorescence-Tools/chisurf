from __future__ import annotations

import pathlib
from typing import Optional

import chisurf
import chisurf.macros.core_fit
from chisurf.gui import QtWidgets, QtCore


def save_project(main_window: QtWidgets.QMainWindow, event: Optional[QtCore.QEvent] = None) -> None:
    """
    Save the current project using the existing project directory if available,
    otherwise fall back to the Save As flow.
    """
    # Inform user about experimental status
    chisurf.gui.widgets.general.MyMessageBox(
        label="Project Save",
        info="Saving current session as a project. This feature is experimental.",
        show_fortune=False,
    )

    current_dir = getattr(main_window, "_current_project_dir", None)
    if isinstance(current_dir, pathlib.Path) and current_dir.is_dir():
        try:
            chisurf.working_path = current_dir.parent
        except Exception:
            pass

        try:
            chisurf.macros.core_fit.save_project(
                target_path=current_dir.parent.as_posix(),
                project_name=current_dir.name,
            )
        except Exception:
            return

        try:
            if (current_dir / "project.json").exists():
                main_window.add_recent_project(current_dir)
        except Exception:
            pass
        return

    save_project_as(main_window, event=event)


def save_project_as(main_window: QtWidgets.QMainWindow, event: Optional[QtCore.QEvent] = None) -> None:
    """Prompt for a project folder/name and save the current session."""
    path, _ = chisurf.gui.widgets.get_directory()
    if not path:
        return

    project_name, ok = QtWidgets.QInputDialog.getText(
        main_window,
        "Save Project As",
        "Project name:",
        QtWidgets.QLineEdit.Normal,
        "chisurf_project",
    )
    if not ok or not project_name:
        return

    project_dir = path / project_name
    try:
        needs_confirm = False
        if project_dir.exists():
            needs_confirm = True
        if (project_dir / "project.json").exists():
            needs_confirm = True
    except Exception:
        needs_confirm = False

    if needs_confirm:
        try:
            result = QtWidgets.QMessageBox.question(
                main_window,
                "Overwrite Project?",
                f"The project folder already exists:\n\n{project_dir}\n\nOverwrite it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
        except Exception:
            result = QtWidgets.QMessageBox.No
        if result != QtWidgets.QMessageBox.Yes:
            return

    try:
        chisurf.working_path = path
    except Exception:
        pass

    try:
        chisurf.macros.core_fit.save_project(target_path=path.as_posix(), project_name=project_name)
    except Exception:
        return
    try:
        if (project_dir / "project.json").exists():
            main_window._current_project_dir = project_dir
    except Exception:
        pass

    try:
        main_window.add_recent_project(project_dir)
    except Exception:
        pass

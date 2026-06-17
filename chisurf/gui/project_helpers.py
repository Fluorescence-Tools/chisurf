from __future__ import annotations

import json
import os
import pathlib

import chisurf as cs
from chisurf import logging
from chisurf.gui import QtCore, QtWidgets, run_on_gui_thread


def _recent_projects_file() -> pathlib.Path:
    """Return the path to the persisted recent-projects JSON file.

    Returns
    -------
    pathlib.Path
        ``~/.chisurf/recent_projects.json`` (via ``get_path``) or a
        plain ``~/.chisurf/recent_projects.json`` fallback.
    """
    try:
        return cs.core.settings.get_path("settings") / "recent_projects.json"
    except Exception:
        return pathlib.Path.home() / ".chisurf" / "recent_projects.json"


def load_recent_projects() -> list[str]:
    fp = _recent_projects_file()
    try:
        if not fp.is_file():
            return []
    except Exception:
        return []
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = data.get("projects", []) if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for it in items:
        try:
            s = str(it)
        except Exception:
            continue
        if s:
            out.append(s)
    return out


def store_recent_projects(projects: list[str]) -> None:
    fp = _recent_projects_file()
    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    payload = {"projects": list(projects or [])}
    try:
        fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        try:
            logging.exception(f"Failed to write recent projects file: {fp}")
        except Exception:
            pass


def set_recent_projects(window, projects: list[str]) -> None:
    try:
        window._recent_projects = list(projects or [])
    except Exception:
        window._recent_projects = []


def add_recent_project(window, project_path) -> None:
    try:
        p = os.path.abspath(os.path.normpath(str(project_path)))
    except Exception:
        return
    if not p:
        return

    try:
        current = list(getattr(window, "_recent_projects", []) or [])
    except Exception:
        current = []

    def _key(s: str) -> str:
        try:
            s2 = os.path.normpath(str(s))
        except Exception:
            s2 = str(s)
        return s2.lower() if os.name == "nt" else s2

    seen = set()
    new_list: list[str] = []
    for item in [p] + current:
        if not item:
            continue
        k = _key(item)
        if k in seen:
            continue
        seen.add(k)
        new_list.append(item)

    max_n = 10
    new_list = new_list[:max_n]
    set_recent_projects(window, new_list)
    store_recent_projects(new_list)
    refresh_recent_projects_menu(window)


def clear_recent_projects(window) -> None:
    set_recent_projects(window, [])
    store_recent_projects([])
    refresh_recent_projects_menu(window)


def open_recent_project(window, project_path: str) -> None:
    try:
        path = pathlib.Path(project_path)
    except Exception:
        return

    try:
        if path.is_dir():
            csp_path = path / "project.csp"
            if not csp_path.is_file():
                QtWidgets.QMessageBox.warning(
                    window,
                    "Invalid Project",
                    "The selected folder does not contain a project archive (project.csp).",
                )
                try:
                    current = list(getattr(window, "_recent_projects", []) or [])
                    current = [p for p in current if os.path.normpath(p) != os.path.normpath(project_path)]
                    set_recent_projects(window, current)
                    store_recent_projects(current)
                    refresh_recent_projects_menu(window)
                except Exception:
                    pass
                return
            path = csp_path
        elif path.suffix.lower() != ".csp":
            QtWidgets.QMessageBox.warning(
                window,
                "Invalid Project",
                "Please select a ChiSurf project archive (*.csp).",
            )
            try:
                current = list(getattr(window, "_recent_projects", []) or [])
                current = [p for p in current if os.path.normpath(p) != os.path.normpath(project_path)]
                set_recent_projects(window, current)
                store_recent_projects(current)
                refresh_recent_projects_menu(window)
            except Exception:
                pass
            return
        if not path.is_file():
            try:
                current = list(getattr(window, "_recent_projects", []) or [])
                current = [p for p in current if os.path.normpath(p) != os.path.normpath(project_path)]
                set_recent_projects(window, current)
                store_recent_projects(current)
                refresh_recent_projects_menu(window)
            except Exception:
                pass
            return
    except Exception:
        return

    try:
        cs.working_path = path.parent
    except Exception:
        pass

    try:
        cs.core.actions.dispatch(
            name="project.load",
            payload={"project_path": path.as_posix()},
        )
    except Exception:
        try:
            logging.exception(f"Failed to load recent project: {path}")
        except Exception:
            pass
        return

    try:
        window._current_project_path = path
    except Exception:
        pass

    add_recent_project(window, path)


def refresh_recent_projects_menu(window) -> None:
    """Rebuild the hidden-menubar Recent Projects menu and sync the ribbon button.

    Parameters
    ----------
    window : QMainWindow
        The application main window.
    """
    # Widget access must happen on the GUI thread. If this function is called
    # from a worker thread (e.g. project auto-save during sampling), reschedule
    # it on the GUI thread and return immediately.
    try:
        app = QtWidgets.QApplication.instance()
        if app is not None and QtCore.QThread.currentThread() is not app.thread():
            run_on_gui_thread(refresh_recent_projects_menu, window)
            return
    except Exception:
        pass

    menu = getattr(window, "_menu_recent_projects", None)
    if menu is not None:
        try:
            menu.clear()
        except Exception:
            pass

        try:
            projects = list(getattr(window, "_recent_projects", []) or [])
        except Exception:
            projects = []

        if not projects:
            try:
                a = QtWidgets.QAction("No recent projects", window)
                a.setEnabled(False)
                menu.addAction(a)
            except Exception:
                pass
        else:
            for i, p in enumerate(projects):
                try:
                    label = f"&{i + 1} {p}"
                    a = QtWidgets.QAction(label, window)
                    a.triggered.connect(
                        lambda _checked=False, pp=p: open_recent_project(window, pp)
                    )
                    menu.addAction(a)
                except Exception:
                    continue

        try:
            menu.addSeparator()
        except Exception:
            pass
        try:
            clear_action = QtWidgets.QAction("Clear Recent Projects", window)
            clear_action.triggered.connect(
                lambda _checked=False: clear_recent_projects(window)
            )
            menu.addAction(clear_action)
        except Exception:
            pass

    # Also refresh the ribbon's dedicated Recent Projects drop-down (if present).
    try:
        integration = getattr(window, "_ribbon_integration", None)
        if integration is not None and hasattr(integration, "_refresh_ribbon_recent_projects"):
            integration._refresh_ribbon_recent_projects()
    except Exception:
        pass


def init_recent_projects_menu(window) -> None:
    """Create and insert the Recent Projects submenu into the normal File menu.

    The submenu is inserted into ``menuFile`` before the Exit separator so it
    appears as ``File > Recent Projects`` in the traditional menu bar.  It is
    also refreshed whenever a project is opened/saved/cleared via
    :func:`refresh_recent_projects_menu`.

    Parameters
    ----------
    window : QMainWindow
        The application main window.
    """
    if getattr(window, "_menu_recent_projects", None) is not None:
        refresh_recent_projects_menu(window)
        return

    try:
        recent_menu = QtWidgets.QMenu("Recent Projects", window)
        window._menu_recent_projects = recent_menu
    except Exception:
        return

    # Insert into menuFile before the Exit separator so the item appears as
    # File > Recent Projects in the normal (non-ribbon) menu bar.
    inserted = False
    try:
        file_menu = window.menuFile
        exit_action = getattr(window, "actionExit_2", None)
        if exit_action is not None:
            # Find the separator that precedes Exit and insert before it.
            actions = file_menu.actions()
            target = None
            for i, a in enumerate(actions):
                if a.isSeparator():
                    # Check if Exit follows this separator
                    remaining = actions[i + 1:]
                    if any(r is exit_action for r in remaining):
                        target = a
                        break
            if target is not None:
                file_menu.insertMenu(target, recent_menu)
                inserted = True
            else:
                file_menu.insertMenu(exit_action, recent_menu)
                inserted = True
    except Exception:
        pass

    if not inserted:
        # Fallback: append to whatever menu is available
        try:
            window.menuFile.addMenu(recent_menu)
        except Exception:
            try:
                window.menuProject.addMenu(recent_menu)
            except Exception:
                pass

    try:
        set_recent_projects(window, load_recent_projects())
    except Exception:
        set_recent_projects(window, [])

    refresh_recent_projects_menu(window)

from __future__ import annotations

import json
import os
import pathlib

import chisurf
from chisurf import logging
from chisurf.gui import QtWidgets


def _recent_projects_file() -> pathlib.Path:
    try:
        return chisurf.settings.get_path("settings") / "recent_projects.json"
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


def open_recent_project(window, project_dir: str) -> None:
    try:
        path = pathlib.Path(project_dir)
    except Exception:
        return

    try:
        project_file = path / "project.json"
        if not project_file.exists():
            QtWidgets.QMessageBox.warning(
                window,
                "Invalid Project",
                "The selected folder does not contain a valid project file (project.json).",
            )
            try:
                current = list(getattr(window, "_recent_projects", []) or [])
                current = [p for p in current if os.path.normpath(p) != os.path.normpath(project_dir)]
                set_recent_projects(window, current)
                store_recent_projects(current)
                refresh_recent_projects_menu(window)
            except Exception:
                pass
            return
    except Exception:
        return

    try:
        chisurf.working_path = path
    except Exception:
        pass

    try:
        chisurf.action_controller.execute(
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
        window._current_project_dir = path
    except Exception:
        pass

    add_recent_project(window, path)


def refresh_recent_projects_menu(window) -> None:
    menu = getattr(window, "_menu_recent_projects", None)
    if menu is None:
        return
    try:
        menu.clear()
    except Exception:
        return

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
                a.triggered.connect(lambda _checked=False, pp=p: open_recent_project(window, pp))
                menu.addAction(a)
            except Exception:
                continue

    try:
        menu.addSeparator()
    except Exception:
        pass
    try:
        clear_action = QtWidgets.QAction("Clear Recent Projects", window)
        clear_action.triggered.connect(lambda _checked=False: clear_recent_projects(window))
        menu.addAction(clear_action)
    except Exception:
        pass


def init_recent_projects_menu(window) -> None:
    if getattr(window, "_menu_recent_projects", None) is not None:
        refresh_recent_projects_menu(window)
        return

    try:
        recent_menu = QtWidgets.QMenu("Recent Projects", window)
        window._menu_recent_projects = recent_menu
    except Exception:
        return

    try:
        window.menuProject.insertMenu(window.actionClose_Project, recent_menu)
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

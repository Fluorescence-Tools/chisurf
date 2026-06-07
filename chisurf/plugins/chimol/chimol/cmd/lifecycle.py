from __future__ import annotations

import copy
from typing import List

from .base import BaseCmd


class LifecycleMixin(BaseCmd):
    """Object and session lifecycle commands."""

    def _mixin_commands(self):
        """Return lifecycle command handlers."""

        return {
            "delete": self._cmd_delete,
            "reinitialize": self._cmd_reinitialize,
            "reinit": self._cmd_reinitialize,
            "copy": self._cmd_copy,
        }

    def _cmd_delete(self, args: List[str]) -> None:
        """Delete objects by id/name, or delete all loaded objects."""

        if not args:
            self._emit_error("Usage: delete <object_name|id|all> [more ...]")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        targets = [(token or "").strip() for token in args if (token or "").strip()]
        if not targets:
            self._emit_error("Usage: delete <object_name|id|all> [more ...]")
            return

        removed: list[str] = []
        failed: list[str] = []

        if any(target.lower() in ("all", "*") for target in targets):
            try:
                objects = list(viewer.list_objects())
            except Exception as exc:
                self._emit_error(f"Failed to list objects: {exc}")
                return
            for obj in objects:
                oid = str(obj.get("id", "")).strip()
                if not oid:
                    continue
                oname = str(obj.get("name") or oid).strip()
                try:
                    ok = bool(viewer.remove_object(oid))
                except Exception:
                    ok = False
                if ok:
                    removed.append(f"{oname} ({oid})")
                else:
                    failed.append(oname)
            self._named_selections.clear()
        else:
            for target in targets:
                obj = self._find_object_by_name(viewer, target)
                if obj is None:
                    if target.lower() in self._named_selections:
                        del self._named_selections[target.lower()]
                        removed.append(target)
                    else:
                        failed.append(target)
                    continue

                oid = str(obj.get("id", "")).strip()
                oname = str(obj.get("name") or oid).strip()
                if not oid:
                    failed.append(target)
                    continue

                try:
                    ok = bool(viewer.remove_object(oid))
                except Exception:
                    ok = False

                if ok:
                    removed.append(f"{oname} ({oid})")
                else:
                    failed.append(target)

        self._refresh_window_objects(window)

        if removed:
            self._emit_message("Deleted: " + ", ".join(removed))
        if failed:
            self._emit_error("Not found or failed: " + ", ".join(failed))

    def _cmd_reinitialize(self, args: List[str]) -> None:
        """Reset Chimol state similar to PyMOL ``reinitialize``."""

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        what = (args[0] if args else "everything").strip().lower()
        if what not in ("everything", "settings", "store_defaults", "original_settings", "purge_defaults"):
            self._emit_error("Usage: reinitialize [everything|settings]")
            return

        if what == "everything":
            try:
                for obj in list(viewer.list_objects()):
                    oid = str(obj.get("id", "")).strip()
                    if oid:
                        viewer.remove_object(oid)
            except Exception as exc:
                self._emit_error(f"Failed to delete objects: {exc}")
                return
            self._named_selections.clear()

        self._reset_viewer_display(viewer)
        self._refresh_window_objects(window)
        self._emit_message(f"Reinitialized {what}")

    def _cmd_copy(self, args: List[str]) -> None:
        """Create a new object by copying an existing object."""

        joined = " ".join(args).strip()
        if not joined or "," not in joined:
            self._emit_error("Usage: copy target, source")
            return

        target, source = [part.strip() for part in joined.split(",", 1)]
        if not target or not source:
            self._emit_error("Usage: copy target, source")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        src = self._find_object_by_name(viewer, source)
        if src is None:
            self._emit_error(f"Unknown source object: {source}")
            return

        src_id = str(src.get("id", "")).strip()
        if not src_id:
            self._emit_error(f"Unknown source object: {source}")
            return

        try:
            new_id = viewer.copy_object(src_id, name=target)
        except AttributeError:
            new_id = self._copy_object_fallback(viewer, src_id, target)
        except Exception as exc:
            self._emit_error(f"Failed to copy {source}: {exc}")
            return

        if not new_id:
            self._emit_error(f"Failed to copy {source}")
            return

        self._refresh_window_objects(window)
        self._emit_message(f"Copied {source} to {target}")

    def _copy_object_fallback(self, viewer, source_id: str, target: str):
        entry = getattr(viewer, "_objects", {}).get(source_id)
        if entry is None:
            return None
        if not hasattr(viewer, "_create_object"):
            return None
        new_id = viewer._create_object(name=target)
        new_entry = getattr(viewer, "_objects", {}).get(new_id)
        if new_entry is None:
            return None
        new_entry.state = copy.deepcopy(entry.state)
        new_entry.visible = bool(getattr(entry, "visible", True))
        try:
            viewer.set_active_object(new_id)
        except Exception:
            pass
        try:
            viewer._update_view()
        except Exception:
            pass
        return new_id

    def _reset_viewer_display(self, viewer) -> None:
        for method, value in (
            ("set_cartoon_visible", True),
            ("set_trace_visible", False),
            ("set_atoms_visible_all", False),
            ("set_sticks_visible", False),
            ("set_dots_visible", False),
            ("set_surface_visible", False),
            ("set_metaballs_visible", False),
            ("set_plane_visible", False),
        ):
            func = getattr(viewer, method, None)
            if callable(func):
                try:
                    func(value)
                except Exception:
                    pass

        for method, args in (
            ("clear_color_overrides", ()),
            ("set_color_mode", ("by_sequence",)),
            ("reset_view", ()),
            ("set_selected_residues", ([],)),
        ):
            func = getattr(viewer, method, None)
            if callable(func):
                try:
                    func(*args)
                except Exception:
                    pass

    def _refresh_window_objects(self, window) -> None:
        try:
            if window is not None:
                window._refresh_objects_from_viewer()
        except Exception:
            pass

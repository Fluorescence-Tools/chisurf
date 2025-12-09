from __future__ import annotations

from typing import Callable, Dict, List, Type

from .base import BaseCmd
from .loader import LoaderCommands
from .selection import SelectionMixin
from .rendering import RenderingMixin
from .measurements import MeasurementMixin

MixinType = Type[BaseCmd]


class Cmd(LoaderCommands, SelectionMixin, RenderingMixin, MeasurementMixin, BaseCmd):
    """Thin aggregator that wires together all command mixins."""

    def _builtin_commands(self) -> Dict[str, Callable[[List[str]], object]]:
        cmds: Dict[str, Callable[[List[str]], object]] = {}
        for mixin in (
            LoaderCommands,
            SelectionMixin,
            RenderingMixin,
            MeasurementMixin,
        ):
            helper = getattr(mixin, "_mixin_commands", None)
            if callable(helper):
                cmds.update(helper(self))
        cmds.update(
            {
                "help": self._cmd_help,
                "objects": self._cmd_objects,
                "get_names": self._cmd_get_names,
                "delete": self._cmd_delete,
                "quit": self._cmd_quit,
                "exit": self._cmd_quit,
            }
        )
        return cmds

    # Python convenience API (thin wrappers around internal commands)
    def help(self) -> str:
        return self._cmd_help([])

    def load(self, *paths: str) -> None:
        self._cmd_load(list(paths))

    def open(self, *paths: str) -> None:
        self._cmd_load(list(paths))

    def fetch(self, *pdb_ids: str) -> None:
        self._cmd_fetch(list(pdb_ids))

    def fetch_emdb(self, *emdb_ids: str) -> None:
        self._cmd_fetch_emdb(list(emdb_ids))

    def fetch_ihm(self, *entry_ids: str) -> None:
        self._cmd_fetch_ihm(list(entry_ids))

    def bg_color(self, color: str) -> None:
        self._cmd_bg_color([color])

    def show(self, rep: str) -> None:
        self._cmd_show([rep])

    def hide(self, rep: str) -> None:
        self._cmd_hide([rep])

    def as_(self, rep: str) -> None:
        self._cmd_as([rep])

    def center(self) -> None:
        self._cmd_center([])

    def orient(self) -> None:
        self._cmd_orient([])

    def zoom(self) -> None:
        self._cmd_zoom([])

    def reset(self) -> None:
        self._cmd_reset([])

    def color(self, mode: str) -> None:
        self._cmd_color([mode])

    def distance(self, *tokens: str) -> None:
        self._cmd_distance(list(tokens))

    def angle(self, *tokens: str) -> None:
        self._cmd_angle(list(tokens))

    def dihedral(self, *tokens: str) -> None:
        self._cmd_dihedral(list(tokens))

    def rms(self, *args: str) -> None:
        self._cmd_rms(list(args))

    def split_chains(self, prefix: str | None = None) -> None:
        args: list[str] = []
        if prefix:
            args.append(str(prefix))
        self._cmd_split_chains(args)

    def frame(self, index: int) -> None:
        self._cmd_frame([str(index)])
        self._cmd_frame_next([])

    def frame_prev(self) -> None:
        self._cmd_frame_prev([])

    def select(self, *tokens: str) -> None:
        self._cmd_select(list(tokens))

    def set(self, name: str, value: str) -> None:
        self._cmd_set([name, value])

    def objects(self) -> None:
        self._cmd_objects([])

    def get_names(self) -> None:
        self._cmd_get_names([])

    def enable(self, *tokens: str) -> None:
        self._cmd_enable(list(tokens))

    def disable(self, *tokens: str) -> None:
        self._cmd_disable(list(tokens))

    def deselect(self) -> None:
        self._cmd_deselect([])

    def quit(self) -> None:
        self._cmd_quit([])

    def exit(self) -> None:
        self._cmd_quit([])

    def delete(self, *tokens: str) -> None:
        """Delete objects by id or name (PyMOL-style delete)."""
        self._cmd_delete(list(tokens))

    # ------------------------------------------------------------------
    # Internal handlers (falling back to BaseCmd helpers for errors)
    # ------------------------------------------------------------------

    def _cmd_delete(self, args: List[str]) -> None:
        if not args:
            self._emit_error("Usage: delete <object_name|id> [more ...]")
            return

        window, viewer = self._require_window_and_viewer()
        if viewer is None:
            return

        removed: list[str] = []
        failed: list[str] = []

        for token in args:
            target = (token or "").strip()
            if not target:
                continue

            obj = self._find_object_by_name(viewer, target)
            if obj is None:
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

        # Refresh UI/store if we have a window
        try:
            if window is not None:
                window._refresh_objects_from_viewer()
        except Exception:
            pass

        if removed:
            self._emit_message("Deleted: " + ", ".join(removed))
        if failed:
            self._emit_error("Not found or failed: " + ", ".join(failed))
